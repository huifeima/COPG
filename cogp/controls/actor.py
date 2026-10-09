"""Matched physical and execution-information actor augmentation.

Both arms add the same 23-input unary side network. The physical arm is an
active deterministic projection of existing actor observations; the execution
arm reads the first 13 global and first 10 per-shepherd boundary fields.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from .features import DOG_FEATURE_NAMES, GLOBAL_FEATURE_NAMES, executor_features_from_state

SIDE_SCHEMA = "boundary-execution-or-physical-projection"
ACTOR_SCHEMA = "matched-information-actor"
VARIANTS = ("physical", "execution")
SIDE_WIDTH = 23
PROJECTION_SEED = 178913
SIDE_INITIALISATION_OFFSET = 178291
TRAINING_DECISIONS, TRAINING_STREAMS, ROLLOUT_SIZE, RESET_BLOCK = 32768, 8, 1024, 4096


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def feature_width():
    if GLOBAL_FEATURE_NAMES[13] != "commitment_active" or DOG_FEATURE_NAMES[10] != "active_task_valid":
        raise RuntimeError("Boundary feature ordering changed")
    return SIDE_WIDTH


def physical_rows(global_np, max_sheep, max_dogs, n, m, width=SIDE_WIDTH):
    """Active geometric rows using only the original policy's observation x."""
    g = np.asarray(global_np, np.float32)
    if g.ndim != 1 or len(g) != 5*max_sheep+5*max_dogs+6:
        raise ValueError("Wrong padded physical observation size")
    if not 0 < n <= max_sheep or not 0 < m <= max_dogs or width < 1:
        raise ValueError("Invalid active population or projection width")
    sheep = g[:4*max_sheep].reshape(max_sheep, 4)[:n]
    dogs = g[5*max_sheep:5*max_sheep+4*max_dogs].reshape(max_dogs, 4)[:m]
    goal, extras = g[-6:-4], g[-4:]
    pooled = np.concatenate([sheep.mean(0), sheep.std(0), dogs.mean(0), goal, extras,
                             np.asarray([n/max_sheep, m/max_dogs], np.float32)])
    base = np.stack([np.concatenate([dog, pooled, sheep[:, :2].mean(0)-dog[:2], goal-dog[:2]]) for dog in dogs])
    projection = np.random.default_rng(PROJECTION_SEED).normal(size=(base.shape[-1], width)).astype(np.float32)/math.sqrt(base.shape[-1])
    result = np.tanh(base @ projection).astype(np.float32)
    if not np.isfinite(result).all(): raise ValueError("Nonfinite physical side feature")
    return result


def execution_rows(feature, m):
    glob = np.asarray(feature["global"], np.float32)[:13]
    dogs = np.asarray(feature["per_dog"], np.float32)[:m, :10]
    mask = np.asarray(feature["dog_mask"], np.float32)[:m]
    if glob.shape != (13,) or dogs.shape != (m, 10) or mask.shape != (m,) or not np.all(mask == 1):
        raise ValueError("Execution side-feature active-dog mask mismatch")
    result = np.concatenate([np.broadcast_to(glob, (m, len(glob))), dogs], -1)
    if not np.isfinite(result).all(): raise ValueError("Nonfinite execution side feature")
    return result


def side_features(env, co, ctx, global_np, max_sheep, max_dogs, variant):
    m, n = int(env.num_dogs), int(env.num_sheep)
    if variant == "execution": return execution_rows(executor_features_from_state(env, co, ctx), m)
    if variant == "physical": return physical_rows(global_np, max_sheep, max_dogs, n, m)
    raise ValueError("Unknown matched actor arm")


def enrich_snapshot(snapshot, env, co, ctx):
    """Return a new immutable TaskSnapshot; never alter the captured packet."""
    payload = snapshot.data
    raw = executor_features_from_state(env, co, ctx)
    payload["executor_features"] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in raw.items()}
    payload["actor_side_schema"] = SIDE_SCHEMA
    return type(snapshot)(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))


def rows_from_state(policy, state):
    payload = state.snapshot.data
    if payload.get("actor_side_schema") != SIDE_SCHEMA:
        raise RuntimeError("Training snapshot lacks immutable execution features")
    n, m = state.structure.active_sheep, state.structure.active_dogs
    if policy.side_variant == "execution": return execution_rows(payload["executor_features"], m)
    return physical_rows(state.global_np, int(payload["max_sheep"]), int(payload["max_dogs"]), n, m, policy.side_width)


def attach_side_encoder(policy, variant, width=SIDE_WIDTH):
    """Install same side architecture and random stream for both arms."""
    if variant not in VARIANTS or int(width) != SIDE_WIDTH: raise ValueError("Invalid actor control")
    if hasattr(policy, "side_encoder"): raise RuntimeError("Actor augmentation already installed")
    input_dim = 3*int(policy.hidden)+12
    # Fork protects the base policy's later categorical action sampling RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(torch.initial_seed()+SIDE_INITIALISATION_OFFSET)
        policy.side_encoder = nn.Sequential(nn.Linear(width,64), nn.Tanh(), nn.Linear(64,64), nn.Tanh())
        policy.side_score = nn.Sequential(nn.Linear(input_dim+128,64), nn.Tanh(), nn.Linear(64,1))
    policy.side_variant, policy.side_width, policy.side_schema = variant, width, SIDE_SCHEMA
    policy._bound_actor_side = None

    def hook(module, args, output):
        x, raw = args[0], policy._bound_actor_side
        if raw is None: raise RuntimeError("Bind immutable actor side features before inference")
        raw = torch.as_tensor(raw, dtype=x.dtype, device=x.device)
        if x.ndim not in (3,4) or raw.ndim != x.ndim-1 or raw.shape[:-1] != x.shape[:-2]:
            raise RuntimeError(f"Actor side shape mismatch: {tuple(raw.shape)} vs {tuple(x.shape)}")
        local = policy.side_encoder(raw)
        pooled = local.mean(dim=-2, keepdim=True).expand_as(local)
        side = torch.cat([local, pooled], -1).unsqueeze(-2).expand(*x.shape[:-1], 128)
        return output + policy.side_score(torch.cat([x, side], -1))
    policy._actor_side_hook = policy.unary_net.register_forward_hook(hook)
    return policy


@contextmanager
def actor_batch_context(policy, states):
    """Explicit training hook; no global module replacement or dynamic exec."""
    if not hasattr(policy, "side_variant"):
        yield
        return
    previous = policy._bound_actor_side
    policy._bound_actor_side = np.stack([rows_from_state(policy, state) for state in states])
    try: yield
    finally: policy._bound_actor_side = previous


def refresh_actor(session):
    policy = session.policy
    if not hasattr(policy, "side_variant"): raise ValueError("No matched actor bound")
    policy._bound_actor_side = side_features(session.env, session.co, session.ctx,
        session.ob.global_state_vector(session.env), session.ob.max_sheep, session.ob.max_dogs, policy.side_variant)


def configure_learner(learner, variant, actor_lr=3e-4):
    """Fresh-learner extension: critic and PPO losses remain unchanged.

    Worker capture must call enrich_snapshot(), and the batched model forward
    must use actor_batch_context(). The shared COGP critic is bound separately
    at evaluation, rather than the actor's newly trained PPO critic.
    """
    if int(learner.decisions) != 0: raise ValueError("Matched actors require fresh random initialisation")
    attach_side_encoder(learner.policy, variant)
    learner.policy.to(learner.device)
    learner.actor_opt = torch.optim.Adam(learner.policy.parameters(), lr=actor_lr, eps=1e-5)
    return learner


def save_actor(policy, path, *, seed, decisions, metadata=None):
    if decisions != TRAINING_DECISIONS: raise ValueError("Actor controls save the exact final budget only")
    if not hasattr(policy, "side_variant"): raise ValueError("No matched actor augmentation")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(schema=ACTOR_SCHEMA, side_schema=SIDE_SCHEMA, variant=policy.side_variant,
                   side_width=policy.side_width, seed=int(seed), training_decisions=int(decisions),
                   state_dict={k:v.detach().cpu() for k,v in policy.state_dict().items()}, metadata=metadata or {},
                   initialization="random", planning_critic="shared-frozen-parent")
    temporary = path.with_suffix(".tmp.pt")
    torch.save(payload, temporary); temporary.replace(path)
    receipt = {k:payload[k] for k in ("schema", "side_schema", "variant", "side_width", "seed", "training_decisions", "initialization", "planning_critic")}
    receipt["sha256"] = sha256(path)
    path.with_suffix(".json").write_text(json.dumps(receipt, sort_keys=True, indent=2)+"\n")
    return receipt


def exact_tensor_key_map(state, prefix_map, expected_state):
    """Explicit bijective checkpoint conversion, followed by strict loading.

    prefix_map is supplied by an import manifest, never guessed from a file
    name. Every tensor must map once to one known key with identical shape.
    """
    result = {}
    for key, value in state.items():
        matches = [prefix for prefix in prefix_map if key.startswith(prefix)]
        if len(matches) > 1: raise ValueError("Ambiguous tensor prefix mapping")
        dest = prefix_map[matches[0]]+key[len(matches[0]):] if matches else key
        if dest in result or dest not in expected_state: raise ValueError("Unknown or duplicate mapped tensor key: "+dest)
        if tuple(value.shape) != tuple(expected_state[dest].shape): raise ValueError("Mapped tensor shape mismatch: "+dest)
        result[dest] = value
    if set(result) != set(expected_state): raise ValueError("Mapped checkpoint has missing tensors")
    return result


def load_actor(path, policy_factory: Callable, device="cpu"):
    path = Path(path)
    receipt = json.loads(path.with_suffix(".json").read_text())
    before = sha256(path)
    if receipt.get("sha256") != before: raise RuntimeError("Actor checkpoint checksum mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if sha256(path) != before: raise RuntimeError("Actor changed during loading")
    for key in ("schema", "side_schema", "variant", "side_width", "seed", "training_decisions", "initialization", "planning_critic"):
        if payload.get(key) != receipt.get(key): raise RuntimeError("Actor receipt mismatch: "+key)
    if payload.get("schema") != ACTOR_SCHEMA or payload.get("side_schema") != SIDE_SCHEMA or payload.get("training_decisions") != TRAINING_DECISIONS:
        raise RuntimeError("Wrong actor kind or incomplete final budget")
    policy = policy_factory()
    attach_side_encoder(policy, payload["variant"], payload["side_width"])
    policy.load_state_dict(payload["state_dict"], strict=True)
    return policy.to(device).eval().requires_grad_(False), receipt


def bind_actor(session, loaded):
    """Replace actor only; retain the shared frozen critic and normalization."""
    actor, receipt = loaded
    session.policy = actor
    refresh_actor(session)
    return receipt


def episode_seed(seed, block, worker, episode):
    """Disjoint paired actor-training world addresses from the recorded protocol."""
    if not (201 <= seed <= 208 and 0 <= block < 64 and 0 <= worker < TRAINING_STREAMS and 0 <= episode < 20000):
        raise ValueError("Actor training seed address out of range")
    return 178100000000000+(seed-201)*10000000000+block*1000000+worker*20000+episode
