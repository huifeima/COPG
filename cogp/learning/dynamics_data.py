"""Frozen-parent acceleration supervision on explicitly addressed new worlds.

The mixture and decision budget follow the stated experiment protocol. World
addresses and the random sampler below are explicit new-project definitions;
unavailable historical sampler seeds are not reconstructed or guessed.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from cogp.coordination.ranking import exact_forward, prepare
from cogp.coordination.snapshot import task_key
from cogp.dynamics import ForceArchive, sha256


CANONICAL_SCENARIOS = ("compact", "single_outlier", "double_outlier", "split")
BEHAVIOR_MODES = ("greedy", "categorical", "uniform")
DEFAULT_MIXTURE = (.7, .2, .1)
EXTRA_DECISIONS = 2048


def policy_digest(policy: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


@contextmanager
def observe_true_forces(env: Any, recorder: ForceArchive,
                        episode_id: int) -> Iterator[None]:
    """Observe the actual capped force return without replacing force values.

    The environment calls this routine after shepherd movement and before each
    sheep integration substep. Restoration preserves whether the instance had
    an override originally, including on exceptions.
    """
    name = "_compute_sheep_forces"
    original = getattr(env, name)
    had_override = name in vars(env)
    override = vars(env).get(name)

    def observe():
        returned = original()
        if not isinstance(returned, tuple) or len(returned) != 3:
            raise RuntimeError("True force routine must return (clipped_accel, repulsive_load, graph)")
        clipped = np.asarray(returned[0])
        maximum = float(env.cfg.sheep_max_accel)
        if not np.isfinite(clipped).all() or np.any(np.linalg.norm(clipped, axis=-1) > maximum + 1e-3):
            raise RuntimeError("Force supervision is not finite capped acceleration")
        recorder.observe(env, clipped, int(episode_id))
        return returned

    setattr(env, name, observe)
    try:
        yield
    finally:
        if had_override:
            setattr(env, name, override)
        else:
            delattr(env, name)


def choose_assignment(policy: Any, snapshot: Any, rng: np.random.Generator,
                      mixture=DEFAULT_MIXTURE) -> tuple[Any, dict[str, Any]]:
    """Sample from the same exact feasible M=2 support for all three modes."""
    probabilities = np.asarray(mixture, dtype=np.float64)
    if probabilities.shape != (3,) or not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("Behavior mixture must contain three finite nonnegative probabilities")
    if not np.isclose(probabilities.sum(), 1., rtol=0., atol=1e-12):
        raise ValueError("Behavior mixture must sum to one")
    state = prepare(snapshot)
    if int(state.structure.active_dogs) != 2 or not state.specs:
        raise ValueError("Dynamics collection requires a nonempty exact M=2 feasible support")
    with torch.no_grad():
        scores, _ = exact_forward(policy, state)
    values = scores.detach().cpu().double().numpy()
    if values.shape != (len(state.specs),) or not np.isfinite(values).all():
        raise RuntimeError("Nonfinite or inconsistent parent assignment scores")
    mode = int(rng.choice(3, p=probabilities))
    if mode == 0:
        selected = int(np.argmax(values))
    elif mode == 1:
        weights = np.exp(values - values.max())
        selected = int(rng.choice(len(values), p=weights / weights.sum()))
    else:
        selected = int(rng.integers(len(values)))
    metadata = {
        "mode": BEHAVIOR_MODES[mode], "candidate_index": selected,
        "candidate_count": len(values), "candidate_digest": state.candidate_hash,
        "selected_assignment": task_key(state.specs[selected]),
        "selected_logit": float(values[selected]),
        "native_index": int(np.argmax(values)),
    }
    return state.specs[selected], metadata


def collect(session: Any, out: str | Path, seed: int, *, world_seed_origin: int,
            decisions: int = EXTRA_DECISIONS, sampling_seed: int | None = None,
            scenario_seed: int | None = None, reservoir_seed: int | None = None,
            mixture=DEFAULT_MIXTURE, capacity: int = 60000, stride: int = 12,
            parent_checkpoint: str | Path | None = None,
            occupied_world_seeds=()) -> dict[str, Any]:
    """Collect real labels using a frozen HCCG Session, never candidate branches.

    ``session`` provides env, policy, snapshot, reset(case), and step(spec).
    Its step result is a mapping with done/reward/raw/duration/primitive_steps.
    A different budget/capacity/stride is explicitly marked test-only.
    """
    if type(seed) is not int or seed < 0 or type(world_seed_origin) is not int or world_seed_origin < 0:
        raise ValueError("Parent seed and new-world seed origin must be nonnegative integers")
    if type(decisions) is not int or not 0 < decisions <= 100000:
        raise ValueError("Decision budget must be an integer in [1,100000]")
    if getattr(session.policy, "mainline_method", None) != "HCCG":
        raise ValueError("The shared dynamics archive must be collected by its frozen HCCG parent")
    checkpoint_hash = sha256(parent_checkpoint) if parent_checkpoint else None
    commitment = getattr(getattr(session, "co", None), "max_option_high_steps", None)
    target = Path(out)
    if target.suffix != ".npz":
        target = target / "acceleration.npz"
    paths = {
        "dataset": target,
        "archive_manifest": target.with_suffix(".manifest.json"),
        "collection_report": target.with_suffix(".collection.json"),
        "decisions": target.with_suffix(".decisions.jsonl"),
        "episodes": target.with_suffix(".episodes.json"),
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Refusing to overwrite an existing dynamics collection; choose a new output")
    target.parent.mkdir(parents=True, exist_ok=True)
    sampling_seed = seed + 700000003 if sampling_seed is None else int(sampling_seed)
    scenario_seed = seed + 800000009 if scenario_seed is None else int(scenario_seed)
    reservoir_seed = seed + 900000011 if reservoir_seed is None else int(reservoir_seed)
    if min(sampling_seed, scenario_seed, reservoir_seed) < 0:
        raise ValueError("Sampler seeds must be nonnegative")
    action_rng = np.random.default_rng(sampling_seed)
    reset_rng = np.random.default_rng(scenario_seed)
    archive = ForceArchive(reservoir_seed, capacity=capacity, stride=stride)
    supplied = set(int(value) for value in occupied_world_seeds)
    counts, decision_scenarios, episode_scenarios = Counter(), Counter(), Counter()
    episodes, decision_rows = [], []
    original_policy_mode = bool(session.policy.training)
    original_weights = policy_digest(session.policy)
    session.policy.eval()
    count = 0
    episode_id = 0
    try:
        while count < decisions:
            # This address is a declared clean-project sampler, not historical replay.
            world_seed = world_seed_origin + seed * 1_000_000 + episode_id
            if world_seed in supplied:
                raise ValueError(f"New data world seed overlaps an explicitly supplied reset manifest: {world_seed}")
            scenario = str(reset_rng.choice(CANONICAL_SCENARIOS))
            case = {
                "kind": "canonical", "scenario": scenario, "seed": world_seed,
                "dogs": 2, "sheep": 15, "case_id": f"dynamics_s{seed}_e{episode_id:05d}",
                "family": "dynamics_training", "physics_multiplier": 1.,
            }
            session.reset(case)
            env = session.env
            if int(env.num_dogs) != 2 or int(env.num_sheep) != 15:
                raise RuntimeError("Data collection reset did not produce M=2,N=15")
            episode_scenarios[scenario] += 1
            initial_primitive = int(env.primitive_step)
            episode_decisions = 0
            done = False
            with observe_true_forces(env, archive, episode_id):
                while count < decisions and not done:
                    if session.env is not env:
                        raise RuntimeError("Session replaced its environment during a data episode")
                    spec, action = choose_assignment(session.policy, session.snapshot, action_rng, mixture)
                    before = int(env.primitive_step)
                    transition = session.step(spec)
                    if not isinstance(transition, dict):
                        raise TypeError("Session.step must return a transition mapping")
                    after = int(env.primitive_step)
                    if after <= before or int(transition["duration"]) <= 0:
                        raise RuntimeError("A collected joint decision must advance real physical time")
                    done = bool(transition["done"])
                    counts[action["mode"]] += 1
                    decision_scenarios[scenario] += 1
                    decision_rows.append({
                        "decision": count, "episode_id": episode_id, "world_seed": world_seed,
                        "scenario": scenario, **action, "primitive_start": before,
                        "primitive_end": after, "reward_checks": int(transition["duration"]),
                        "reward": float(transition["reward"]), "raw_reward": float(transition["raw"]),
                        "done": done,
                    })
                    count += 1
                    episode_decisions += 1
            metrics = env.metrics()
            episodes.append({**case, "episode_id": episode_id, "decisions": episode_decisions,
                             "primitive_steps": int(env.primitive_step) - initial_primitive,
                             "terminated": done, "truncated_by_collection_budget": not done,
                             "success": bool(float(metrics["success"]) > .5)})
            episode_id += 1
        if policy_digest(session.policy) != original_weights:
            raise RuntimeError("Frozen HCCG weights changed during dynamics data collection")
        if archive.force_calls == 0:
            raise RuntimeError("No true sheep-force calls observed; label collection hook was not used")
        archive_report = archive.save(target)
        report = {
            "schema": "dynamics-data-collection", "status": "complete",
            "parent": "HCCG", "training_seed": seed, "policy_tensor_sha256": original_weights,
            "parent_checkpoint_sha256": checkpoint_hash,
            "decisions": count, "decision_budget": decisions,
            "requested_execution_checks": commitment,
            "population": {"sheep": 15, "shepherds": 2},
            "canonical_scenarios": list(CANONICAL_SCENARIOS),
            "reset_scenario_probabilities": {name: .25 for name in CANONICAL_SCENARIOS},
            "behavior_mixture": dict(zip(BEHAVIOR_MODES, map(float, mixture))),
            "behavior_counts": {name: counts[name] for name in BEHAVIOR_MODES},
            "decisions_by_scenario": {name: decision_scenarios[name] for name in CANONICAL_SCENARIOS},
            "episodes_by_scenario": {name: episode_scenarios[name] for name in CANONICAL_SCENARIOS},
            "episodes": len(episodes), "archive": archive_report,
            "sampler_seeds": {"action": sampling_seed, "scenario": scenario_seed,
                              "reservoir": reservoir_seed},
            "world_seed_origin": world_seed_origin,
            "world_address_rule": "world_seed_origin + training_seed * 1000000 + episode_id",
            "world_seeds": [episode["seed"] for episode in episodes],
            "exact_historical_replay": False,
            "historical_sampler_status": "original reset/action/reservoir sampler identities not recovered",
            "independence_scope": "checked only against explicitly supplied occupied reset seeds",
            "paper_protocol_budget": decisions == EXTRA_DECISIONS and capacity == 60000 and stride == 12
                                    and tuple(mixture) == DEFAULT_MIXTURE and commitment == 64,
            "test_only": decisions != EXTRA_DECISIONS or capacity != 60000 or stride != 12
                         or tuple(mixture) != DEFAULT_MIXTURE or commitment != 64,
            "deployed_forecasting_used": False,
            "label_access_scope": "supervised training data only",
            "implementation_sha256": sha256(__file__),
        }
        paths["decisions"].write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in decision_rows))
        paths["episodes"].write_text(json.dumps(episodes, indent=2, allow_nan=False) + "\n")
        report["outputs_sha256"] = {key: sha256(path) for key, path in paths.items()
                                     if key != "collection_report"}
        paths["collection_report"].write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return report
    finally:
        session.policy.train(original_policy_mode)
