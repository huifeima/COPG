"""Matched offline physical/execution-state terminal critics.

Each arm receives exactly the same paused endpoint snapshots, labels, splits,
initial weights, minibatches, and update budget. Labels complete the original
unfinished commitment, then use frozen greedy HCCG until episode termination.
These are supervised value refits, not planner-aware policy evaluations.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

SCHEMA = "matched-information-critic"
REWARD_SCALE = 0.10
MODEL_KIND = "pooled-physical-backbone-matched-side"
FIT_SEED_OFFSET = 178000
DOG_SIDE_DIM, GLOBAL_SIDE_DIM = 43, 21


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n")
    os.replace(temporary, path)


def _cfg(config):
    p = dict(decisions=[0, 2], horizons=[12, 64], E=64, candidate_limit=4,
             updates=5000, batch_size=128, hidden=64, lr=3e-4,
             weight_decay=1e-4, save_every=500)
    p.update(config.get("critic", {}))
    for key in ("E", "candidate_limit", "updates", "batch_size", "hidden", "save_every"):
        if isinstance(p[key], bool) or int(p[key]) != p[key] or p[key] < 1:
            raise ValueError("Invalid critic "+key)
    if p["candidate_limit"] > 4: raise ValueError("Only the fixed first four candidates are permitted")
    if not p["decisions"] or sorted(set(p["decisions"])) != p["decisions"] or any(d < 0 or int(d) != d for d in p["decisions"]):
        raise ValueError("Invalid decision indices")
    if not p["horizons"] or sorted(set(p["horizons"])) != p["horizons"] or any(h < 1 or h > p["E"] or int(h) != h for h in p["horizons"]):
        raise ValueError("Invalid label horizons")
    if p["lr"] <= 0 or p["weight_decay"] < 0: raise ValueError("Invalid optimizer settings")
    return p


def endpoint_tail_return(prefix_reward, prefix_duration, full_reward, full_duration,
                         continuation_return, gamma, terminal=False):
    """Exact future scaled return at a paused endpoint, no critic bootstrap.

    Prefix and full reward are already-discounted, UNSCALED rewards from the
    same start and same commitment. The continuation return is scaled and
    discounted from the true option endpoint. Algebra is performed in float64.
    """
    vals = [prefix_reward, full_reward, continuation_return, gamma]
    if not np.isfinite(vals).all() or not 0 < gamma <= 1:
        raise ValueError("Nonfinite return or invalid gamma")
    if not 0 <= prefix_duration <= full_duration:
        raise ValueError("Invalid prefix/full duration")
    if terminal:
        if prefix_duration != full_duration:
            raise ValueError("Terminal prefix cannot precede full commitment endpoint")
        return 0.0
    discount = float(gamma) ** int(prefix_duration)
    if discount <= 1e-250:
        raise ValueError("Discount underflow: cannot recover endpoint tail")
    remainder = REWARD_SCALE * (float(full_reward) - float(prefix_reward)) / discount
    result = remainder + float(gamma) ** (int(full_duration) - int(prefix_duration)) * float(continuation_return)
    if not math.isfinite(result):
        raise ValueError("Nonfinite endpoint label")
    return result


def _expanded_physical(x, width):
    """Active deterministic x-only side control; no zero-filled dummy branch."""
    x = np.asarray(x, np.float32)
    if x.shape[-1] == 0 or width < 1:
        raise ValueError("Empty physical side representation")
    bank = [x, x * x, np.sin(x), np.cos(x), np.tanh(x)]
    order = 2
    while sum(a.shape[-1] for a in bank) < width:
        bank.extend((np.sin(order * x), np.cos(order * x)))
        order += 1
    return np.concatenate(bank, axis=-1)[..., :width].astype(np.float32)


def physical_features(env):
    """Pure geometric/velocity input, without clocks, phases or memory."""
    s = np.asarray(env.sheep_pos, np.float64)
    sv = np.asarray(env.sheep_vel, np.float64)
    d = np.asarray(env.dog_pos, np.float64)
    dv = np.asarray(env.dog_vel, np.float64)
    goal = np.asarray(env.goal, np.float64)
    if s.ndim != 2 or d.ndim != 2 or s.shape[1] != 2 or d.shape[1] != 2 or not len(s) or not len(d):
        raise ValueError("Invalid physical entities")
    size = float(env.cfg.world_size)
    sspeed = max(1e-6, float(getattr(env.cfg, "sheep_max_speed", 1.0)))
    dspeed = max(1e-6, float(getattr(env.cfg, "dog_max_speed", 1.0)))
    center = s.mean(0)
    sf = np.concatenate((s / size, sv / sspeed, (goal - s) / size, (s - center) / size), 1)
    df = np.concatenate((d / size, dv / dspeed, (goal - d) / size, (d - center) / size), 1)
    gf = np.r_[goal / size, center / size, s.std(0) / size,
               np.mean(sv, 0) / sspeed, np.mean(dv, 0) / dspeed,
               len(s) / 15.0, len(d) / 2.0]
    out = {"sheep": sf.astype(np.float32), "dogs": df.astype(np.float32),
           "global": gf.astype(np.float32), "sheep_mask": np.ones(len(s), np.float32),
           "dog_mask": np.ones(len(d), np.float32)}
    if not all(np.isfinite(x).all() for x in out.values()):
        raise ValueError("Nonfinite physical feature")
    return out


def encode_endpoint(session, snapshot=None):
    """Return both matched inputs from one read-only endpoint state.

    Membership masks are converted to geometric centroids and fractions, never
    interpreted as ordinal sheep IDs. Side inputs have exactly equal dimension.
    This is a prespecified execution augmentation, not a Markov-sufficiency claim.
    """
    from .features import executor_features, features
    local = snapshot.session if snapshot is not None else session
    phys = physical_features(local.env)
    ex = features(snapshot) if snapshot is not None else executor_features(local)
    sd = np.asarray(ex["per_dog"], np.float32)
    sg = np.asarray(ex["global"], np.float32)
    if sd.ndim != 2 or sd.shape[0] != len(phys["dogs"]) or sg.ndim != 1:
        raise ValueError("Execution feature shape mismatch")
    # Preserve explicit target/component membership with a shared geometric map.
    sheep = phys["sheep"]
    summaries = []
    for key in ("component_mask", "target_mask"):
        mask = np.asarray(ex[key], np.float32)
        if mask.shape != (len(sd), len(sheep)):
            raise ValueError("Execution membership shape mismatch: " + key)
        count = mask.sum(-1, keepdims=True)
        centroid = mask @ sheep[:, :4] / np.maximum(count, 1.0)
        summaries.append(np.concatenate((centroid, count / len(sheep), (count > 0).astype(np.float32)), -1))
    sd = np.concatenate((sd, *summaries), -1)
    out = {**phys, "execution_dogs": sd, "execution_global": sg,
           "physical_dogs": _expanded_physical(phys["dogs"], sd.shape[-1]),
           "physical_global": _expanded_physical(phys["global"], len(sg))}
    if not all(np.isfinite(v).all() for v in out.values()):
        raise ValueError("Nonfinite endpoint feature")
    return {k: np.array(v, copy=True) for k, v in out.items()}, str(ex["schema"])


def _make_model(dog_side_dim, global_side_dim, hidden=64):
    import torch
    from torch import nn

    class MatchedCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.sheep = nn.Sequential(nn.Linear(8, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
            self.dogs = nn.Sequential(nn.Linear(8, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
            self.side_dogs = nn.Sequential(nn.Linear(dog_side_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
            self.side_global = nn.Sequential(nn.Linear(global_side_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
            self.head = nn.Sequential(nn.Linear(7*hidden+12, hidden), nn.SiLU(), nn.Linear(hidden, 1))

        @staticmethod
        def pool(x, mask):
            weights = mask.unsqueeze(-1)
            mean = (x*weights).sum(1) / weights.sum(1).clamp_min(1)
            maximum = x.masked_fill(weights <= 0, -torch.inf).amax(1)
            maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
            return torch.cat((mean, maximum), -1)

        def forward(self, batch, arm):
            pooled = (self.pool(self.sheep(batch["sheep"]), batch["sheep_mask"]),
                      self.pool(self.dogs(batch["dogs"]), batch["dog_mask"]),
                      self.pool(self.side_dogs(batch[arm+"_dogs"]), batch["dog_mask"]),
                      self.side_global(batch[arm+"_global"]), batch["global"])
            return self.head(torch.cat(pooled, -1)).squeeze(-1)

    return MatchedCritic()


FEATURE_KEYS = ("sheep", "dogs", "global", "sheep_mask", "dog_mask", "physical_dogs", "physical_global", "execution_dogs", "execution_global")


def _load_split(config, seed, split):
    root = Path(config["out"]) / "critic_labels" / f"s{seed}" / split
    files = sorted(root.glob("*.json"))
    if not files:
        raise RuntimeError("Missing critic label split: " + str(root))
    groups, manifest, worlds, schema = [], [], set(), None
    for file in files:
        r = json.loads(file.read_text())
        data = file.with_suffix(".npz")
        ident = r.get("identity", {})
        if r.get("status") != "COMPLETE" or ident.get("schema") != SCHEMA or ident.get("study_hash") != config["study_hash"] or ident.get("seed") != seed or ident.get("split") != split:
            raise RuntimeError("Critic label identity mismatch: " + str(file))
        if _sha(data) != r.get("dataset_sha256"):
            raise RuntimeError("Critic label hash mismatch")
        this_schema = r.get("feature_schema")
        if schema is None: schema = this_schema
        if schema != this_schema: raise RuntimeError("Mixed execution feature schemas")
        case = ident["case"]
        if case.get("kind") != "canonical" or case.get("dogs") != 2 or case.get("sheep") != 15:
            raise RuntimeError("Critic labels contain non-training populations")
        world = (case["kind"], case["scenario"], case["seed"])
        if world in worlds: raise RuntimeError("Duplicate critic training world")
        worlds.add(world)
        with np.load(data, allow_pickle=False) as z:
            block = {k: np.array(z[k], copy=True) for k in (*FEATURE_KEYS, "target")}
        if not all(np.isfinite(a).all() for a in block.values()):
            raise RuntimeError("Nonfinite critic dataset")
        if len(block["target"]) != r["samples"]: raise RuntimeError("Critic sample receipt mismatch")
        _validate_arrays(block)
        groups.append(block)
        manifest.append(dict(case=ident["case"], sha256=r["dataset_sha256"], samples=r["samples"]))
    return {k: np.concatenate([g[k] for g in groups], 0) for k in groups[0]}, manifest, worlds, schema


def _target_norm(y):
    y = np.asarray(y, np.float64)
    if not len(y) or not np.isfinite(y).all(): raise ValueError("Invalid target normalization data")
    return {"mean": float(y.mean()), "std": max(float(y.std()), 1e-3)}


def _as_tensors(arrays, device):
    import torch
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in arrays.items()}


def fit_critics(config, job):
    """Identical labels/split/init/minibatches/budget; final checkpoints only."""
    import torch
    p = _cfg(config)
    seed = int(job["seed"])
    device = str(job.get("device", "cpu"))
    train, train_manifest, train_worlds, schema = _load_split(config, seed, "train")
    val, val_manifest, val_worlds, val_schema = _load_split(config, seed, "validation")
    if train["execution_dogs"].shape[-1] != DOG_SIDE_DIM or train["execution_global"].shape[-1] != GLOBAL_SIDE_DIM:
        raise RuntimeError("Matched critic feature dimensions differ from protocol")
    if train_worlds & val_worlds:
        raise RuntimeError("Training/validation world overlap")
    if schema != val_schema: raise RuntimeError("Train/validation feature schema mismatch")
    norm = _target_norm(train["target"])
    dims = dict(dog_side_dim=int(train["execution_dogs"].shape[-1]),
                global_side_dim=int(train["execution_global"].shape[-1]), hidden=int(p["hidden"]))
    for data in (train, val):
        if data["physical_dogs"].shape != data["execution_dogs"].shape or data["physical_global"].shape != data["execution_global"].shape:
            raise RuntimeError("Matched critic dimensions differ")
    identity = dict(schema=SCHEMA, study_hash=config["study_hash"], seed=seed, model_kind=MODEL_KIND,
                    feature_schema=schema, dims=dims, train_manifest=train_manifest,
                    validation_manifest=val_manifest, config=p, target_normalization=norm,
                    checkpoint_selection="final_update", paired_batch_seed=seed+FIT_SEED_OFFSET,
                    physical_side="deterministic x-only polynomial/trigonometric expansion",
                    executor_side="executor fields plus geometric task-membership summaries")
    directory = Path(config["out"]) / "critics" / f"s{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / "complete.json"
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old.get("identity") != identity: raise RuntimeError("Changed critic fit identity")
        for arm, item in old["checkpoints"].items():
            if _sha(item["path"]) != item["sha256"]: raise RuntimeError("Critic checkpoint changed: " + arm)
        return {**old, "reused": True}
    tensors = _as_tensors(train, device)
    val_tensors = _as_tensors(val, device)
    checkpoints, metrics, parameters = {}, {}, {}
    for arm in ("physical", "execution"):
        torch.manual_seed(seed+FIT_SEED_OFFSET)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed+FIT_SEED_OFFSET)
        model = _make_model(**dims).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(p["lr"]), weight_decay=float(p["weight_decay"]))
        parameters[arm] = sum(x.numel() for x in model.parameters())
        rng = np.random.default_rng(seed+FIT_SEED_OFFSET)
        progress = directory / (arm + ".training.pt")
        start_update = 0
        if progress.exists():
            saved = torch.load(progress, map_location=device, weights_only=True)
            if saved.get("identity_hash") != _digest(identity): raise RuntimeError("Changed partial critic training identity")
            model.load_state_dict(saved["state_dict"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            rng.bit_generator.state = saved["rng_state"]
            start_update = int(saved["update"])
        model.train()
        for update in range(start_update, int(p["updates"])):
            ids = rng.integers(0, len(train["target"]), size=int(p["batch_size"]))
            idx = torch.as_tensor(ids, dtype=torch.long, device=device)
            batch = {k: v[idx] for k, v in tensors.items()}
            targets = (batch["target"] - norm["mean"]) / norm["std"]
            prediction = model(batch, arm)
            loss = torch.mean((prediction-targets)**2)
            if not bool(torch.isfinite(loss)): raise RuntimeError("Nonfinite critic loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if (update+1) % int(p["save_every"]) == 0 or update+1 == int(p["updates"]):
                tmp = progress.with_suffix(".tmp.pt")
                torch.save(dict(identity_hash=_digest(identity), update=update+1,
                                state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                                rng_state=rng.bit_generator.state), tmp)
                os.replace(tmp, progress)
        model.eval()
        with torch.no_grad():
            def evaluate(batch):
                pred = model(batch, arm) * norm["std"] + norm["mean"]
                err = pred-batch["target"]
                return dict(mae=float(err.abs().mean().cpu()), rmse=float(err.square().mean().sqrt().cpu()),
                            signed_bias=float(err.mean().cpu()), samples=len(err))
            metrics[arm] = {"train": evaluate(tensors), "validation": evaluate(val_tensors)}
        path = directory / (arm + ".pt")
        payload = dict(schema=SCHEMA, model_kind=MODEL_KIND, arm=arm, dims=dims,
                       feature_schema=schema, state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                       target_normalization=norm, identity_hash=_digest(identity),
                       study_hash=config["study_hash"], seed=seed, parameters=parameters[arm],
                       final_update=int(p["updates"]), metrics=metrics[arm],
                       reward_scale=REWARD_SCALE, target="same-commitment-then-frozen-HCCG-MC")
        tmp = path.with_suffix(".tmp.pt")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        checkpoints[arm] = dict(path=str(path), sha256=_sha(path))
        # Per-model receipt means a checkpoint must not be silently mixed with
        # another normalization/schema during deployment.
        _write(path.with_suffix(".json"), dict(schema=SCHEMA, sha256=_sha(path), arm=arm,
                                               identity_hash=_digest(identity), target_normalization=norm,
                                               feature_schema=schema, study_hash=config["study_hash"]))
    if parameters["physical"] != parameters["execution"]:
        raise RuntimeError("Matched critics differ in parameter count")
    result = dict(identity=identity, status="COMPLETE", checkpoints=checkpoints,
                  parameters=parameters, metrics=metrics, train_worlds=len(train_worlds),
                  validation_worlds=len(val_worlds), train_samples=len(train["target"]),
                  validation_samples=len(val["target"]))
    _write(receipt, result)
    return {**result, "reused": False}


def _array_content_digest(arrays):
    h=hashlib.sha256()
    for key in sorted(arrays):
        value=np.ascontiguousarray(arrays[key])
        h.update(key.encode()); h.update(str(value.dtype).encode()); h.update(str(value.shape).encode()); h.update(value.tobytes())
    return h.hexdigest()


def write_label_archive(root, identity, rows, *, metadata=None):
    """Commit one training-world archive with immutable per-row provenance.

    rows contain both matched feature arms and one shared Monte Carlo target.
    The runtime is responsible for collecting the true original commitment and
    the specified greedy frozen-HCCG continuation; no value model is used here.
    """
    if identity.get("schema") != SCHEMA or identity.get("split") not in ("train", "validation"):
        raise ValueError("Labels require a matched-critic identity and explicit training split")
    case = identity.get("case", {})
    if case.get("kind") != "canonical" or case.get("dogs") != 2 or case.get("sheep") != 15:
        raise ValueError("Labels are restricted to canonical training populations")
    case_id = str(case.get("case_id", ""))
    if not case_id or Path(case_id).name != case_id or case_id in (".", ".."):
        raise ValueError("Invalid label case ID")
    if not rows: raise ValueError("No label endpoints")
    arrays = {key:np.stack([r[key] for r in rows]) for key in rows[0]}
    _validate_arrays(arrays)
    directory = Path(root)/"critic_labels"/f"s{identity['seed']}"/identity["split"]
    directory.mkdir(parents=True, exist_ok=True)
    data = directory/(case_id+".npz"); receipt = directory/(case_id+".json")
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old["identity"] != identity or old["dataset_sha256"] != _sha(data) or old.get("array_content_sha256") != _array_content_digest(arrays):
            raise RuntimeError("Existing label archive identity or checksum changed")
        return {**old, "reused":True}
    temporary = data.with_name(data.name+".tmp.npz")
    np.savez_compressed(temporary, **arrays); temporary.replace(data)
    result = dict(identity=identity, status="COMPLETE", dataset_sha256=_sha(data), samples=len(rows),
                  feature_schema="execution-state", array_content_sha256=_array_content_digest(arrays), rows=metadata or [],
                  supervision="true-simulator-training-worlds", no_critic_bootstrap=True)
    _write(receipt, result)
    return {**result, "reused":False}


def _validate_arrays(arrays):
    if set((*FEATURE_KEYS, "target"))-set(arrays): raise ValueError("Incomplete matched feature archive")
    if any(not np.isfinite(np.asarray(value)).all() for value in arrays.values()):
        raise ValueError("Nonfinite matched feature archive")
    if arrays["sheep"].ndim != 3 or arrays["dogs"].ndim != 3:
        raise ValueError("Endpoint entity arrays must include a row axis")
    count, n, m = len(arrays["target"]), arrays["sheep"].shape[1], arrays["dogs"].shape[1]
    expected = dict(sheep=(count,n,8), dogs=(count,m,8), global_=(count,12),
                    sheep_mask=(count,n), dog_mask=(count,m), physical_dogs=(count,m,DOG_SIDE_DIM),
                    execution_dogs=(count,m,DOG_SIDE_DIM), physical_global=(count,GLOBAL_SIDE_DIM),
                    execution_global=(count,GLOBAL_SIDE_DIM), target=(count,))
    expected["global"] = expected.pop("global_")
    for key, shape in expected.items():
        if arrays[key].shape != shape: raise ValueError("Wrong matched feature dimensions: "+key)
    if count < 1 or n < 1 or m < 1: raise ValueError("Empty matched feature archive")
    for mask in ("sheep_mask", "dog_mask"):
        if not np.all(np.isin(arrays[mask], (0,1))): raise ValueError("Nonbinary entity mask")


def _terminal(env, done=False):
    return bool(done or int(env.primitive_step) >= int(env.cfg.max_steps) or env.metrics().get("success", 0.0) > .5)


class CriticPredictor:
    def __init__(self, model, payload, device, sha256):
        self.model, self.payload, self.device = model, payload, device
        self.arm, self.sha256 = payload["arm"], sha256
        self.norm, self.feature_schema = payload["target_normalization"], payload["feature_schema"]

    def value(self, session, snapshot=None):
        import torch
        endpoint = snapshot.session if snapshot is not None else session
        if _terminal(endpoint.env, bool(getattr(snapshot, "done", False))): return 0.0
        values, schema = encode_endpoint(session, snapshot)
        if schema != self.feature_schema: raise RuntimeError("Runtime critic feature schema mismatch")
        batch = _as_tensors({k:v[None] for k,v in values.items()}, self.device)
        with torch.no_grad():
            value = float(self.model(batch, self.arm).item())*self.norm["std"]+self.norm["mean"]
        if not math.isfinite(value): raise RuntimeError("Nonfinite critic prediction")
        return value


def load_critic(path, device="cpu"):
    import torch
    path = Path(path)
    receipt = json.loads(path.with_suffix(".json").read_text())
    before = _sha(path)
    if receipt.get("sha256") != before: raise RuntimeError("Critic checkpoint checksum mismatch")
    payload = torch.load(path, map_location=device, weights_only=True)
    if _sha(path) != before: raise RuntimeError("Critic changed while loading")
    if payload.get("schema") != SCHEMA or payload.get("model_kind") != MODEL_KIND:
        raise RuntimeError("Wrong critic checkpoint kind")
    if payload.get("arm") not in ("physical", "execution") or payload.get("reward_scale") != REWARD_SCALE:
        raise RuntimeError("Unknown critic input arm or reward scaling")
    for key in ("arm", "identity_hash", "target_normalization", "feature_schema", "study_hash"):
        if receipt.get(key) != payload.get(key): raise RuntimeError("Critic receipt mismatch: "+key)
    norm = payload["target_normalization"]
    if not np.isfinite([norm["mean"], norm["std"]]).all() or norm["std"] <= 0:
        raise RuntimeError("Invalid critic target normalization")
    dims = payload["dims"]
    if dims["dog_side_dim"] != DOG_SIDE_DIM or dims["global_side_dim"] != GLOBAL_SIDE_DIM:
        raise RuntimeError("Wrong matched side feature dimensions")
    model = _make_model(**dims).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    if sum(p.numel() for p in model.parameters()) != payload.get("parameters"):
        raise RuntimeError("Critic parameter-count receipt mismatch")
    return CriticPredictor(model, payload, str(device), before)


def fit_matched_critics(dataset_root, *, seed, study_hash, device="cpu", options=None):
    """Convenience API for the integrated CLI; never consumes evaluation data."""
    return fit_critics(dict(out=str(dataset_root), study_hash=study_hash, critic=options or {}),
                       dict(seed=int(seed), device=device))


def make_matched_critic(hidden=64):
    """Construct the fixed matched architecture without loading any weights."""
    return _make_model(DOG_SIDE_DIM, GLOBAL_SIDE_DIM, hidden)
