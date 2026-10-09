"""Matched dynamics fitting, explicit episode splits and verified checkpoints."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from .models import (STATE_FIELDS, MODEL_KINDS, BoidsID, AccelerationModel,
                     build_model, geometry_config)

CHECKPOINT_FORMAT = "cogp-dynamics"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_atomic(path: Path, payload: Any) -> None:
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_model(path: str | Path, device: str = "cpu") -> AccelerationModel:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Dynamics checkpoint must be a mapping")
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Unsupported dynamics checkpoint schema")
    if not isinstance(checkpoint.get("geometry"), Mapping):
        raise ValueError("Checkpoint must contain explicit geometry")
    from .models import GEOMETRY_DEFAULTS
    if set(checkpoint["geometry"]) != set(GEOMETRY_DEFAULTS):
        raise ValueError("Checkpoint geometry must contain exactly the known physical scales")
    model = build_model(checkpoint["kind"], checkpoint["geometry"]).float()
    validate_state_dict(model, checkpoint.get("state_dict"))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if checkpoint.get("parameters", model.parameter_count) != model.parameter_count:
        raise ValueError("Dynamics checkpoint parameter-count header differs from architecture")
    model.checkpoint_identity=dict(checkpoint.get('identity',{}))
    model.checkpoint_identity_sha256=checkpoint.get('identity_sha256')
    model.checkpoint_origin=dict(checkpoint.get('origin',{}))
    model.checkpoint_sha256=sha256(path)
    return model.to(device).eval().requires_grad_(False)


def validate_state_dict(model: AccelerationModel, state: Any) -> None:
    """Reject casts, missing buffers, nonfinite tensors or hidden extra weights."""
    if not isinstance(state, Mapping):
        raise ValueError("Dynamics checkpoint must contain a tensor state_dict")
    expected = model.state_dict()
    if set(state) != set(expected):
        raise RuntimeError(f"State-dict keys differ: missing={sorted(set(expected)-set(state))}; "
                           f"extra={sorted(set(state)-set(expected))}")
    for key, reference in expected.items():
        tensor = state[key]
        if not isinstance(tensor, Tensor):
            raise ValueError(f"State-dict entry {key} is not a tensor")
        if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
            raise RuntimeError(f"State-dict shape or dtype differs at {key}")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"State-dict entry {key} contains nonfinite values")
    if isinstance(model, BoidsID) and not torch.equal(state["gain_scales"].cpu(), expected["gain_scales"]):
        raise ValueError("BoidsID conditioning scales do not match checkpoint geometry")


def _reference(split_manifest: Any) -> tuple[dict[str, Any] | None, str | None]:
    if split_manifest is None:
        return None, None
    if isinstance(split_manifest, Mapping):
        report = dict(split_manifest)
        return report, hashlib.sha256(_canonical(report).encode()).hexdigest()
    path = Path(split_manifest)
    return json.loads(path.read_text()), sha256(path)


def _split(episode_ids: np.ndarray, seed: int, report: dict | None,
           dataset_hash: str) -> tuple[np.ndarray, np.ndarray, np.random.Generator]:
    ids = np.unique(episode_ids)
    if len(ids) < 5:
        raise ValueError("At least five episodes are required")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(ids)  # Match the shared RNG state before minibatch sampling.
    cut = max(1, int(np.ceil(len(shuffled) * 0.2)))
    val_ids, train_ids = shuffled[:cut], shuffled[cut:]
    if report is not None:
        identity = report.get("identity", report.get("fit_identity", {}))
        reference_hash = identity.get("dataset_sha256", report.get("dataset_sha256"))
        if reference_hash is None or reference_hash != dataset_hash:
            raise ValueError("Episode split manifest dataset hash is missing or differs")
        ref_seed = identity.get("seed", report.get("seed"))
        if ref_seed is not None and int(ref_seed) != seed:
            raise ValueError("Episode split manifest uses a different seed")
        if "train_episode_ids" not in report or "validation_episode_ids" not in report:
            raise ValueError("Episode split manifest must contain explicit train/validation episode IDs")
        train_ids = np.asarray(report["train_episode_ids"], dtype=ids.dtype)
        val_ids = np.asarray(report["validation_episode_ids"], dtype=ids.dtype)
    train_set, val_set, all_set = set(train_ids.tolist()), set(val_ids.tolist()), set(ids.tolist())
    if (not train_set or not val_set or train_set & val_set or train_set | val_set != all_set
            or len(train_set) != len(train_ids) or len(val_set) != len(val_ids)):
        raise ValueError("Episode split must be a disjoint, nonempty, complete partition")
    return train_ids, val_ids, rng


class _Cache:
    """Use a conservative GPU cache when feasible, otherwise gather on CPU."""
    def __init__(self, data: dict[str, np.ndarray], device: str):
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Only CPU and CUDA fitting are supported")
        self.tensors = {key: torch.as_tensor(data[key], dtype=torch.bool if key == "social_graph" else torch.float32)
                        for key in (*STATE_FIELDS, "accel")}
        size = sum(t.numel() * t.element_size() for t in self.tensors.values())
        self.info = {"mode": "cpu_tensor", "dataset_tensor_bytes": size}
        if self.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(self.device)
            budget = max(0, min(free // 4, free - 1024 ** 3))
            self.info.update(cuda_free_bytes=int(free), cuda_cache_budget_bytes=int(budget))
            if size <= budget:
                moved = {}
                try:
                    for key, value in self.tensors.items():
                        moved[key] = value.to(self.device)
                    self.tensors = moved
                    self.info["mode"] = "cuda_resident"
                except torch.cuda.OutOfMemoryError:
                    moved.clear()
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()
                    self.info["mode"] = "cpu_gather_to_cuda"
            else:
                self.info["mode"] = "cpu_gather_to_cuda"

    def batch(self, indexes: np.ndarray) -> tuple[dict[str, Tensor], Tensor]:
        index_device = self.tensors["accel"].device
        indices = torch.as_tensor(indexes, dtype=torch.long, device=index_device)
        selected = {key: torch.index_select(value, 0, indices).to(self.device)
                    for key, value in self.tensors.items()}
        return {key: selected[key] for key in STATE_FIELDS}, selected["accel"]


def _synchronize(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def fit(dataset_path, out_path, kind, config, seed, updates=5000,
        batch_size=64, device="cpu", split_manifest=None) -> dict[str, Any]:
    """Fit a predictor; reuse only a complete, hash-verified matching fit.

    Explicit manifests from a shared fit pin both data identity and episode
    partition. Without one, the seeded 80/20 episode algorithm is used.
    An interrupted fit is restarted deterministically; partial weights are not
    treated as complete. Only validation acceleration error selects checkpoints.
    """
    if kind not in MODEL_KINDS or updates < 1 or batch_size < 1:
        raise ValueError("Invalid kind, updates or batch_size")
    dataset_path, out_path = Path(dataset_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = out_path.with_suffix(".report.json")
    report_hash_path = Path(str(report_path) + ".sha256")
    started = time.perf_counter()
    dataset_hash = sha256(dataset_path)
    reference, reference_hash = _reference(split_manifest)
    geometry = geometry_config(config)
    identity = {
        "format": CHECKPOINT_FORMAT, "kind": kind, "seed": int(seed), "updates": int(updates),
        "batch_size": int(batch_size), "dataset_sha256": dataset_hash,
        "split_manifest_sha256": reference_hash, "geometry": geometry,
        "learning_rate": 0.001, "weight_decay": 1e-5, "minimum_learning_rate": 1e-4,
        "loss": "mean_squared_capped_acceleration_divided_by_accel_scale_squared",
        "split": "manifest_episode_ids" if reference is not None else "seeded_episode_80_20",
        "implementation_sha256": {"fit": sha256(__file__), "models": sha256(Path(__file__).with_name("models.py"))},
    }
    identity_hash = hashlib.sha256(_canonical(identity).encode()).hexdigest()
    if out_path.exists() and report_path.exists() and report_hash_path.exists():
        try:
            old = json.loads(report_path.read_text())
            if (old.get("status") == "complete" and old.get("identity_sha256") == identity_hash
                    and old.get("checkpoint_sha256") == sha256(out_path)
                    and report_hash_path.read_text().strip() == sha256(report_path)):
                load_model(out_path, "cpu")  # Check schema/state-dict too.
                return old
        except (OSError, ValueError, RuntimeError, KeyError, EOFError):
            pass
    with np.load(dataset_path, allow_pickle=False) as z:
        required = (*STATE_FIELDS, "accel", "episode_id")
        if set(required) - set(z.files):
            raise ValueError("Dataset is missing required dynamics fields")
        data = {key: z[key] for key in required}
    length = len(data["accel"])
    if length < 1 or any(len(value) != length for value in data.values()):
        raise ValueError("Dataset fields must have the same nonzero length")
    if any(not np.isfinite(value).all() for value in data.values()):
        raise ValueError("Dataset contains nonfinite data")
    if data["dog_pos"].ndim != 3 or data["dog_pos"].shape[1] != 2:
        raise ValueError("Paper-matched dynamics fitting uses M=2 data only")
    if data["accel"].shape != data["sheep_pos"].shape:
        raise ValueError("Acceleration labels must match sheep position shape")
    if data["episode_id"].ndim != 1 or not np.issubdtype(data["episode_id"].dtype, np.integer):
        raise ValueError("episode_id must be a one-dimensional integer array")
    train_ids, val_ids, rng = _split(data["episode_id"], int(seed), reference, dataset_hash)
    train_index = np.flatnonzero(np.isin(data["episode_id"], train_ids))
    val_index = np.flatnonzero(np.isin(data["episode_id"], val_ids))
    cap = geometry["sheep_max_accel"]
    norms = np.linalg.norm(data["accel"], axis=-1)
    if float(norms.max()) > cap + max(1e-3, cap * 1e-4):
        raise ValueError("Labels exceed the expected post-cap acceleration limit")
    saturated = norms >= cap * (1.0 - 1e-5)
    saturation = {"all": float(saturated.mean()), "training": float(saturated[train_index].mean()),
                  "validation": float(saturated[val_index].mean()), "threshold": cap * (1.0 - 1e-5)}
    torch.manual_seed(int(seed))
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model = build_model(kind, geometry).float().to(device)
    initial_coefficients = model.coefficient_report() if isinstance(model, BoidsID) else None
    cache = _Cache(data, device)
    del data
    first_batch, _ = cache.batch(train_index[:1])
    model.validate_batch(first_batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates, eta_min=1e-4)
    history = []
    best = float("inf")
    best_step = 0
    interval = max(1, min(100, updates // 10))

    @torch.no_grad()
    def evaluate() -> dict[str, Any]:
        model.eval()
        totals = torch.zeros(6, dtype=torch.float64, device=device)
        for start in range(0, len(val_index), batch_size):
            inputs, target = cache.batch(val_index[start:start + batch_size])
            squared = (model(inputs) - target).double().square().sum(-1)
            sat = torch.linalg.vector_norm(target, dim=-1) >= saturation["threshold"]
            totals += torch.stack((squared.sum(), squared.new_tensor(squared.numel() * 2),
                                   squared[sat].sum(), sat.sum().double() * 2,
                                   squared[~sat].sum(), (~sat).sum().double() * 2))
        values = totals.cpu().tolist()
        if not np.isfinite(values).all() or values[1] <= 0:
            raise RuntimeError("Nonfinite/empty validation error")
        return {"mse": values[0] / values[1], "force_rmse": float(np.sqrt(values[0] / values[1])),
                "saturated_force_rmse": float(np.sqrt(values[2] / values[3])) if values[3] else None,
                "unsaturated_force_rmse": float(np.sqrt(values[4] / values[5])) if values[5] else None}

    initial_validation = evaluate()
    fit_started = time.perf_counter()
    for step in range(1, updates + 1):
        model.train()
        indexes = rng.choice(train_index, size=min(batch_size, len(train_index)), replace=False)
        inputs, target = cache.batch(indexes)
        loss = ((model(inputs) - target) / cap).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not (torch.isfinite(loss.detach()) & torch.isfinite(gradient)).item():
            raise RuntimeError(f"Nonfinite fit loss/gradient at update {step}")
        optimizer.step()
        scheduler.step()
        if step == 1 or step % interval == 0 or step == updates:
            validation = evaluate()
            row = {"step": step, "training_normalized_mse": float(loss.item()),
                   "gradient_norm_before_clip": float(gradient.item()),
                   "validation_force_rmse": validation["force_rmse"],
                   "validation_saturated_force_rmse": validation["saturated_force_rmse"],
                   "validation_unsaturated_force_rmse": validation["unsaturated_force_rmse"]}
            if isinstance(model, BoidsID):
                row["coefficients"] = model.coefficient_report()
            history.append(row)
            if validation["mse"] < best:
                best, best_step = validation["mse"], step
                checkpoint = {"format": CHECKPOINT_FORMAT, "kind": kind, "geometry": geometry,
                              "parameters": model.parameter_count,
                              "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                              "identity": identity, "identity_sha256": identity_hash,
                              "best_step": step, "train_episode_ids": train_ids.tolist(),
                              "validation_episode_ids": val_ids.tolist(), "validation": validation}
                temp_checkpoint = Path(str(out_path) + ".tmp")
                torch.save(checkpoint, temp_checkpoint)
                os.replace(temp_checkpoint, out_path)
            print(f"[{kind}] {step}/{updates} validation acceleration RMSE={validation['force_rmse']:.6f}", flush=True)
    _synchronize(device)
    fit_seconds = time.perf_counter() - fit_started
    best_model = load_model(out_path, "cpu")
    warnings = []
    if best_step == updates:
        warnings.append("Best checkpoint is final update; optimization may not have converged.")
    if best >= initial_validation["mse"]:
        warnings.append("No validated improvement over initialization; inspect fitting before attributing a model gap.")
    if saturation["training"] > 0.5:
        warnings.append("Over half of training accelerations are saturated; physical gains may be weakly identifiable.")
    report = {
        "status": "complete", "identity": identity, "identity_sha256": identity_hash,
        "checkpoint_sha256": sha256(out_path), "best_step": best_step,
        "best_validation_force_rmse": float(np.sqrt(best)), "initial_validation": initial_validation,
        "final_validation": validation, "train_episode_ids": train_ids.tolist(),
        "validation_episode_ids": val_ids.tolist(), "training_episodes": len(train_ids),
        "validation_episodes": len(val_ids), "training_samples": len(train_index),
        "validation_samples": len(val_index), "parameters": best_model.parameter_count,
        "optimizer_updates": int(updates), "processed_training_samples": int(updates * min(batch_size, len(train_index))),
        "label": "simulator_clipped_sheep_acceleration", "label_saturation_fraction": saturation,
        "cache": cache.info, "elapsed_seconds": fit_seconds,
        "end_to_end_fit_seconds": time.perf_counter() - started,
        "torch_version": str(torch.__version__), "device": str(device),
        "convergence_warnings": warnings, "history": history,
    }
    if isinstance(best_model, BoidsID):
        report["best_coefficients"] = best_model.coefficient_report()
        report["initial_coefficients"] = initial_coefficients
        report["coefficient_initialization"] = "unit_dimensionless_gains_with_geometry_based_scales"
    _json_atomic(report_path, report)
    temp_hash = Path(str(report_hash_path) + ".tmp")
    temp_hash.write_text(sha256(report_path) + "\n")
    os.replace(temp_hash, report_hash_path)
    return report
