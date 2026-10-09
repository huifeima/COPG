"""Explicit exact-tensor normalization of the two available physical controls."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from .fit import CHECKPOINT_FORMAT, load_model, sha256, validate_state_dict
from .models import build_model, geometry_config


CONTROL_KINDS = ("boids_id", "local_pooled")


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def normalize_controls(source: str | Path, out: str | Path, kind: str,
                       geometry: Any = None, seed: int | None = None) -> dict[str, Any]:
    """Convert only the specified control architecture, without old code imports.

    Source kind, geometry, complete tensor keys/shapes/dtypes, and optional seed
    are checked. Values are copied exactly; no tensor reshaping, dropping,
    dtype casting, interpolation or coefficient inference is permitted.
    Original training labels and checkpoint selection are not reverified by
    this operation; an existing numerical validation value is not invented.
    """
    if kind not in CONTROL_KINDS:
        raise ValueError("Only boids_id and local_pooled control checkpoints can be normalized; "
                         "Graph/DeepSets architectures are not inferred")
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError("Requested training seed must be a nonnegative integer")
    source, destination = Path(source), Path(out)
    if source.resolve() == destination.resolve():
        raise ValueError("Normalization cannot overwrite its source checkpoint")
    receipt_path = destination.with_suffix(".conversion.json")
    if destination.exists() or receipt_path.exists():
        raise FileExistsError("Choose a new control checkpoint output; normalization does not overwrite")
    source_hash = sha256(source)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if sha256(source) != source_hash:
        raise RuntimeError("Source checkpoint changed while loading")
    if not isinstance(payload, Mapping):
        raise ValueError("Control source must be a checkpoint mapping")
    if payload.get("kind") != kind:
        raise ValueError("Requested control kind differs from source metadata")
    if not isinstance(payload.get("geometry"), Mapping):
        raise ValueError("Source must contain explicit control geometry; it cannot be guessed")
    source_geometry = geometry_config(payload["geometry"])
    # Geometry in the available controls is explicit, not a partly filled config.
    if set(payload["geometry"]) != set(source_geometry):
        raise ValueError("Source control geometry must contain exactly the known physical scales")
    target_geometry = source_geometry if geometry is None else geometry_config(geometry)
    if target_geometry != source_geometry:
        raise ValueError("Requested geometry differs from the source control's scales")
    identity = payload.get("identity", {})
    if not isinstance(identity, Mapping):
        raise ValueError("Source identity must be a mapping when supplied")
    if "geometry" in identity and geometry_config(identity["geometry"]) != source_geometry:
        raise ValueError("Source identity geometry conflicts with its checkpoint geometry")
    source_seed = identity.get("seed", payload.get("seed"))
    if source_seed is not None and (type(source_seed) is not int or source_seed < 0):
        raise ValueError("Source training seed is invalid")
    if seed is not None and source_seed is not None and seed != source_seed:
        raise ValueError("Requested training seed differs from source metadata")
    model = build_model(kind, source_geometry).float()
    if payload.get("parameters", model.parameter_count) != model.parameter_count:
        raise ValueError("Source control parameter-count metadata differs from architecture")
    state = payload.get("state_dict")
    validate_state_dict(model, state)
    # strict=True is kept in addition to the explicit shape/dtype/value checks.
    model.load_state_dict(state, strict=True)
    tensors = {key: value.detach().cpu().clone() for key, value in state.items()}
    proof = [{"source_key": key, "target_key": key,
              "shape": list(tensors[key].shape), "dtype": str(tensors[key].dtype),
              "source_tensor_sha256": tensor_sha256(state[key]),
              "target_tensor_sha256": tensor_sha256(tensors[key])}
             for key in sorted(tensors)]
    origin = {
        "operation": "strict_control_checkpoint_normalization",
        "source_sha256": source_hash, "source_kind": kind,
        "source_training_seed": source_seed, "requested_seed": seed,
        "seed_verified_against_source_metadata": seed is not None and source_seed is not None,
        "exact_tensor_mapping": True, "training_protocol_verified": False,
        "validation_rmse_recomputed": False, "numerical_experiment_reproduced": False,
    }
    converted = {"format": CHECKPOINT_FORMAT, "kind": kind,
                 "geometry": source_geometry, "parameters": model.parameter_count,
                 "state_dict": tensors, "origin": origin}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(destination) + ".tmp")
    torch.save(converted, temporary)
    os.replace(temporary, destination)
    loaded = load_model(destination)
    for key, tensor in loaded.state_dict().items():
        if not torch.equal(tensor.cpu(), tensors[key]):
            raise RuntimeError("Normalized checkpoint reload changed a source tensor")
    receipt = {
        "schema": "control_checkpoint_conversion", "status": "complete", **origin,
        "output_checkpoint_sha256": sha256(destination),
        "trainable_parameters": model.parameter_count,
        "state_tensor_elements": sum(value.numel() for value in tensors.values()),
        "tensor_keys": len(tensors), "tensor_proof": proof,
        "source_training_metadata_present": sorted(set(payload) & {
            "identity", "best_step", "train_episode_ids", "validation_episode_ids", "validation"}),
        "source_training_metadata_scope": "presence only; labels, split and selection not reverified",
        "functional_scope": "same named control architecture; original experiment not rerun",
        "implementation_sha256": sha256(__file__),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt
