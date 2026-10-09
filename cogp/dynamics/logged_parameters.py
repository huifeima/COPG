"""Recover a fixed Boids predictor from explicitly reported fitted coefficients.

This reader does not manufacture the missing original checkpoint. It preserves
the seven reported positive coefficients directly, avoiding a log/exp round trip.
It is deliberately separate from the scientific checkpoint registry and loader.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from .models import BoidsID, GEOMETRY_DEFAULTS, STATE_FIELDS, geometry_config


RECOVERY_FORMAT = "cogp-logged-boids-parameters"
RECOVERY_ORIGIN = "logged_parameter_recovery"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def recover_report_coefficients(
    report_path: str | Path, *, expected_sha256: str, expected_seed: int | None = None,
) -> dict[str, Any]:
    """Validate a complete, hash-pinned training report and expose its coefficients.

    The expected report hash must come from an independently retained manifest.
    Computing and passing it ad hoc verifies byte consistency only, not provenance.
    Report/checkpoint hashes remain distinct: the latter identifies bytes that are
    still missing and is never assigned to this recovered parameter representation.
    """
    expected_sha256 = _hash(expected_sha256, "expected_sha256")
    path = Path(report_path)
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError("Boids report SHA-256 mismatch")
    report = json.loads(raw)
    if not isinstance(report, dict) or report.get("status") != "complete":
        raise ValueError("Boids coefficient recovery requires a complete report")
    identity = report.get("identity")
    if not isinstance(identity, dict) or identity.get("kind") != "boids_id":
        raise ValueError("Boids report identity must declare kind boids_id")
    seed = _integer(identity.get("seed"), "report seed")
    if expected_seed is not None and seed != _integer(expected_seed, "expected_seed"):
        raise ValueError("Boids report training seed mismatch")
    if (_integer(identity.get("steps"), "reported fit steps") != 5000
            or _integer(report.get("optimizer_updates"), "optimizer_updates") != 5000):
        raise ValueError("Boids recovery requires the reported 5000-update fit")
    best_step = _integer(report.get("best_step"), "best_step")
    if best_step > 5000:
        raise ValueError("best_step exceeds the reported training updates")
    geometry = identity.get("geometry")
    if not isinstance(geometry, dict) or set(geometry) != set(GEOMETRY_DEFAULTS):
        raise ValueError("Boids report must explicitly provide all geometry scales")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           for value in geometry.values()):
        raise ValueError("Boids geometry scales must be numeric, not Boolean")
    geometry = geometry_config(geometry)
    coefficients = report.get("best_coefficients")
    if not isinstance(coefficients, dict) or set(coefficients) != set(BoidsID.PARAMETER_NAMES):
        raise ValueError("Boids report must provide exactly all seven best coefficients")
    ordered = {}
    for name in BoidsID.PARAMETER_NAMES:
        value = coefficients[name]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f"Reported coefficient {name} must be finite and positive")
        ordered[name] = float(value)
    return {
        "format": RECOVERY_FORMAT,
        "kind": "boids_id",
        "origin": RECOVERY_ORIGIN,
        "test_only": bool(report.get("test_only", False)),
        "seed": seed,
        "optimizer_updates": 5000,
        "best_step": best_step,
        "geometry": geometry,
        "coefficients": ordered,
        "source_report_name": path.name,
        "source_report_sha256": actual,
        "missing_original_checkpoint_sha256": _hash(
            report.get("checkpoint_sha256"), "original checkpoint_sha256"),
        "scope": (
            "Reported positive coefficients evaluated by the available Boids formula; "
            "not an original checkpoint, optimizer state, or proof of historical replay."
        ),
    }


class ReportedBoidsPredictor(BoidsID):
    """Fixed reported coefficients on the available Boids physical-state interface.

    Use ``from_report`` to validate its source. This class has no trainable
    parameters and does not satisfy the scientific checkpoint-loader contract.
    Neither an original log-parameter tensor nor an original checkpoint is inferred.
    """

    def __init__(self, recovered: Mapping[str, Any]):
        if recovered.get("format") != RECOVERY_FORMAT or recovered.get("origin") != RECOVERY_ORIGIN:
            raise ValueError("Expected a validated logged-parameter recovery record")
        if recovered.get("kind") != "boids_id" or recovered.get("optimizer_updates") != 5000:
            raise ValueError("Recovered record must declare the original Boids fit identity")
        _integer(recovered.get("seed"), "recovered seed")
        _hash(recovered.get("source_report_sha256"), "source_report_sha256")
        _hash(recovered.get("missing_original_checkpoint_sha256"), "missing_original_checkpoint_sha256")
        if not isinstance(recovered.get("geometry"), Mapping) or set(recovered["geometry"]) != set(GEOMETRY_DEFAULTS):
            raise ValueError("Recovered record must explicitly provide all geometry scales")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               for value in recovered["geometry"].values()):
            raise ValueError("Recovered geometry scales must be numeric, not Boolean")
        coefficients = recovered.get("coefficients")
        if not isinstance(coefficients, Mapping) or set(coefficients) != set(self.PARAMETER_NAMES):
            raise ValueError("Recovered record must contain all seven coefficients")
        values = [coefficients[name] for name in self.PARAMETER_NAMES]
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("Recovered coefficients must be finite and positive")
        super().__init__(recovered["geometry"])
        del self.log_parameters
        del self.gain_scales
        self.register_buffer("reported_coefficients", torch.tensor(values, dtype=torch.float32))
        if not torch.isfinite(self.reported_coefficients).all() or not (self.reported_coefficients > 0).all():
            raise ValueError("Recovered coefficients are not representable as positive float32")
        self.recovery = dict(recovered)
        self.test_only = bool(recovered.get("test_only", False))
        self.origin_kind = RECOVERY_ORIGIN
        self.checkpoint_origin = {"type": RECOVERY_ORIGIN, "source_training_seed": recovered["seed"]}
        self.source_report_sha256 = recovered["source_report_sha256"]
        self.missing_original_checkpoint_sha256 = recovered["missing_original_checkpoint_sha256"]
        self.eval()

    @classmethod
    def from_report(cls, report_path: str | Path, *, expected_sha256: str,
                    expected_seed: int | None = None) -> "ReportedBoidsPredictor":
        return cls(recover_report_coefficients(
            report_path, expected_sha256=expected_sha256, expected_seed=expected_seed))

    def positive_parameters(self) -> Tensor:
        return self.reported_coefficients

    @torch.no_grad()
    def predict_numpy(self, env: Any) -> np.ndarray:
        """Recompute local adjacency without reading force coefficients or force labels."""
        device, dtype = self.reported_coefficients.device, self.reported_coefficients.dtype
        values = {key: np.array(getattr(env, key), copy=True) for key in STATE_FIELDS[:-1]}
        values["social_graph"] = np.array(env.social_graph_mask(), copy=True)
        batch = {
            key: torch.as_tensor(value, device=device,
                                 dtype=torch.bool if key == "social_graph" else dtype).unsqueeze(0)
            for key, value in values.items()
        }
        return self(batch)[0].cpu().numpy().astype(np.float32, copy=False)
