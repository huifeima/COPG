"""Acceleration errors along isolated true first-segment diagnostic branches.

This module is supervision/diagnosis only, never deployed candidate scoring.
Its exposure distribution consists of the tested generator's actual candidate
executions from the diagnostic states, not the parent collection distribution.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from cogp.planning.branching import Rollout, clone_branch, state_digest


@dataclass
class ForceError:
    total_squared_error: float = 0.
    count: int = 0
    force_calls: int = 0
    sheep_accelerations: int = 0
    saturated_squared_error: float = 0.
    saturated_count: int = 0
    acceleration_scale: float | None = None

    def add(self, prediction: np.ndarray, truth: np.ndarray, limit: float) -> None:
        if self.acceleration_scale is None:
            self.acceleration_scale = limit
        elif limit != self.acceleration_scale:
            raise ValueError("Acceleration limit changed within one diagnostic branch")
        squared = np.square(prediction - truth)
        saturated = np.linalg.norm(truth, axis=-1) >= limit * (1. - 1e-5)
        self.total_squared_error += float(squared.sum(dtype=np.float64))
        self.count += int(squared.size)
        self.force_calls += 1
        self.sheep_accelerations += int(len(truth))
        self.saturated_squared_error += float(squared[saturated].sum(dtype=np.float64))
        self.saturated_count += int(squared[saturated].size)

    def summary(self) -> dict[str, Any]:
        mse = self.total_squared_error / self.count if self.count else None
        unsaturated_count = self.count - self.saturated_count
        unsaturated_error = max(0., self.total_squared_error - self.saturated_squared_error)
        return {
            "total_squared_error": self.total_squared_error,
            "count": self.count,
            "coordinate_count": self.count,
            "force_calls": self.force_calls,
            "sheep_accelerations": self.sheep_accelerations,
            "rmse": float(np.sqrt(mse)) if mse is not None else None,
            "normalized_mse": mse / self.acceleration_scale ** 2 if mse is not None else None,
            "acceleration_scale": self.acceleration_scale,
            "saturated_coordinate_count": self.saturated_count,
            "saturated_rmse": float(np.sqrt(self.saturated_squared_error / self.saturated_count))
                              if self.saturated_count else None,
            "unsaturated_rmse": float(np.sqrt(unsaturated_error / unsaturated_count))
                                if unsaturated_count else None,
            "label": "true_simulator_capped_acceleration",
            "prediction": "model_acceleration_with_deployed_euclidean_norm_cap",
            "state_timing": "after_shepherd_motion_before_each_sheep_integration_substep",
            "rmse_unit": "physical_acceleration_coordinate",
            "exposure": "true first-segment states under this tested candidate",
            "statistical_scope": "nested descriptive force samples, not independent training replicates",
            "deployed_scoring_used": False,
        }


@contextmanager
def record_force_error(env: Any, predictor: Any) -> Iterator[ForceError]:
    """Observe one true branch without changing any returned forces.

    The predictor is evaluated on the exact pre-integration state. Recursive
    attempts to read true forces inside ``predict_numpy`` are rejected.
    Existing predicted-force replacement routines are rejected as they are not
    real simulator labels. The original method is restored on every exit.
    """
    if predictor is None:
        raise ValueError("A fitted predictor is required for an acceleration-error diagnostic")
    name = "_compute_sheep_forces"
    original = getattr(env, name)
    original_class_function = getattr(type(env), name)
    if getattr(original, "__func__", None) is not original_class_function:
        raise RuntimeError("Force-error diagnostics require the unmodified true simulator force routine")
    had_override = name in vars(env)
    override = vars(env).get(name)
    errors = ForceError()
    in_prediction = False

    def observed():
        nonlocal in_prediction
        if in_prediction:
            raise RuntimeError("Predictor attempted true-force access during error measurement")
        result = original()
        if not isinstance(result, tuple) or len(result) != 3:
            raise RuntimeError("True force return must contain capped acceleration, load and graph")
        truth = np.asarray(result[0], dtype=np.float64)
        limit = float(env.cfg.sheep_max_accel)
        shape = (int(env.num_sheep), 2)
        if (truth.shape != shape or not np.isfinite(truth).all()
                or not np.isfinite(limit) or limit <= 0
                or np.any(np.linalg.norm(truth, axis=1) > limit + 1e-3)):
            raise RuntimeError("Diagnostic ground truth is not finite capped sheep acceleration")
        before = state_digest(env, None, None)
        in_prediction = True
        try:
            prediction = np.asarray(predictor.predict_numpy(env), dtype=np.float64)
        finally:
            in_prediction = False
        if state_digest(env, None, None) != before:
            raise RuntimeError("Acceleration predictor mutated the true diagnostic state")
        if prediction.shape != shape or not np.isfinite(prediction).all():
            raise RuntimeError("Predictor returned invalid acceleration in diagnostic")
        prediction = prediction.copy()
        norm = np.linalg.norm(prediction, axis=1)
        prediction *= np.minimum(1., limit / np.maximum(norm, 1e-12))[:, None]
        errors.add(prediction, truth, limit)
        return result

    setattr(env, name, observed)
    try:
        yield errors
    finally:
        if had_override:
            setattr(env, name, override)
        else:
            delattr(env, name)


def rollout_with_force_error(session: Any, spec: Any, predictor: Any,
                             H: int = 64) -> tuple[Rollout, dict[str, Any]]:
    """Execute one isolated true first segment and return its error diagnostic.

    This is the same requested-H commitment flow as ``rollout_branch(...,
    predictor=None)``. The candidate identity stays fixed; no continuation or
    additional candidate selection is simulated. Realised duration can differ
    from requested H because executor events and acquisition semantics apply.
    """
    if type(H) is not int or H <= 0:
        raise ValueError("H must be a positive reward-check budget")
    before = state_digest(session.env, session.co, session.ctx)
    local_env, local_co, local_ctx = clone_branch(session.env, session.co, session.ctx)
    local_co.max_option_high_steps = H
    start = int(local_env.primitive_step)
    with record_force_error(local_env, predictor) as errors:
        reward, done, info, duration, raw, pending = local_co.step_committed(local_env, local_ctx, spec)
    terminal = bool(done or local_env.primitive_step >= local_env.cfg.max_steps
                    or local_env.metrics()["success"] > .5)
    endpoint_ctx = pending if pending is not None else local_ctx if terminal else local_co.prepare(local_env)
    if state_digest(session.env, session.co, session.ctx) != before:
        raise RuntimeError("True first-segment diagnostics mutated live execution state")
    result = Rollout(local_env, local_co, endpoint_ctx, float(reward), float(raw), int(duration),
                     int(local_env.primitive_step) - start, terminal, dict(info))
    summary = errors.summary()
    summary.update(requested_H=H, realised_reward_checks=int(duration),
                   primitive_steps=result.primitive_steps, terminal=terminal,
                   branch_scope="one isolated true first segment, no continuation")
    if summary["force_calls"] == 0:
        raise RuntimeError("True first-segment diagnostic observed no sheep-force calls")
    return result, summary


def evaluate_force_error(session: Any, spec: Any, predictor: Any,
                         H: int = 64) -> dict[str, Any]:
    """Return descriptive force errors from exactly one isolated true branch."""
    return rollout_with_force_error(session, spec, predictor, H)[1]
