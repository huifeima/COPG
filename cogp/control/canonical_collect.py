"""Task-axis canonical Collect policy evaluation."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
from cogp.environment.geometry import unit, perp
from cogp.control.reflection import reflect_action, reflect_observation

@dataclass(frozen=True)
class CanonicalCollectDiagnostics:
    canonicalized: bool
    signed_lateral: float
    easy_side_sign: int
    reflection_matrix: np.ndarray
    native_action: np.ndarray
    canonical_action: np.ndarray
    action_gap: float
    action_cosine: float

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(2)
    b = np.asarray(b, dtype=np.float64).reshape(2)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-09 or nb < 1e-09:
        return 1.0 if na < 1e-09 and nb < 1e-09 else 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))

def nominal_task_axis_frame(env) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin = np.asarray([env.cfg.flock_center_x, env.cfg.flock_center_y], dtype=np.float64)
    g = unit(env.goal.astype(np.float64) - origin)
    if np.linalg.norm(g) < 1e-09:
        g = np.array([0.0, -1.0], dtype=np.float64)
    s = perp(g)
    return (origin, g, s)

def nominal_task_axis_reflection(env) -> np.ndarray:
    _, g, _ = nominal_task_axis_frame(env)
    return 2.0 * np.outer(g, g) - np.eye(2, dtype=np.float64)

def collect_target_signed_lateral(env, target_index: int) -> float:
    origin, _, s = nominal_task_axis_frame(env)
    q = int(np.clip(int(target_index), 0, max(0, env.num_sheep - 1)))
    return float(np.dot(env.sheep_pos[q].astype(np.float64) - origin, s))

class CanonicalCollectController:

    def __init__(self, model, *, canonicalize_hard_side: bool=True, easy_side_sign: int=+1, side_deadband: float=1.0):
        self.model = model
        self.canonicalize_hard_side = bool(canonicalize_hard_side)
        self.easy_side_sign = +1 if int(easy_side_sign) >= 0 else -1
        self.side_deadband = float(max(0.0, side_deadband))

    def should_canonicalize(self, signed_lateral: float) -> bool:
        if not self.canonicalize_hard_side:
            return False
        if abs(float(signed_lateral)) <= self.side_deadband:
            return False
        return int(np.sign(float(signed_lateral))) != int(self.easy_side_sign)

    def predict(self, env, obs: Dict[str, np.ndarray], target_index: int, *, deterministic: bool=True, return_diagnostics: bool=False):
        native, _ = self.model.predict(obs, deterministic=deterministic)
        native = np.asarray(native, dtype=np.float32).reshape(2)
        signed = collect_target_signed_lateral(env, target_index)
        if not self.should_canonicalize(signed):
            if not return_diagnostics:
                return native
            I = np.eye(2, dtype=np.float32)
            d = CanonicalCollectDiagnostics(canonicalized=False, signed_lateral=float(signed), easy_side_sign=int(self.easy_side_sign), reflection_matrix=I, native_action=native.copy(), canonical_action=native.copy(), action_gap=0.0, action_cosine=1.0)
            return (native, d)
        R = nominal_task_axis_reflection(env)
        mirrored_obs, _ = reflect_observation(obs, R)
        mirrored_action, _ = self.model.predict(mirrored_obs, deterministic=deterministic)
        canonical = reflect_action(np.asarray(mirrored_action, dtype=np.float32), R)
        if not return_diagnostics:
            return canonical
        d = CanonicalCollectDiagnostics(canonicalized=True, signed_lateral=float(signed), easy_side_sign=int(self.easy_side_sign), reflection_matrix=np.asarray(R, dtype=np.float32), native_action=native.copy(), canonical_action=canonical.copy(), action_gap=float(np.linalg.norm(native.astype(np.float64) - canonical.astype(np.float64))), action_cosine=_cosine(native, canonical))
        return (canonical, d)
