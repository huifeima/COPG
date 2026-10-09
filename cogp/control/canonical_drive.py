"""Task-axis canonical Drive policy evaluation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
from cogp.control.reflection import reflect_action, reflect_observation, reflection_matrix_from_obs
DRIVE_ROLES = ('rear', 'left', 'right')

@dataclass(frozen=True)
class CanonicalDriveDiagnostics:
    canonicalized: bool
    reflection_matrix: np.ndarray
    native_action: np.ndarray
    canonical_action: np.ndarray
    action_gap: float
    action_cosine: float

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(2)
    b = np.asarray(b, dtype=np.float64).reshape(2)
    na, nb = (float(np.linalg.norm(a)), float(np.linalg.norm(b)))
    if na < 1e-09 or nb < 1e-09:
        return 1.0 if na < 1e-09 and nb < 1e-09 else 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))

class CanonicalDriveController:

    def __init__(self, model, *, canonicalize_right: bool=True):
        self.model = model
        self.canonicalize_right = bool(canonicalize_right)

    def predict(self, obs: Dict[str, np.ndarray], role: str, *, deterministic: bool=True, return_diagnostics: bool=False):
        role = str(role).lower()
        if role not in DRIVE_ROLES:
            raise ValueError(f'Drive role must be one of {DRIVE_ROLES}, got {role!r}')
        native, _ = self.model.predict(obs, deterministic=deterministic)
        native = np.asarray(native, dtype=np.float32).reshape(2)
        if role != 'right' or not self.canonicalize_right:
            if not return_diagnostics:
                return native
            I = np.eye(2, dtype=np.float32)
            d = CanonicalDriveDiagnostics(canonicalized=False, reflection_matrix=I, native_action=native.copy(), canonical_action=native.copy(), action_gap=0.0, action_cosine=1.0)
            return (native, d)
        R = reflection_matrix_from_obs(obs)
        mirrored_obs, _ = reflect_observation(obs, R)
        mirrored_action, _ = self.model.predict(mirrored_obs, deterministic=deterministic)
        canonical = reflect_action(np.asarray(mirrored_action, dtype=np.float32), R)
        if not return_diagnostics:
            return canonical
        gap = float(np.linalg.norm(native.astype(np.float64) - canonical.astype(np.float64)))
        d = CanonicalDriveDiagnostics(canonicalized=True, reflection_matrix=np.asarray(R, dtype=np.float32), native_action=native.copy(), canonical_action=canonical.copy(), action_gap=gap, action_cosine=_cosine(native, canonical))
        return (canonical, d)
