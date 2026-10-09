"""Predictive shepherd collision shield."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

@dataclass
class SafetyShieldDiagnostic:
    applied: bool
    pair_corrections: int
    correction_norm_mean: float
    correction_norm_max: float
    min_pair_distance: float
    min_predicted_distance_before: float
    min_predicted_distance_after: float

class DogDogSafetyShield:

    def __init__(self, *, soft_radius: float=42.0, hard_radius: float=20.0, separation_speed: float=2.5, prediction_horizon: float=0.75):
        self.soft_radius = float(soft_radius)
        self.hard_radius = float(hard_radius)
        self.separation_speed = float(separation_speed)
        self.prediction_horizon = float(prediction_horizon)
        if self.soft_radius <= self.hard_radius:
            raise ValueError('soft_radius must exceed hard_radius')

    @staticmethod
    def _clip_velocity(v: np.ndarray, max_speed: float) -> np.ndarray:
        out = np.asarray(v, dtype=np.float64).copy()
        speed = np.linalg.norm(out, axis=1)
        mask = speed > float(max_speed)
        if np.any(mask):
            out[mask] *= (float(max_speed) / speed[mask])[:, None]
        return out

    @staticmethod
    def _pair_min(points: np.ndarray) -> float:
        p = np.asarray(points, dtype=np.float64)
        if len(p) < 2:
            return float('inf')
        return min((float(np.linalg.norm(p[i] - p[j])) for i in range(len(p)) for j in range(i + 1, len(p))))

    def _predicted_pair_min(self, pos: np.ndarray, vel: np.ndarray) -> float:
        p = np.asarray(pos, dtype=np.float64)
        v = np.asarray(vel, dtype=np.float64)
        if len(p) < 2:
            return float('inf')
        best = float('inf')
        T = max(0.0, self.prediction_horizon)
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                r = p[i] - p[j]
                u = v[i] - v[j]
                uu = float(np.dot(u, u))
                if uu <= 1e-12:
                    t = 0.0
                else:
                    t = float(np.clip(-np.dot(r, u) / uu, 0.0, T))
                best = min(best, float(np.linalg.norm(r + t * u)))
        return best

    def project(self, env, actions: np.ndarray, *, gains: Optional[np.ndarray]=None) -> Tuple[np.ndarray, SafetyShieldDiagnostic]:
        a = np.asarray(actions, dtype=np.float64).reshape(env.num_dogs, 2)
        if gains is None:
            g = np.ones(env.num_dogs, dtype=np.float64)
        else:
            g = np.clip(np.asarray(gains, dtype=np.float64).reshape(env.num_dogs), 0.0, 1.0)
        desired = a * g[:, None] * float(env.cfg.dog_max_speed)
        before_vel = self._clip_velocity(desired, env.cfg.dog_max_speed)
        vel = before_vel.copy()
        pos = env.dog_pos.astype(np.float64)
        corrections = np.zeros_like(vel)
        pair_corrections = 0
        min_pair = self._pair_min(pos)
        soft_span = max(1e-06, self.soft_radius - self.hard_radius)
        for _ in range(2):
            for i in range(env.num_dogs):
                for j in range(i + 1, env.num_dogs):
                    delta = pos[i] - pos[j]
                    d = float(np.linalg.norm(delta))
                    if d <= 1e-09 or d >= self.soft_radius:
                        continue
                    n = delta / d
                    relative_normal = float(np.dot(vel[i] - vel[j], n))
                    urgency = float(np.clip((self.soft_radius - d) / soft_span, 0.0, 1.0))
                    pair_changed = False
                    if relative_normal < 0.0:
                        c = -0.5 * relative_normal * urgency
                        dv = c * n
                        vel[i] += dv
                        vel[j] -= dv
                        corrections[i] += dv
                        corrections[j] -= dv
                        pair_changed = True
                    if d < self.hard_radius + 6.0:
                        near = float(np.clip((self.hard_radius + 6.0 - d) / 6.0, 0.0, 1.0))
                        sep = 0.5 * self.separation_speed * near * n
                        vel[i] += sep
                        vel[j] -= sep
                        corrections[i] += sep
                        corrections[j] -= sep
                        pair_changed = True
                    pair_corrections += int(pair_changed)
            vel = self._clip_velocity(vel, env.cfg.dog_max_speed)
        out = vel / max(1e-09, float(env.cfg.dog_max_speed))
        out = np.clip(out, -1.0, 1.0).astype(np.float32)
        cn = np.linalg.norm(corrections, axis=1)
        diag = SafetyShieldDiagnostic(applied=bool(pair_corrections > 0), pair_corrections=int(pair_corrections), correction_norm_mean=float(np.mean(cn)) if len(cn) else 0.0, correction_norm_max=float(np.max(cn)) if len(cn) else 0.0, min_pair_distance=float(min_pair), min_predicted_distance_before=float(self._predicted_pair_min(pos, before_vel)), min_predicted_distance_after=float(self._predicted_pair_min(pos, vel)))
        return (out, diag)
