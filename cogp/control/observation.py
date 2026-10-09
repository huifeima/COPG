"""Frozen boundary-set controller observation."""

from __future__ import annotations
from typing import Optional, Dict, Tuple
import numpy as np
from cogp.environment.config import PlannerConfig
from cogp.environment.geometry import unit, convex_hull_indices

class BASCWPObservationBuilder:

    def __init__(self, max_boundary: int=10, obs_clip: float=5.0, pb_radius: float=70.0):
        self.max_boundary = int(max_boundary)
        self.obs_clip = float(obs_clip)
        self.pb_radius = float(pb_radius)

    def _select(self, env, dog_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p = env.sheep_pos.astype(np.float64)
        hull = convex_hull_indices(p)
        c = p[hull].mean(axis=0) if len(hull) else p.mean(axis=0)
        d = np.linalg.norm(p - dog_pos[None, :], axis=1)
        r = min(80.0, env.cfg.dog_influence_radius)
        cand = np.where(d <= r)[0]
        if cand.size == 0:
            return (np.array([int(np.argmin(d))], dtype=np.int64), hull.astype(np.int64))
        pressure = np.maximum(0.0, 1.0 - d[cand] / max(r, 1e-08))
        dog_side = unit(dog_pos - c)
        raw = (p[cand] - c[None, :]) @ dog_side
        frontier = (raw - raw.min()) / max(float(raw.max() - raw.min()), 1e-08) if len(raw) > 1 else np.ones_like(raw)
        hull_set = set(map(int, hull.tolist()))
        hb = np.asarray([1.0 if int(i) in hull_set else 0.0 for i in cand])
        score = 2.0 * pressure + frontier + 0.25 * hb
        order = np.lexsort((d[cand], -score))
        return (cand[order][:self.max_boundary].astype(np.int64), hull.astype(np.int64))

    def build(self, env, dog_index: int, task_target: np.ndarray, dog_anchor: np.ndarray) -> Dict[str, np.ndarray]:
        scale = max(env.cfg.world_size, 1e-08)
        svs = max(env.cfg.sheep_max_speed, 1e-08)
        dvs = max(env.cfg.dog_max_speed, 1e-08)
        p = env.sheep_pos.astype(np.float32)
        v = env.sheep_vel.astype(np.float32)
        dog = env.dog_pos[dog_index].astype(np.float32)
        dvel = env.dog_vel[dog_index].astype(np.float32)
        target = np.asarray(task_target, dtype=np.float32)
        anchor = np.asarray(dog_anchor, dtype=np.float32)
        selected, hull = self._select(env, dog.astype(np.float64))
        pa = p[hull].mean(axis=0) if len(hull) else p.mean(axis=0)
        X = np.zeros((self.max_boundary, 12), dtype=np.float32)
        M = np.zeros(self.max_boundary, dtype=np.float32)
        for k, i in enumerate(selected):
            pi, vi = (p[i], v[i])
            sd, sh, sp = (dog - pi, pa - pi, target - pi)
            dd, dh, dt = map(float, [np.linalg.norm(sd), np.linalg.norm(sh), np.linalg.norm(sp)])
            pressure = max(0.0, 1.0 - dd / max(env.cfg.dog_influence_radius, 1e-08))
            X[k] = [sd[0] / scale, sd[1] / scale, dd / scale, sh[0] / scale, sh[1] / scale, dh / scale, sp[0] / scale, sp[1] / scale, max(0.0, dt - self.pb_radius) / scale, vi[0] / svs, vi[1] / svs, pressure]
            M[k] = 1.0
        pa_task = target - pa
        dog_pa = pa - dog
        dog_anchor = anchor - dog
        hull_radius = float(np.max(np.linalg.norm(p - pa[None, :], axis=1)))
        G = np.asarray([pa_task[0] / scale, pa_task[1] / scale, float(np.linalg.norm(pa_task)) / scale, dog_pa[0] / scale, dog_pa[1] / scale, dvel[0] / dvs, dvel[1] / dvs, self.pb_radius / scale, hull_radius / scale, float(len(selected)) / max(self.max_boundary, 1), float(env.num_sheep) / max(env.cfg.max_sheep, 1), dog_anchor[0] / scale, dog_anchor[1] / scale, float(np.linalg.norm(dog_anchor)) / scale], dtype=np.float32)
        return {'boundary_set': np.clip(np.nan_to_num(X), -self.obs_clip, self.obs_clip).astype(np.float32), 'boundary_mask': M.astype(np.float32), 'global': np.clip(np.nan_to_num(G), -self.obs_clip, self.obs_clip).astype(np.float32)}
