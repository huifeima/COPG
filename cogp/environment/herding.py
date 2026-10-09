"""Primitive shepherding simulator with reward-check success semantics."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import math
import numpy as np
from cogp.environment.config import EnvConfig
from cogp.environment.geometry import unit, perp, clip_norm, convex_hull_indices, sample_disk
SCENARIOS = ('compact', 'single_outlier', 'double_outlier', 'split', 'mixed')

class MultiDogHerdingEnv:
    VERSION = 'aamas2027-graphherd-env-'

    def __init__(self, cfg: Optional[EnvConfig]=None):
        self.cfg = cfg if cfg is not None else EnvConfig()
        self.cfg.validate()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.num_sheep = int(self.cfg.num_sheep)
        self.num_dogs = int(self.cfg.num_dogs)
        self.sheep_pos = np.zeros((self.num_sheep, 2), dtype=np.float32)
        self.sheep_vel = np.zeros((self.num_sheep, 2), dtype=np.float32)
        self.dog_pos = np.zeros((self.num_dogs, 2), dtype=np.float32)
        self.dog_vel = np.zeros((self.num_dogs, 2), dtype=np.float32)
        self.dog_path_length = np.zeros(self.num_dogs, dtype=np.float64)
        self.goal = np.array([self.cfg.goal_x, self.cfg.goal_y], dtype=np.float32)
        self.primitive_step = 0
        self.high_step = 0
        self.success_hold = 0
        self.current_scenario = 'compact'
        self._last_direct_dog_mask = np.zeros((self.num_dogs, self.num_sheep), dtype=bool)
        self._last_repulsive_load = np.zeros(self.num_sheep, dtype=np.float32)
        self._last_graph_mask = np.zeros((self.num_sheep, self.num_sheep), dtype=bool)

    def reset(self, seed: Optional[int]=None, scenario: str='mixed', num_sheep: Optional[int]=None, num_dogs: Optional[int]=None) -> Dict[str, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        if num_sheep is not None:
            if not 1 <= int(num_sheep) <= self.cfg.max_sheep:
                raise ValueError(f'num_sheep must be <= {self.cfg.max_sheep}')
            self.num_sheep = int(num_sheep)
        if num_dogs is not None:
            if not 1 <= int(num_dogs) <= self.cfg.max_dogs:
                raise ValueError(f'num_dogs must be <= {self.cfg.max_dogs}')
            self.num_dogs = int(num_dogs)
        if scenario not in SCENARIOS:
            raise ValueError(f'scenario must be one of {SCENARIOS}')
        if scenario == 'mixed':
            scenario = str(self.rng.choice(['compact', 'single_outlier', 'double_outlier', 'split'], p=[0.4, 0.35, 0.15, 0.1]))
        self.current_scenario = scenario
        self.primitive_step = 0
        self.high_step = 0
        self.success_hold = 0
        center = np.array([self.cfg.flock_center_x, self.cfg.flock_center_y], dtype=np.float64)
        self.goal = np.array([self.cfg.goal_x, self.cfg.goal_y], dtype=np.float32)
        compact_radius = self.cfg.population_scaled_radius(self.num_sheep)
        if scenario == 'split' and self.num_sheep >= 6:
            n1 = self.num_sheep // 2
            n2 = self.num_sheep - n1
            lateral = perp(unit(self.goal.astype(np.float64) - center))
            half_gap = 0.5 * self.cfg.split_center_gap
            centers = [center + half_gap * lateral, center - half_gap * lateral]
            r1 = self.cfg.population_scaled_radius(n1, factor=self.cfg.split_radius_factor)
            r2 = self.cfg.population_scaled_radius(n2, factor=self.cfg.split_radius_factor)
            p1 = self._sample_cluster(centers[0], n1, radius=r1)
            p2 = self._sample_cluster(centers[1], n2, radius=r2)
            sheep_pos = np.concatenate([p1, p2], axis=0)
        else:
            sheep_pos = self._sample_cluster(center, self.num_sheep, radius=compact_radius)
        if scenario == 'single_outlier' and self.num_sheep >= 2:
            sheep_pos[-1] = self._make_outlier(center, side=+1.0, base_radius=compact_radius)
        elif scenario == 'double_outlier' and self.num_sheep >= 3:
            sheep_pos[-1] = self._make_outlier(center, side=+1.0, base_radius=compact_radius)
            sheep_pos[-2] = self._make_outlier(center, side=-1.0, base_radius=compact_radius)
        common = 0.15 * unit(self.goal.astype(np.float64) - center)
        sheep_vel = np.tile(common[None, :], (self.num_sheep, 1))
        sheep_vel += self.rng.normal(0.0, 0.04, size=sheep_vel.shape)
        g = unit(self.goal.astype(np.float64) - np.mean(sheep_pos, axis=0))
        side = perp(g)
        c_s = np.mean(sheep_pos, axis=0)
        rear_extent = max(0.0, float(-np.min((sheep_pos - c_s[None, :]) @ g)))
        dog_center = c_s - (rear_extent + self.cfg.dog_start_margin) * g
        offsets = np.linspace(-(self.num_dogs - 1) / 2.0, (self.num_dogs - 1) / 2.0, self.num_dogs)
        dog_pos = np.stack([dog_center + o * self.cfg.dog_lateral_spacing * side for o in offsets], axis=0)
        dog_pos += self.rng.normal(0.0, 4.0, size=dog_pos.shape)
        self.sheep_pos = self._clip_world(sheep_pos).astype(np.float32)
        self.sheep_vel = sheep_vel.astype(np.float32)
        self.dog_pos = self._clip_world(dog_pos).astype(np.float32)
        self.dog_vel = np.zeros((self.num_dogs, 2), dtype=np.float32)
        self.dog_path_length = np.zeros(self.num_dogs, dtype=np.float64)
        self._last_direct_dog_mask = np.zeros((self.num_dogs, self.num_sheep), dtype=bool)
        self._last_repulsive_load = np.zeros(self.num_sheep, dtype=np.float32)
        self._last_graph_mask = self.social_graph_mask()
        return self.state()

    def _sample_cluster(self, center: np.ndarray, n: int, radius: float) -> np.ndarray:
        placed: List[np.ndarray] = []
        for _ in range(n):
            p = None
            for _attempt in range(500):
                cand = sample_disk(self.rng, center, radius)
                cand = self._clip_world(cand)
                if not placed:
                    p = cand
                    break
                d = np.linalg.norm(np.asarray(placed) - cand[None, :], axis=1)
                if np.all(d >= self.cfg.initial_min_sheep_distance):
                    p = cand
                    break
            if p is None:
                p = self._clip_world(sample_disk(self.rng, center, radius))
            placed.append(np.asarray(p, dtype=np.float64))
        return np.asarray(placed, dtype=np.float64)

    def _make_outlier(self, center: np.ndarray, side: float, base_radius: Optional[float]=None) -> np.ndarray:
        g = unit(self.goal.astype(np.float64) - center)
        s = perp(g)
        r = float(self.cfg.population_scaled_radius(self.num_sheep) if base_radius is None else base_radius)
        lateral = r + float(self.rng.uniform(60.0, 90.0))
        offset = side * lateral * s
        back = float(self.rng.uniform(-25.0, 45.0)) * g
        return self._clip_world(center + offset + back)

    def _clip_world(self, p: np.ndarray) -> np.ndarray:
        m = 10.0
        return np.clip(np.asarray(p, dtype=np.float64), m, self.cfg.world_size - m)

    def social_graph_mask(self) -> np.ndarray:
        p = self.sheep_pos.astype(np.float64)
        n = len(p)
        if n <= 1:
            return np.zeros((n, n), dtype=bool)
        delta = p[:, None, :] - p[None, :, :]
        dist = np.linalg.norm(delta, axis=-1)
        np.fill_diagonal(dist, np.inf)
        raw = dist <= self.cfg.sheep_social_radius
        k = min(self.cfg.sheep_max_neighbors, n - 1)
        if k >= n - 1:
            return raw
        directed = np.zeros_like(raw)
        for i in range(n):
            cand = np.where(raw[i])[0]
            if cand.size:
                keep = cand[np.argsort(dist[i, cand])[:k]]
                directed[i, keep] = True
        return np.logical_or(directed, directed.T)

    def flock_center(self) -> np.ndarray:
        return np.mean(self.sheep_pos, axis=0).astype(np.float32)

    def flock_velocity(self) -> np.ndarray:
        return np.mean(self.sheep_vel, axis=0).astype(np.float32)

    def hull_indices(self) -> np.ndarray:
        return convex_hull_indices(self.sheep_pos)

    def dispersion(self) -> float:
        c = self.flock_center()
        return float(np.mean(np.linalg.norm(self.sheep_pos - c[None, :], axis=1)))

    def max_radius(self) -> float:
        c = self.flock_center()
        return float(np.max(np.linalg.norm(self.sheep_pos - c[None, :], axis=1)))

    def outlier_mask(self) -> np.ndarray:
        c = self.flock_center()
        radial = np.linalg.norm(self.sheep_pos - c[None, :], axis=1)
        graph = self.social_graph_mask()
        degree = graph.sum(axis=1)
        return np.logical_or(radial > self.cfg.outlier_radius, degree == 0)

    def all_inside_goal(self) -> bool:
        d = np.linalg.norm(self.sheep_pos - self.goal[None, :], axis=1)
        return bool(np.all(d <= self.cfg.goal_radius))

    def mean_goal_distance(self) -> float:
        return float(np.mean(np.linalg.norm(self.sheep_pos - self.goal[None, :], axis=1)))

    def centroid_goal_distance(self) -> float:
        return float(np.linalg.norm(self.flock_center() - self.goal))

    def state(self) -> Dict[str, np.ndarray]:
        return {'sheep_pos': self.sheep_pos.copy(), 'sheep_vel': self.sheep_vel.copy(), 'dog_pos': self.dog_pos.copy(), 'dog_vel': self.dog_vel.copy(), 'goal': self.goal.copy(), 'social_graph': self.social_graph_mask().copy(), 'outlier_mask': self.outlier_mask().copy()}

    def step(self, dog_actions: np.ndarray) -> Tuple[Dict[str, np.ndarray], bool, Dict[str, float]]:
        a = np.asarray(dog_actions, dtype=np.float64)
        if a.shape != (self.num_dogs, 2):
            raise ValueError(f'dog_actions must have shape {(self.num_dogs, 2)}, got {a.shape}')
        a = np.clip(a, -1.0, 1.0)
        self._update_dogs(a)
        self._update_sheep()
        self.primitive_step += 1
        done = self.primitive_step >= self.cfg.max_steps
        info = self.metrics()
        return (self.state(), done, info)

    def _update_dogs(self, actions: np.ndarray) -> None:
        desired = actions * self.cfg.dog_max_speed
        repel = np.zeros_like(desired)
        for i in range(self.num_dogs):
            for j in range(i + 1, self.num_dogs):
                delta = self.dog_pos[i].astype(np.float64) - self.dog_pos[j].astype(np.float64)
                d = float(np.linalg.norm(delta))
                if d < self.cfg.dog_collision_radius and d > 1e-09:
                    mag = self.cfg.dog_collision_gain * (1.0 - d / self.cfg.dog_collision_radius)
                    f = mag * delta / d
                    repel[i] += f
                    repel[j] -= f
        vel = desired + repel
        speed = np.linalg.norm(vel, axis=1)
        idx = speed > self.cfg.dog_max_speed
        if np.any(idx):
            vel[idx] *= (self.cfg.dog_max_speed / speed[idx])[:, None]
        old_pos = self.dog_pos.astype(np.float64).copy()
        self.dog_vel = vel.astype(np.float32)
        self.dog_pos = self._clip_world(old_pos + self.cfg.dt * vel).astype(np.float32)
        self.dog_path_length += np.linalg.norm(self.dog_pos.astype(np.float64) - old_pos, axis=1)

    def _boundary_force(self, p: np.ndarray) -> np.ndarray:
        m = self.cfg.boundary_margin
        w = self.cfg.world_size
        f = np.zeros(2, dtype=np.float64)
        if p[0] < m:
            f[0] += (m - p[0]) / m
        if p[0] > w - m:
            f[0] -= (p[0] - (w - m)) / m
        if p[1] < m:
            f[1] += (m - p[1]) / m
        if p[1] > w - m:
            f[1] -= (p[1] - (w - m)) / m
        return self.cfg.boundary_gain * f

    def _compute_sheep_forces(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = self.sheep_pos.astype(np.float64)
        v = self.sheep_vel.astype(np.float64)
        n = self.num_sheep
        graph = self.social_graph_mask()
        rep = np.zeros((n, 2), dtype=np.float64)
        att = np.zeros((n, 2), dtype=np.float64)
        ali = np.zeros((n, 2), dtype=np.float64)
        rep_load = np.zeros(n, dtype=np.float64)
        d0 = self.cfg.sheep_rest_distance
        for i in range(n):
            for j in range(i + 1, n):
                if not graph[i, j]:
                    continue
                delta = p[i] - p[j]
                d = float(np.linalg.norm(delta))
                if d < 1e-09:
                    ang = 2.0 * math.pi * (i + 1) / max(2, n + 1)
                    u = np.array([math.cos(ang), math.sin(ang)])
                    d = 1e-09
                else:
                    u = delta / d
                if d < d0:
                    mag = self.cfg.k_repulsion * (d0 - d)
                    pair = mag * u
                    rep[i] += pair
                    rep[j] -= pair
                    rep_load[i] += mag
                    rep_load[j] += mag
                elif d < self.cfg.sheep_social_radius:
                    mag = self.cfg.k_attraction * (d - d0)
                    pair = mag * u
                    att[i] -= pair
                    att[j] += pair
        for i in range(n):
            nbr = np.where(graph[i])[0]
            if nbr.size:
                ali[i] = self.cfg.k_alignment * (np.mean(v[nbr], axis=0) - v[i])
        dog_force = np.zeros((n, 2), dtype=np.float64)
        direct = np.zeros((self.num_dogs, n), dtype=bool)
        rd = self.cfg.dog_influence_radius
        for d_idx in range(self.num_dogs):
            dp = self.dog_pos[d_idx].astype(np.float64)
            delta = p - dp[None, :]
            dist = np.linalg.norm(delta, axis=1)
            active = dist < rd
            direct[d_idx] = active
            for i in np.where(active)[0]:
                q = max(0.0, 1.0 - float(dist[i]) / rd)
                dog_force[i] += self.cfg.k_dog * q ** self.cfg.dog_force_power * unit(delta[i])
        damping = -self.cfg.sheep_damping * v
        boundary = np.stack([self._boundary_force(x) for x in p], axis=0)
        total = rep + att + ali + dog_force + damping + boundary
        norms = np.linalg.norm(total, axis=1)
        idx = norms > self.cfg.sheep_max_accel
        if np.any(idx):
            total[idx] *= (self.cfg.sheep_max_accel / norms[idx])[:, None]
        self._last_direct_dog_mask = direct
        self._last_repulsive_load = rep_load.astype(np.float32)
        self._last_graph_mask = graph
        return (total, rep_load, graph)

    def _update_sheep(self) -> None:
        dt = self.cfg.dt / self.cfg.physics_substeps
        for _ in range(self.cfg.physics_substeps):
            total, _, _ = self._compute_sheep_forces()
            vel = self.sheep_vel.astype(np.float64) + dt * total
            speed = np.linalg.norm(vel, axis=1)
            idx = speed > self.cfg.sheep_max_speed
            if np.any(idx):
                vel[idx] *= (self.cfg.sheep_max_speed / speed[idx])[:, None]
            pos = self.sheep_pos.astype(np.float64) + dt * vel
            before = pos.copy()
            pos = self._clip_world(pos)
            hit = np.any(np.abs(before - pos) > 1e-09, axis=1)
            vel[hit] *= -0.2
            self.sheep_vel = vel.astype(np.float32)
            self.sheep_pos = pos.astype(np.float32)

    def high_level_reward(self, before: Dict[str, float], role_switches: int=0) -> Tuple[float, Dict[str, float]]:
        self.high_step += 1
        all_in = self.all_inside_goal()
        self.success_hold = self.success_hold + 1 if all_in else 0
        after = self.metrics()
        goal_progress = float(before['mean_goal_distance'] - after['mean_goal_distance'])
        centroid_progress = float(before['centroid_goal_distance'] - after['centroid_goal_distance'])
        dispersion = float(after['dispersion'])
        outliers = int(after['outlier_count'])
        collisions = int(after['dog_collision_count'])
        reward = self.cfg.reward_goal_progress * goal_progress + self.cfg.reward_centroid_progress * centroid_progress - self.cfg.penalty_dispersion * dispersion - self.cfg.penalty_outlier * outliers - self.cfg.penalty_dog_collision * collisions - self.cfg.penalty_time
        if after['success']:
            reward += self.cfg.reward_success
        parts = {'goal_progress': goal_progress, 'centroid_progress': centroid_progress, 'dispersion': dispersion, 'outlier_count': float(outliers), 'dog_collision_count': float(collisions), 'role_switches': float(role_switches), 'reward': float(reward)}
        return (float(reward), parts)

    def metrics(self) -> Dict[str, float]:
        c = self.flock_center()
        d_goal = np.linalg.norm(self.sheep_pos - self.goal[None, :], axis=1)
        radial = np.linalg.norm(self.sheep_pos - c[None, :], axis=1)
        graph = self.social_graph_mask()
        out = self.outlier_mask()
        dog_d = np.linalg.norm(self.dog_pos[:, None, :] - self.dog_pos[None, :, :], axis=-1)
        dog_collisions = 0
        for i in range(self.num_dogs):
            for j in range(i + 1, self.num_dogs):
                dog_collisions += int(dog_d[i, j] < self.cfg.dog_collision_radius)
        all_in = bool(np.all(d_goal <= self.cfg.goal_radius))
        success = self.success_hold >= self.cfg.success_hold_high_steps
        return {'mean_goal_distance': float(np.mean(d_goal)), 'max_goal_distance': float(np.max(d_goal)), 'centroid_goal_distance': float(np.linalg.norm(c - self.goal)), 'dispersion': float(np.mean(radial)), 'max_radius': float(np.max(radial)), 'outlier_count': float(np.sum(out)), 'graph_edge_count': float(np.sum(graph) / 2), 'dog_collision_count': float(dog_collisions), 'avg_dog_path_length': float(np.mean(self.dog_path_length)), 'all_inside_goal': float(all_in), 'success': float(success), 'primitive_step': float(self.primitive_step), 'high_step': float(self.high_step)}

    def direct_actuation_sets(self) -> List[set]:
        return [set(np.where(self._last_direct_dog_mask[j])[0].tolist()) for j in range(self.num_dogs)]

    def actuation_overlap(self) -> float:
        sets = self.direct_actuation_sets()
        if len(sets) <= 1:
            return 0.0
        vals = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                u = sets[i] | sets[j]
                vals.append(0.0 if not u else len(sets[i] & sets[j]) / len(u))
        return float(np.mean(vals)) if vals else 0.0
