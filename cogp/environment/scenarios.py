"""Canonical and scaling-world reset procedures."""

from __future__ import annotations
import math
from typing import List
import numpy as np
from cogp.environment.geometry import unit, perp

def _place_dogs_behind_flock(env, rng: np.random.Generator) -> None:
    p = env.sheep_pos.astype(np.float64)
    c = p.mean(axis=0)
    g = unit(env.goal.astype(np.float64) - c)
    if np.linalg.norm(g) < 1e-09:
        g = np.asarray([0.0, -1.0], dtype=np.float64)
    s = perp(g)
    rear_extent = max(0.0, float(-np.min((p - c[None, :]) @ g)))
    dc = c - (rear_extent + float(env.cfg.dog_start_margin)) * g
    offs = np.linspace(-(env.num_dogs - 1) / 2.0, (env.num_dogs - 1) / 2.0, env.num_dogs)
    dog = np.stack([dc + o * float(env.cfg.dog_lateral_spacing) * s for o in offs], axis=0)
    dog += rng.normal(0.0, 4.0, size=dog.shape)
    env.dog_pos = np.clip(dog, 10.0, env.cfg.world_size - 10.0).astype(np.float32)
    env.dog_vel = np.zeros((env.num_dogs, 2), dtype=np.float32)
    env.dog_path_length = np.zeros(env.num_dogs, dtype=np.float64)

def _finish_reset(env, scenario: str):
    env.current_scenario = str(scenario)
    env.primitive_step = 0
    env.high_step = 0
    env.success_hold = 0
    env._last_direct_dog_mask = np.zeros((env.num_dogs, env.num_sheep), dtype=bool)
    env._last_repulsive_load = np.zeros(env.num_sheep, dtype=np.float32)
    env._last_graph_mask = env.social_graph_mask()

def reset_triple_matched(env, seed: int):
    env.reset(seed=int(seed), scenario='compact', num_sheep=env.num_sheep, num_dogs=env.num_dogs)
    if env.num_sheep < 4:
        raise ValueError('triple_matched requires >=4 sheep')
    rng = np.random.default_rng(int(seed) + 1000091)
    main = np.arange(env.num_sheep - 3)
    c = env.sheep_pos[main].mean(axis=0).astype(np.float64)
    g = unit(env.goal.astype(np.float64) - c)
    if np.linalg.norm(g) < 1e-09:
        g = np.asarray([0.0, -1.0], dtype=np.float64)
    s = perp(g)
    radius = float(env.cfg.population_scaled_radius(env.num_sheep)) + 75.0
    for q, ang in zip(range(env.num_sheep - 3, env.num_sheep), (70.0, 110.0, -90.0)):
        th = math.radians(ang)
        d = math.cos(th) * g + math.sin(th) * s
        env.sheep_pos[q] = np.clip(c + radius * d + rng.normal(0, 2, size=2), 20.0, env.cfg.world_size - 20.0)
        env.sheep_vel[q] = rng.normal(0, 0.03, size=2)
    _place_dogs_behind_flock(env, rng)
    _finish_reset(env, 'triple_matched')

def _disk_point(rng, center, radius):
    th = rng.uniform(0, 2 * np.pi)
    rr = radius * np.sqrt(rng.uniform(0, 1))
    return center + rr * np.asarray([np.cos(th), np.sin(th)])

def _sample_minsep(rng, n, sampler, min_sep):
    pts = []
    for _ in range(n):
        p = None
        for _a in range(1200):
            c = np.asarray(sampler(), dtype=np.float64)
            if not pts or np.all(np.linalg.norm(np.asarray(pts) - c[None, :], axis=1) >= min_sep):
                p = c
                break
        if p is None:
            p = np.asarray(sampler(), dtype=np.float64)
        pts.append(p)
    return np.asarray(pts, dtype=np.float64)

def reset_random_scatter(env, seed: int, mode: str='multi_cluster', radius: float=220.0, cluster_count_min: int=2, cluster_count_max: int=5, cluster_radius: float=48.0):
    env.reset(seed=int(seed), scenario='compact', num_sheep=env.num_sheep, num_dogs=env.num_dogs)
    rng = np.random.default_rng(int(seed) + 1000173)
    env.rng = rng
    center = np.asarray([env.cfg.flock_center_x, env.cfg.flock_center_y], dtype=np.float64)
    margin = 20.0
    min_sep = max(8.0, 0.85 * env.cfg.initial_min_sheep_distance)
    clip = lambda p: np.clip(np.asarray(p, dtype=np.float64), margin, env.cfg.world_size - margin)
    if mode == 'wide_disk':
        sheep = _sample_minsep(rng, env.num_sheep, lambda: clip(_disk_point(rng, center, radius)), min_sep)
    elif mode == 'multi_cluster':
        lo = max(2, int(cluster_count_min))
        hi = max(lo, int(cluster_count_max))
        k = int(rng.integers(lo, hi + 1))
        centers = _sample_minsep(rng, k, lambda: clip(_disk_point(rng, center, 0.78 * radius)), max(70.0, 1.6 * cluster_radius))
        ids = np.arange(env.num_sheep)
        rng.shuffle(ids)
        groups = np.array_split(ids, k)
        sheep = np.zeros((env.num_sheep, 2), dtype=np.float64)
        for ci, grp in enumerate(groups):
            local: List[np.ndarray] = []
            for idx in grp.tolist():
                p = None
                for _a in range(1000):
                    cand = clip(_disk_point(rng, centers[ci], cluster_radius))
                    if not local or np.all(np.linalg.norm(np.asarray(local) - cand[None, :], axis=1) >= min_sep):
                        p = cand
                        break
                if p is None:
                    p = clip(_disk_point(rng, centers[ci], cluster_radius))
                sheep[int(idx)] = p
                local.append(p)
    else:
        raise ValueError(mode)
    env.sheep_pos = sheep.astype(np.float32)
    g = unit(env.goal.astype(np.float64) - sheep.mean(0))
    if np.linalg.norm(g) < 1e-09:
        g = np.asarray([0.0, -1.0])
    vel = np.tile((0.15 * g)[None, :], (env.num_sheep, 1)) + rng.normal(0, 0.04, size=(env.num_sheep, 2))
    env.sheep_vel = vel.astype(np.float32)
    _place_dogs_behind_flock(env, rng)
    _finish_reset(env, f'OOD_{mode}')

def reset_scaling_case(env, seed: int, scenario: str):
    if scenario in ('compact', 'single_outlier', 'double_outlier', 'split'):
        env.reset(seed=int(seed), scenario=scenario, num_sheep=env.num_sheep, num_dogs=env.num_dogs)
    elif scenario == 'triple_matched':
        reset_triple_matched(env, seed)
    elif scenario == 'multi_cluster':
        reset_random_scatter(env, seed, 'multi_cluster')
    elif scenario == 'wide_disk':
        reset_random_scatter(env, seed, 'wide_disk')
    else:
        raise ValueError(scenario)
