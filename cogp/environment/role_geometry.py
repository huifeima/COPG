"""Physical task targets and boundary-relative executor anchors."""

from __future__ import annotations
from typing import Dict, Iterable, Optional, Tuple
import numpy as np
from cogp.environment.geometry import unit, perp
from cogp.environment.config import PlannerConfig

def safe_frame(direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    g = unit(np.asarray(direction, dtype=np.float64))
    if np.linalg.norm(g) < 1e-09:
        g = np.array([0.0, -1.0], dtype=np.float64)
    return (g, perp(g))

def flock_task_frame(env, task_target: np.ndarray | None=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = env.flock_center().astype(np.float64)
    target = env.goal.astype(np.float64) if task_target is None else np.asarray(task_target, dtype=np.float64)
    g, s = safe_frame(target - c)
    return (c, g, s)

def directional_extents(sheep_pos: np.ndarray, center: np.ndarray, g: np.ndarray, s: np.ndarray) -> Dict[str, float]:
    rel = np.asarray(sheep_pos, dtype=np.float64) - np.asarray(center, dtype=np.float64)[None, :]
    pg = rel @ np.asarray(g, dtype=np.float64)
    ps = rel @ np.asarray(s, dtype=np.float64)
    return {'front': float(max(0.0, np.max(pg))), 'rear': float(max(0.0, -np.min(pg))), 'left': float(max(0.0, np.max(ps))), 'right': float(max(0.0, -np.min(ps)))}

def _largest_component_indices_from_mask(graph: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    n = int(graph.shape[0])
    allowed = np.asarray(allowed, dtype=bool).reshape(n)
    seen = np.zeros(n, dtype=bool)
    best: list[int] = []
    for root in range(n):
        if not allowed[root] or seen[root]:
            continue
        stack = [root]
        seen[root] = True
        comp: list[int] = []
        while stack:
            i = stack.pop()
            comp.append(int(i))
            nbrs = np.where(graph[i] & allowed)[0]
            for j in nbrs:
                j = int(j)
                if not seen[j]:
                    seen[j] = True
                    stack.append(j)
        if len(comp) > len(best):
            best = comp
    return np.asarray(best, dtype=np.int64)

def collect_main_component_indices(env, target_index: int, exclude_indices: Optional[Iterable[int]]=None) -> np.ndarray:
    n = int(env.num_sheep)
    q = int(np.clip(int(target_index), 0, max(0, n - 1)))
    allowed = np.ones(n, dtype=bool)
    allowed[q] = False
    if exclude_indices is not None:
        for x in exclude_indices:
            xi = int(x)
            if 0 <= xi < n:
                allowed[xi] = False
    remaining = np.where(allowed)[0]
    if remaining.size == 0:
        return np.asarray([q], dtype=np.int64)
    graph = np.asarray(env.social_graph_mask(), dtype=bool)
    comp = _largest_component_indices_from_mask(graph, allowed)
    if comp.size == 0:
        comp = remaining.astype(np.int64)
    return comp

def collect_recovery_target(env, target_index: int, exclude_indices: Optional[Iterable[int]]=None) -> np.ndarray:
    idx = collect_main_component_indices(env, target_index, exclude_indices=exclude_indices)
    p = env.sheep_pos.astype(np.float64)
    target = np.mean(p[idx], axis=0)
    return np.clip(target, 10.0, env.cfg.world_size - 10.0).astype(np.float32)

def target_connected_to_main(env, target_index: int, exclude_indices: Optional[Iterable[int]]=None) -> bool:
    n = int(env.num_sheep)
    if n <= 1:
        return True
    q = int(np.clip(int(target_index), 0, n - 1))
    main = collect_main_component_indices(env, q, exclude_indices=exclude_indices)
    graph = np.asarray(env.social_graph_mask(), dtype=bool)
    if main.size == 0:
        return False
    return bool(np.any(graph[q, main]))

def boundary_relative_anchor(env, role: str, planner_cfg: PlannerConfig, flock_pb: np.ndarray | None=None, target_index: int | None=None, residual: np.ndarray | None=None, collect_exclude_indices: Optional[Iterable[int]]=None) -> np.ndarray:
    role = str(role).lower()
    p = env.sheep_pos.astype(np.float64)
    if role == 'collect':
        if target_index is None:
            raise ValueError('collect anchor requires target_index')
        q = int(np.clip(int(target_index), 0, env.num_sheep - 1))
        recovery = collect_recovery_target(env, q, exclude_indices=collect_exclude_indices).astype(np.float64) if flock_pb is None else np.asarray(flock_pb, dtype=np.float64)
        pq = p[q]
        toward = unit(recovery - pq)
        if np.linalg.norm(toward) < 1e-09:
            c = env.flock_center().astype(np.float64)
            toward = unit(c - pq)
        if np.linalg.norm(toward) < 1e-09:
            toward = np.array([0.0, 1.0], dtype=np.float64)
        lateral = perp(toward)
        anchor = pq - planner_cfg.collect_offset * toward
        if residual is not None:
            off = np.clip(np.asarray(residual, dtype=np.float64).reshape(2), -1.0, 1.0)
            anchor = anchor + planner_cfg.anchor_residual_scale * (off[0] * toward + off[1] * lateral)
        return np.clip(anchor, 10.0, env.cfg.world_size - 10.0).astype(np.float32)
    c, g, s = flock_task_frame(env, task_target=flock_pb)
    ex = directional_extents(p, c, g, s)
    if role in ('rear', 'random'):
        anchor = c - (ex['rear'] + planner_cfg.rear_margin) * g
    elif role == 'left':
        back = planner_cfg.flank_back_fraction * (ex['rear'] + planner_cfg.rear_margin)
        anchor = c - back * g + (ex['left'] + planner_cfg.flank_margin) * s
    elif role == 'right':
        back = planner_cfg.flank_back_fraction * (ex['rear'] + planner_cfg.rear_margin)
        anchor = c - back * g - (ex['right'] + planner_cfg.flank_margin) * s
    else:
        raise ValueError(f'unknown role: {role}')
    if residual is not None:
        off = np.clip(np.asarray(residual, dtype=np.float64).reshape(2), -1.0, 1.0)
        anchor = anchor + planner_cfg.anchor_residual_scale * (off[0] * g + off[1] * s)
    return np.clip(anchor, 10.0, env.cfg.world_size - 10.0).astype(np.float32)

def boundary_extents_for_role(env, flock_pb: np.ndarray | None=None) -> Dict[str, float]:
    c, g, s = flock_task_frame(env, task_target=flock_pb)
    return directional_extents(env.sheep_pos, c, g, s)
