"""Autonomous successor acquisition, with exact reward settlement."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math
import numpy as np
from cogp.environment.config import PlannerConfig
from cogp.environment.topology import graph_lcc_fraction
from cogp.environment.geometry import unit
from cogp.control.safety import DogDogSafetyShield
from cogp.environment.role_geometry import boundary_relative_anchor
from cogp.environment.roles import HighLevelAction, Role
ROLE_NAME = {int(Role.REAR_DRIVE): 'rear', int(Role.LEFT_FLANK): 'left', int(Role.RIGHT_FLANK): 'right'}

@dataclass
class AcquisitionConfig:
    stage_offset: float = 30.0
    ready_radius: float = 20.0
    pair_error_max: float = 25.0
    ready_hold_high_steps: int = 2
    topology_lcc_min: float = 0.9
    abort_lcc: float = 0.86
    nav_clearance: float = 88.0
    arc_extra: float = 12.0
    arc_angle_tol_deg: float = 18.0
    radial_tol: float = 8.0
    arc_lookahead_deg: float = 22.0
    approach_slow_radius: float = 35.0
    min_speed_fraction: float = 0.22
    timeout_primitives: int = 300

def compute_drive_pb(env, cfg: PlannerConfig, *, cap_to_goal: bool=False) -> np.ndarray:
    c = env.flock_center().astype(np.float64)
    goal = env.goal.astype(np.float64)
    delta = goal - c
    d = float(np.linalg.norm(delta))
    g = unit(delta)
    if np.linalg.norm(g) < 1e-09:
        g = np.array([0.0, -1.0], dtype=np.float64)
    lookahead = float(cfg.drive_lookahead)
    if cap_to_goal:
        lookahead = min(lookahead, d)
    pb = c + lookahead * g
    return np.clip(pb, 15.0, env.cfg.world_size - 15.0).astype(np.float32)

def drive_anchors_for_fixed_pb(env, action: HighLevelAction, cfg: PlannerConfig, pb: np.ndarray) -> np.ndarray:
    roles = np.asarray(action.roles, dtype=np.int64).reshape(env.num_dogs)
    offsets = np.asarray(action.offsets, dtype=np.float32).reshape(env.num_dogs, 2)
    anchors = np.zeros((env.num_dogs, 2), dtype=np.float32)
    for j in range(env.num_dogs):
        role = int(roles[j])
        if role not in ROLE_NAME:
            raise ValueError(f'ACQUIRE expects drive-only roles, got role={role}')
        anchors[j] = boundary_relative_anchor(env, ROLE_NAME[role], cfg, flock_pb=np.asarray(pb, dtype=np.float32), residual=offsets[j])
    return anchors

def role_aware_stage_points(env, action: HighLevelAction, cfg: PlannerConfig, pb: np.ndarray, *, stage_offset: float=30.0) -> Tuple[np.ndarray, np.ndarray]:
    anchors = drive_anchors_for_fixed_pb(env, action, cfg, pb).astype(np.float64)
    c = env.flock_center().astype(np.float64)
    g = unit(env.goal.astype(np.float64) - c)
    if np.linalg.norm(g) < 1e-09:
        g = np.array([0.0, -1.0], dtype=np.float64)
    stage = np.zeros_like(anchors)
    for j, a in enumerate(anchors):
        outward = unit(a - c)
        if np.linalg.norm(outward) < 1e-09:
            outward = -g
        stage[j] = a + float(stage_offset) * outward
    stage = np.clip(stage, 10.0, env.cfg.world_size - 10.0)
    return (anchors.astype(np.float32), stage.astype(np.float32))

def acquisition_readiness(env, stage_points: np.ndarray, *, lcc_min: float=0.9, ready_radius: float=20.0, pair_error_max: float=25.0) -> Dict[str, float]:
    dog = env.dog_pos.astype(np.float64)
    stage = np.asarray(stage_points, dtype=np.float64).reshape(env.num_dogs, 2)
    d = np.linalg.norm(dog - stage, axis=1)
    lcc = float(graph_lcc_fraction(env))
    outliers = int(np.sum(env.outlier_mask()))
    pair_error = 0.0
    if env.num_dogs == 2:
        dog_pair = float(np.linalg.norm(dog[0] - dog[1]))
        stage_pair = float(np.linalg.norm(stage[0] - stage[1]))
        pair_error = abs(dog_pair - stage_pair)
    ready = bool(lcc >= float(lcc_min) and outliers == 0 and (float(np.max(d)) <= float(ready_radius)) and (pair_error <= float(pair_error_max)))
    return {'lcc': lcc, 'outliers': float(outliers), 'stage_dist_max': float(np.max(d)), 'stage_dist_mean': float(np.mean(d)), 'stage_dist_sum': float(np.sum(d)), 'pair_error': float(pair_error), 'ready': float(ready)}

def _signed_angle(u: np.ndarray, v: np.ndarray) -> float:
    u = unit(np.asarray(u, dtype=np.float64))
    v = unit(np.asarray(v, dtype=np.float64))
    if np.linalg.norm(u) < 1e-09 or np.linalg.norm(v) < 1e-09:
        return 0.0
    cross = float(u[0] * v[1] - u[1] * v[0])
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(math.atan2(cross, dot))

def _rotate(u: np.ndarray, angle: float) -> np.ndarray:
    ca, sa = (math.cos(angle), math.sin(angle))
    x, y = (float(u[0]), float(u[1]))
    return np.array([ca * x - sa * y, sa * x + ca * y], dtype=np.float64)

class SafePolarAcquisitionController:

    def __init__(self, planner_cfg: PlannerConfig, cfg: Optional[AcquisitionConfig]=None):
        self.planner_cfg = planner_cfg
        self.cfg = cfg if cfg is not None else AcquisitionConfig()

    def _hull_radius(self, env) -> float:
        c = env.flock_center().astype(np.float64)
        return float(np.max(np.linalg.norm(env.sheep_pos.astype(np.float64) - c[None, :], axis=1)))

    def targets(self, env, action: HighLevelAction, pb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return role_aware_stage_points(env, action, self.planner_cfg, pb, stage_offset=self.cfg.stage_offset)

    def primitive_actions(self, env, action: HighLevelAction, pb: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        _, stage = self.targets(env, action, pb)
        c = env.flock_center().astype(np.float64)
        dog = env.dog_pos.astype(np.float64)
        hull_r = self._hull_radius(env)
        stage_r = np.linalg.norm(stage.astype(np.float64) - c[None, :], axis=1)
        safe_arc_r = max(hull_r + float(self.cfg.nav_clearance), float(np.max(stage_r)) + float(self.cfg.arc_extra))
        actions = np.zeros((env.num_dogs, 2), dtype=np.float32)
        radial_out_count = 0
        arc_count = 0
        approach_count = 0
        target_dist = []
        for j in range(env.num_dogs):
            p = dog[j]
            t = stage[j].astype(np.float64)
            rel = p - c
            r = float(np.linalg.norm(rel))
            t_rel = t - c
            if r < 1e-06:
                rel = unit(t_rel)
                if np.linalg.norm(rel) < 1e-09:
                    rel = np.array([1.0, 0.0], dtype=np.float64)
                r = 1e-06
            u = unit(rel)
            ut = unit(t_rel)
            angle_err = _signed_angle(u, ut)
            angle_tol = math.radians(float(self.cfg.arc_angle_tol_deg))
            if abs(angle_err) > angle_tol and r < safe_arc_r - float(self.cfg.radial_tol):
                waypoint = c + safe_arc_r * u
                radial_out_count += 1
            elif abs(angle_err) > angle_tol:
                step = math.radians(float(self.cfg.arc_lookahead_deg))
                step = min(step, abs(angle_err)) * (1.0 if angle_err > 0.0 else -1.0)
                waypoint = c + safe_arc_r * _rotate(u, step)
                arc_count += 1
            else:
                waypoint = t
                approach_count += 1
            delta = waypoint - p
            dist = float(np.linalg.norm(delta))
            target_dist.append(float(np.linalg.norm(t - p)))
            if dist < 1e-06:
                actions[j] = 0.0
                continue
            speed_frac = 1.0
            if np.linalg.norm(t - p) < float(self.cfg.approach_slow_radius):
                speed_frac = max(float(self.cfg.min_speed_fraction), float(np.linalg.norm(t - p) / max(1e-06, self.cfg.approach_slow_radius)))
            actions[j] = np.clip(speed_frac * unit(delta), -1.0, 1.0).astype(np.float32)
        return (actions, {'acq_safe_arc_radius': float(safe_arc_r), 'acq_hull_radius': float(hull_r), 'acq_target_dist_max': float(np.max(target_dist)) if target_dist else 0.0, 'acq_nav_radial_out_fraction': float(radial_out_count / max(1, env.num_dogs)), 'acq_nav_arc_fraction': float(arc_count / max(1, env.num_dogs)), 'acq_nav_approach_fraction': float(approach_count / max(1, env.num_dogs))})

class AcquisitionSession:
    STATUS_ACTIVE = 'ACTIVE'
    STATUS_READY = 'READY'
    STATUS_ABORT_TO_RECOVER = 'ABORT_TO_RECOVER'
    STATUS_TIMEOUT = 'TIMEOUT'

    def __init__(self, planner_cfg: PlannerConfig, action: HighLevelAction, pb: np.ndarray, *, cfg: Optional[AcquisitionConfig]=None):
        self.planner_cfg = planner_cfg
        self.cfg = cfg if cfg is not None else AcquisitionConfig()
        self.action = HighLevelAction(roles=np.asarray(action.roles, dtype=np.int64).copy(), targets=np.asarray(action.targets, dtype=np.int64).copy(), offsets=np.asarray(action.offsets, dtype=np.float32).copy())
        self.pb = np.asarray(pb, dtype=np.float32).reshape(2).copy()
        self.navigator = SafePolarAcquisitionController(planner_cfg, self.cfg)
        self.elapsed_primitives = 0
        self.high_steps = 0
        self.ready_hold = 0
        self.status = self.STATUS_ACTIVE
        self.last_info: Dict[str, float] = {}

    def _topology_abort(self, env) -> bool:
        lcc = float(graph_lcc_fraction(env))
        outliers = int(np.sum(env.outlier_mask()))
        return bool(outliers > 0 or lcc < float(self.cfg.abort_lcc))

    def execute_high_level(self, env, *, option_horizon: int, shield: Optional[DogDogSafetyShield]=None) -> Tuple[float, bool, Dict[str, float]]:
        if self.status != self.STATUS_ACTIVE:
            raise RuntimeError(f'Cannot execute acquisition in status={self.status}')
        before = env.metrics()
        safety_applied = []
        safety_corr = []
        nav_diags = []
        collision_steps = 0
        primitive_done = False
        for _ in range(max(1, int(option_horizon))):
            raw, nd = self.navigator.primitive_actions(env, self.action, self.pb)
            if shield is not None:
                applied, sd = shield.project(env, raw, gains=np.ones(env.num_dogs, dtype=np.float32))
                safety_applied.append(float(bool(sd.applied)))
                safety_corr.append(float(sd.correction_norm_mean))
            else:
                applied = np.clip(raw, -1.0, 1.0).astype(np.float32)
                safety_applied.append(0.0)
                safety_corr.append(0.0)
            _, primitive_done, pinfo = env.step(applied)
            self.elapsed_primitives += 1
            nav_diags.append(nd)
            collision_steps += int(float(pinfo.get('dog_collision_count', 0.0)) > 0.0)
            if self._topology_abort(env):
                self.status = self.STATUS_ABORT_TO_RECOVER
                break
            if self.elapsed_primitives >= int(self.cfg.timeout_primitives):
                self.status = self.STATUS_TIMEOUT
                break
            if primitive_done:
                break
        reward, parts = env.high_level_reward(before, 0)
        self.high_steps += 1
        _, stage = self.navigator.targets(env, self.action, self.pb)
        ready = acquisition_readiness(env, stage, lcc_min=self.cfg.topology_lcc_min, ready_radius=self.cfg.ready_radius, pair_error_max=self.cfg.pair_error_max)
        if self.status == self.STATUS_ACTIVE:
            if ready['ready'] > 0.5:
                self.ready_hold += 1
            else:
                self.ready_hold = 0
            if self.ready_hold >= max(1, int(self.cfg.ready_hold_high_steps)):
                self.status = self.STATUS_READY
        info = env.metrics()
        info.update(parts)
        info.update(ready)
        info.update({'coordination_phase': 'ACQUIRE', 'acq_status': self.status, 'acq_elapsed_primitives': float(self.elapsed_primitives), 'acq_high_steps': float(self.high_steps), 'acq_ready_hold': float(self.ready_hold), 'acq_pb_x': float(self.pb[0]), 'acq_pb_y': float(self.pb[1]), 'acq_collision_primitive_fraction': float(collision_steps / max(1, len(nav_diags))), 'acq_safety_applied_fraction': float(np.mean(safety_applied)) if safety_applied else 0.0, 'acq_safety_correction_norm_mean': float(np.mean(safety_corr)) if safety_corr else 0.0})
        if nav_diags:
            for k in nav_diags[-1]:
                info[k] = float(np.mean([d[k] for d in nav_diags]))
        self.last_info = dict(info)
        done = bool(float(info.get('success', 0.0)) > 0.5 or primitive_done or env.primitive_step >= env.cfg.max_steps)
        return (float(reward), done, info)
