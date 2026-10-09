"""Frozen BASC control, staging, Rear protection, and shared Drive execution."""
from __future__ import annotations
from cogp.control.weights import load_frozen_policy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math, numpy as np
from cogp.control.observation import BASCWPObservationBuilder
from cogp.control.canonical_drive import CanonicalDriveController
from cogp.environment.config import PlannerConfig
from cogp.environment.roles import HighLevelAction, Role, RoleWaypointDecoder
from cogp.control.safety import DogDogSafetyShield
from cogp.environment.role_geometry import boundary_relative_anchor, collect_main_component_indices
from cogp.control.drive import SharedDriveSegmentManager, SharedDriveSegment
from cogp.control.canonical_collect import CanonicalCollectController
from cogp.environment.geometry import unit
ROLE_TO_DRIVE_NAME = {int(Role.REAR_DRIVE): 'rear', int(Role.LEFT_FLANK): 'left', int(Role.RIGHT_FLANK): 'right'}

@dataclass
class DogActionDiagnostic:
    dog_index: int
    role: int
    role_name: str
    canonicalized: bool
    action_gap: float
    action_cosine: float
    action_norm: float

class FrozenPolicyController:

    def __init__(self, collect_checkpoint: str, drive_checkpoint: str, *, planner_cfg: Optional[PlannerConfig]=None, deterministic: bool=True, device: str='auto', canonicalize_right: bool=True):
        self.collect_model = load_frozen_policy(collect_checkpoint, device=device)
        self.drive_model = load_frozen_policy(drive_checkpoint, device=device)
        self.drive = CanonicalDriveController(self.drive_model, canonicalize_right=canonicalize_right)
        self.cfg = planner_cfg if planner_cfg is not None else PlannerConfig()
        self.det = bool(deterministic)
        self.obs_builder = BASCWPObservationBuilder()
        self.canonicalize_right = bool(canonicalize_right)

    @staticmethod
    def _clip_action(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float32).reshape(2)
        n = float(np.linalg.norm(a))
        if n > 1.0:
            a = a / max(n, 1e-09)
        return np.clip(a, -1.0, 1.0).astype(np.float32)

def _pairwise_min_distance(points: np.ndarray) -> float:
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 2:
        return float('inf')
    best = float('inf')
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            best = min(best, float(np.linalg.norm(p[i] - p[j])))
    return best

def _pairwise_overlap_fraction(points: np.ndarray, radius: float) -> float:
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 2:
        return 0.0
    vals = []
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            vals.append(float(np.linalg.norm(p[i] - p[j]) < radius))
    return float(np.mean(vals)) if vals else 0.0

def _dog_sheep_min(env) -> float:
    if env.num_dogs < 1 or env.num_sheep < 1:
        return float('inf')
    d = env.dog_pos[:, None, :].astype(np.float64) - env.sheep_pos[None, :, :].astype(np.float64)
    return float(np.min(np.linalg.norm(d, axis=-1)))

class FixedReferenceExecutor:

    def __init__(self, low_level: FrozenPolicyController, planner_cfg: Optional[PlannerConfig]=None, *, drive_segment_steps: int=60, drive_cap_to_goal: bool=False, anchor_overlap_radius: float=40.0, use_safety_shield: bool=True, safety_soft_radius: float=42.0, safety_hard_radius: float=20.0, safety_separation_speed: float=2.5, safety_prediction_horizon: float=0.75):
        self.low_level = low_level
        self.cfg = planner_cfg if planner_cfg is not None else PlannerConfig()
        self.decoder = RoleWaypointDecoder(self.cfg)
        self.drive_segments = SharedDriveSegmentManager(self.cfg, segment_steps=drive_segment_steps, cap_to_goal=drive_cap_to_goal)
        self.prev_roles: Optional[np.ndarray] = None
        self.prev_targets: Optional[np.ndarray] = None
        self.anchor_overlap_radius = float(anchor_overlap_radius)
        self.use_safety_shield = bool(use_safety_shield)
        self.shield = DogDogSafetyShield(soft_radius=safety_soft_radius, hard_radius=safety_hard_radius, separation_speed=safety_separation_speed, prediction_horizon=safety_prediction_horizon)

    def reset(self, num_dogs: int) -> None:
        self.prev_roles = np.full(int(num_dogs), -1, dtype=np.int64)
        self.prev_targets = np.full(int(num_dogs), -1, dtype=np.int64)
        self.drive_segments.reset()

    def _decode_fixed_drive_pb(self, env, action: HighLevelAction, pb: np.ndarray):
        roles = np.asarray(action.roles, dtype=np.int64).reshape(env.num_dogs)
        if np.any(roles == int(Role.COLLECT)):
            raise ValueError('fixed DRIVE decoder received Collect role')
        pb = np.asarray(pb, dtype=np.float32).reshape(2)
        task_targets = np.repeat(pb[None, :], env.num_dogs, axis=0).astype(np.float32)
        anchors = np.zeros((env.num_dogs, 2), dtype=np.float32)
        for j in range(env.num_dogs):
            role = int(roles[j])
            residual = np.asarray(action.offsets[j], dtype=np.float64)
            if role == int(Role.REAR_DRIVE):
                name = 'rear'
            elif role == int(Role.LEFT_FLANK):
                name = 'left'
            elif role == int(Role.RIGHT_FLANK):
                name = 'right'
            else:
                raise ValueError(f'Unknown DRIVE role id {role}')
            anchors[j] = boundary_relative_anchor(env, name, self.cfg, flock_pb=pb, residual=residual)
        return (pb.copy(), task_targets, anchors, 'DrivingFixedSegment')

    def execute_role(self, env, action: HighLevelAction, *, action_gains: Optional[np.ndarray]=None, coordination_phase: str='UNKNOWN') -> Tuple[float, bool, Dict[str, float]]:
        if self.prev_roles is None or len(self.prev_roles) != env.num_dogs:
            self.reset(env.num_dogs)
        roles = np.asarray(action.roles, dtype=np.int64).reshape(env.num_dogs)
        targets = np.asarray(action.targets, dtype=np.int64).reshape(env.num_dogs)
        if action_gains is None:
            action_gains = np.ones(env.num_dogs, dtype=np.float32)
        action_gains = np.clip(np.asarray(action_gains, dtype=np.float32).reshape(env.num_dogs), 0.0, 1.0)
        contains_collect = bool(np.any(roles == int(Role.COLLECT)))
        use_fixed_drive = str(coordination_phase).upper() == 'DRIVE' and (not contains_collect)
        interrupted_at_boundary = 0
        if not use_fixed_drive:
            interrupted_at_boundary = int(self.drive_segments.interrupt('collect_role' if contains_collect else f'phase_{coordination_phase}'))
        before = env.metrics()
        switches = int(np.sum((self.prev_roles >= 0) & (self.prev_roles != roles)))
        collect_target_switches = 0
        if self.prev_targets is not None:
            mask = (roles == int(Role.COLLECT)) & (self.prev_roles == int(Role.COLLECT))
            collect_target_switches = int(np.sum(mask & (self.prev_targets != targets)))
        overlaps: List[float] = []
        anchor_overlaps: List[float] = []
        anchor_min_dists: List[float] = []
        dog_min_dists: List[float] = []
        dog_sheep_mins: List[float] = []
        action_gaps: List[float] = []
        action_coses: List[float] = []
        safety_applied: List[float] = []
        safety_corr_mean: List[float] = []
        safety_corr_max: List[float] = []
        safety_pred_before: List[float] = []
        safety_pred_after: List[float] = []
        drive_segment_ages: List[float] = []
        drive_pb_center_dists: List[float] = []
        fixed_drive_steps = 0
        segment_created_steps = 0
        segment_expired_steps = 0
        canonical_count = 0
        drive_step_count = 0
        collision_primitive_steps = 0
        mode = 'Unknown'
        last_task_targets = None
        last_anchors = None
        for _ in range(self.cfg.option_horizon):
            if use_fixed_drive:
                pb, created = self.drive_segments.ensure(env)
                flock_pb, task_targets, anchors, mode = self._decode_fixed_drive_pb(env, action, pb)
                fixed_drive_steps += 1
                segment_created_steps += int(created)
                drive_segment_ages.append(float(self.drive_segments.age))
                drive_pb_center_dists.append(float(np.linalg.norm(env.flock_center() - pb)))
            else:
                flock_pb, task_targets, anchors, mode = self.decoder.decode_with_task_targets(env, action)
            last_task_targets = task_targets
            last_anchors = anchors
            dog_actions, diags = self.low_level.actions(env, action, task_targets, anchors, return_diagnostics=True)
            if self.use_safety_shield:
                applied_actions, sd = self.shield.project(env, dog_actions, gains=action_gains)
            else:
                applied_actions = np.clip(dog_actions * action_gains[:, None], -1.0, 1.0).astype(np.float32)

                class _D:
                    applied = False
                    correction_norm_mean = 0.0
                    correction_norm_max = 0.0
                    min_predicted_distance_before = float('inf')
                    min_predicted_distance_after = float('inf')
                sd = _D()
            _, primitive_done, primitive_info = env.step(applied_actions)
            if use_fixed_drive:
                segment_expired_steps += int(self.drive_segments.after_primitive())
            overlaps.append(float(env.actuation_overlap()))
            anchor_overlaps.append(_pairwise_overlap_fraction(anchors, self.anchor_overlap_radius))
            anchor_min_dists.append(_pairwise_min_distance(anchors))
            dog_min_dists.append(_pairwise_min_distance(env.dog_pos))
            dog_sheep_mins.append(_dog_sheep_min(env))
            collision_primitive_steps += int(float(primitive_info.get('dog_collision_count', 0.0)) > 0.0)
            safety_applied.append(float(bool(sd.applied)))
            safety_corr_mean.append(float(sd.correction_norm_mean))
            safety_corr_max.append(float(sd.correction_norm_max))
            safety_pred_before.append(float(sd.min_predicted_distance_before))
            safety_pred_after.append(float(sd.min_predicted_distance_after))
            for d in diags:
                if d.role != int(Role.COLLECT):
                    drive_step_count += 1
                    canonical_count += int(d.canonicalized)
                    action_gaps.append(float(d.action_gap))
                    action_coses.append(float(d.action_cosine))
            if primitive_done:
                break
        reward, parts = env.high_level_reward(before, switches)
        self.prev_roles = roles.copy()
        self.prev_targets = targets.copy()
        info = env.metrics()
        info.update(parts)
        info.update({'mode': mode, 'coordination_phase': str(coordination_phase), 'actuation_overlap': float(np.mean(overlaps)) if overlaps else 0.0, 'actuation_overlap_max': float(np.max(overlaps)) if overlaps else 0.0, 'anchor_overlap': float(np.mean(anchor_overlaps)) if anchor_overlaps else 0.0, 'anchor_overlap_max': float(np.max(anchor_overlaps)) if anchor_overlaps else 0.0, 'anchor_pair_min_dist': float(np.min(anchor_min_dists)) if anchor_min_dists else float('inf'), 'dog_pair_min_dist': float(np.min(dog_min_dists)) if dog_min_dists else float('inf'), 'dog_sheep_min': float(np.min(dog_sheep_mins)) if dog_sheep_mins else float('inf'), 'role_switches': float(switches), 'collect_target_switches': float(collect_target_switches), 'canonical_drive_fraction': float(canonical_count / max(1, drive_step_count)), 'drive_native_vs_canonical_action_gap': float(np.mean(action_gaps)) if action_gaps else 0.0, 'drive_native_vs_canonical_action_cos': float(np.mean(action_coses)) if action_coses else 1.0, 'collision_primitive_fraction': float(collision_primitive_steps / max(1, len(overlaps))), 'action_gain_mean': float(np.mean(action_gains)), 'action_gain_min': float(np.min(action_gains)), 'safety_applied_fraction': float(np.mean(safety_applied)) if safety_applied else 0.0, 'safety_correction_norm_mean': float(np.mean(safety_corr_mean)) if safety_corr_mean else 0.0, 'safety_correction_norm_max': float(np.max(safety_corr_max)) if safety_corr_max else 0.0, 'safety_predicted_pair_min_before': float(np.min(safety_pred_before)) if safety_pred_before else float('inf'), 'safety_predicted_pair_min_after': float(np.min(safety_pred_after)) if safety_pred_after else float('inf'), 'success_hold': float(env.success_hold), 'fixed_drive_fraction': float(fixed_drive_steps / max(1, len(overlaps))), 'drive_segment_created_steps': float(segment_created_steps), 'drive_segment_expired_steps': float(segment_expired_steps), 'drive_segment_interrupt_at_boundary': float(interrupted_at_boundary), 'drive_segment_age_mean': float(np.mean(drive_segment_ages)) if drive_segment_ages else 0.0, 'drive_segment_age_end': float(self.drive_segments.age), 'drive_segment_index_end': float(self.drive_segments.index), 'drive_segment_started_total': float(self.drive_segments.started_count), 'drive_segment_completed_total': float(self.drive_segments.completed_count), 'drive_segment_interrupted_total': float(self.drive_segments.interrupted_count), 'drive_pb_center_dist_mean': float(np.mean(drive_pb_center_dists)) if drive_pb_center_dists else 0.0})
        if last_task_targets is not None:
            info['task_target_spread'] = float(np.mean(np.linalg.norm(last_task_targets - last_task_targets.mean(axis=0), axis=1)))
        if last_anchors is not None:
            info['anchor_spread'] = float(np.mean(np.linalg.norm(last_anchors - last_anchors.mean(axis=0), axis=1)))
        collect_targets = targets[roles == int(Role.COLLECT)]
        if len(collect_targets) <= 1:
            conflict = 0.0
        else:
            conflict = float(len(set(map(int, collect_targets.tolist()))) < len(collect_targets))
        info['collect_target_conflict'] = conflict
        done = bool(info['success'] > 0.5 or env.primitive_step >= env.cfg.max_steps)
        return (float(reward), done, info)

class SuccessorExecutor(FixedReferenceExecutor):

    def prime_drive_segment(self, env, pb: np.ndarray, *, replace: bool=False) -> bool:
        if self.drive_segments.current is not None:
            if not replace:
                return False
            self.drive_segments.interrupt('prime_replace')
        pb = np.asarray(pb, dtype=np.float32).reshape(2)
        seg = SharedDriveSegment(index=int(self.drive_segments.next_index), pb=pb.copy(), start_primitive_step=int(env.primitive_step), age=0)
        self.drive_segments.current = seg
        self.drive_segments.next_index += 1
        self.drive_segments.started_count += 1
        return True

class CanonicalPolicyController(FrozenPolicyController):

    def __init__(self, collect_checkpoint: str, drive_checkpoint: str, *, planner_cfg: Optional[PlannerConfig]=None, deterministic: bool=True, device: str='auto', canonicalize_right: bool=True, canonicalize_collect: bool=True, collect_easy_side_sign: int=+1, collect_side_deadband: float=1.0):
        super().__init__(collect_checkpoint, drive_checkpoint, planner_cfg=planner_cfg, deterministic=deterministic, device=device, canonicalize_right=canonicalize_right)
        self.collect = CanonicalCollectController(self.collect_model, canonicalize_hard_side=canonicalize_collect, easy_side_sign=collect_easy_side_sign, side_deadband=collect_side_deadband)
        self.canonicalize_collect = bool(canonicalize_collect)
        self.reset_collect_diagnostics()

    def reset_collect_diagnostics(self) -> None:
        self._collect_predictions = 0
        self._collect_canonicalized = 0
        self._collect_positive_side = 0
        self._collect_negative_side = 0
        self._collect_gap_sum = 0.0
        self._collect_cos_sum = 0.0

    def collect_diagnostics(self) -> dict:
        n = max(1, int(self._collect_predictions))
        return {'collect_prediction_count': int(self._collect_predictions), 'canonical_collect_count': int(self._collect_canonicalized), 'canonical_collect_fraction': float(self._collect_canonicalized / n), 'collect_positive_side_fraction': float(self._collect_positive_side / n), 'collect_negative_side_fraction': float(self._collect_negative_side / n), 'collect_native_vs_canonical_action_gap': float(self._collect_gap_sum / n), 'collect_native_vs_canonical_action_cos': float(self._collect_cos_sum / n)}

class SurvivorStagingController(CanonicalPolicyController):

    def __init__(self, collect_checkpoint: str, drive_checkpoint: str, *, planner_cfg: Optional[PlannerConfig]=None, deterministic: bool=True, device: str='auto', canonicalize_right: bool=True, canonicalize_collect: bool=True, collect_easy_side_sign: int=+1, collect_side_deadband: float=1.0, enable_survivor_staging: bool=True, staging_enter_distance: float=75.0, staging_exit_distance: float=20.0, staging_exit_hold_primitives: int=2, staging_max_action: float=1.0, staging_min_action: float=0.25, staging_slow_radius: float=45.0, staging_require_single_outlier: bool=True):
        super().__init__(collect_checkpoint, drive_checkpoint, planner_cfg=planner_cfg, deterministic=deterministic, device=device, canonicalize_right=canonicalize_right, canonicalize_collect=canonicalize_collect, collect_easy_side_sign=collect_easy_side_sign, collect_side_deadband=collect_side_deadband)
        self.enable_survivor_staging = bool(enable_survivor_staging)
        self.staging_enter_distance = float(staging_enter_distance)
        self.staging_exit_distance = float(staging_exit_distance)
        self.staging_exit_hold_primitives = max(1, int(staging_exit_hold_primitives))
        self.staging_max_action = float(staging_max_action)
        self.staging_min_action = float(staging_min_action)
        self.staging_slow_radius = max(1e-06, float(staging_slow_radius))
        self.staging_require_single_outlier = bool(staging_require_single_outlier)
        if self.staging_enter_distance <= self.staging_exit_distance:
            raise ValueError('staging_enter_distance must exceed staging_exit_distance')
        if not 0.0 < self.staging_min_action <= self.staging_max_action <= 1.0:
            raise ValueError('require 0 < staging_min_action <= staging_max_action <= 1')
        self.reset_staging(0)

    def reset_staging(self, num_dogs: int) -> None:
        m = max(0, int(num_dogs))
        self._stage_active = np.zeros(m, dtype=bool)
        self._stage_armed = np.zeros(m, dtype=bool)
        self._stage_target = np.full(m, -1, dtype=np.int64)
        self._stage_hold = np.zeros(m, dtype=np.int64)
        self._prev_high_collect_count = 0
        self._prev_high_roles: Optional[np.ndarray] = None
        self._prev_high_targets: Optional[np.ndarray] = None
        self._stage_transition_events = 0
        self._stage_trigger_count = 0
        self._stage_completed_count = 0
        self._stage_skipped_near_count = 0
        self._stage_abort_count = 0
        self._stage_primitive_count = 0
        self._stage_start_distances: List[float] = []
        self._stage_completion_distances: List[float] = []
        self._stage_last_distance = float('nan')

    def _ensure_stage_shape(self, num_dogs: int) -> None:
        if len(self._stage_active) != int(num_dogs):
            self.reset_staging(int(num_dogs))

    def begin_high_level_option(self, env, action: HighLevelAction, *, coordination_phase: str='UNKNOWN') -> None:
        self._ensure_stage_shape(env.num_dogs)
        roles = np.asarray(action.roles, dtype=np.int64).reshape(env.num_dogs)
        targets = np.asarray(action.targets, dtype=np.int64).reshape(env.num_dogs)
        collect_idx = np.where(roles == int(Role.COLLECT))[0]
        n_collect = int(len(collect_idx))
        for j in range(env.num_dogs):
            if not (self._stage_active[j] or self._stage_armed[j]):
                continue
            same = int(roles[j]) == int(Role.COLLECT) and int(targets[j]) == int(self._stage_target[j]) and (n_collect == 1)
            if not same:
                self._stage_active[j] = False
                self._stage_armed[j] = False
                self._stage_target[j] = -1
                self._stage_hold[j] = 0
                self._stage_abort_count += 1
        phase_ok = str(coordination_phase).upper() == 'RECOVER'
        outlier_ok = True
        if self.staging_require_single_outlier:
            outlier_ok = int(np.sum(env.outlier_mask())) == 1
        transition = self.enable_survivor_staging and phase_ok and outlier_ok and (self._prev_high_collect_count >= 2) and (n_collect == 1)
        if transition:
            j = int(collect_idx[0])
            q = int(targets[j])
            self._stage_armed[j] = True
            self._stage_target[j] = q
            self._stage_hold[j] = 0
            self._stage_transition_events += 1
        self._prev_high_collect_count = n_collect
        self._prev_high_roles = roles.copy()
        self._prev_high_targets = targets.copy()

    def staging_diagnostics(self) -> dict:
        active = int(np.sum(self._stage_active)) if len(self._stage_active) else 0
        starts = self._stage_start_distances
        ends = self._stage_completion_distances
        return {'staging_transition_events': int(self._stage_transition_events), 'staging_trigger_count': int(self._stage_trigger_count), 'staging_completed_count': int(self._stage_completed_count), 'staging_skipped_near_count': int(self._stage_skipped_near_count), 'staging_abort_count': int(self._stage_abort_count), 'staging_primitive_count': int(self._stage_primitive_count), 'staging_episode': int(self._stage_trigger_count > 0), 'staging_active_end': int(active), 'staging_start_distance_mean': float(np.mean(starts)) if starts else 0.0, 'staging_start_distance_max': float(np.max(starts)) if starts else 0.0, 'staging_completion_distance_mean': float(np.mean(ends)) if ends else 0.0, 'staging_last_distance': float(self._stage_last_distance)}

class StagingExecutor(SuccessorExecutor):

    def reset(self, num_dogs: int) -> None:
        super().reset(num_dogs)
        if hasattr(self.low_level, 'reset_staging'):
            self.low_level.reset_staging(num_dogs)

    def execute_role(self, env, action: HighLevelAction, *, action_gains: Optional[np.ndarray]=None, coordination_phase: str='UNKNOWN'):
        if hasattr(self.low_level, 'begin_high_level_option'):
            self.low_level.begin_high_level_option(env, action, coordination_phase=coordination_phase)
        return super().execute_role(env, action, action_gains=action_gains, coordination_phase=coordination_phase)

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
    return np.asarray([ca * x - sa * y, sa * x + ca * y], dtype=np.float64)

def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    p = np.asarray(point, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ab = b - a
    den = float(np.dot(ab, ab))
    if den <= 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / den, 0.0, 1.0))
    q = a + t * ab
    return float(np.linalg.norm(p - q))

def safe_polar_collect_step(dog_pos: np.ndarray, anchor: np.ndarray, main_center: np.ndarray, safe_radius: float, *, arc_extra: float=12.0, arc_angle_tol_deg: float=18.0, radial_tol: float=8.0, arc_lookahead_deg: float=22.0, approach_slow_radius: float=35.0, min_action: float=0.22, max_action: float=1.0) -> Tuple[np.ndarray, str, dict]:
    p = np.asarray(dog_pos, dtype=np.float64)
    t = np.asarray(anchor, dtype=np.float64)
    c = np.asarray(main_center, dtype=np.float64)
    safe_r = max(1e-06, float(safe_radius))
    route_r = safe_r + max(0.0, float(arc_extra))
    to_target = t - p
    target_dist = float(np.linalg.norm(to_target))
    if target_dist <= 1e-09:
        return (np.zeros(2, dtype=np.float32), 'hold', {'safe_radius': safe_r, 'route_radius': route_r, 'segment_clearance': float('inf'), 'target_distance': 0.0})
    rel = p - c
    t_rel = t - c
    r = float(np.linalg.norm(rel))
    rt = float(np.linalg.norm(t_rel))
    if r < 1e-09:
        rel = unit(-t_rel)
        if np.linalg.norm(rel) < 1e-09:
            rel = np.asarray([1.0, 0.0], dtype=np.float64)
        r = 1e-09
    u = unit(rel)
    ut = unit(t_rel)
    if np.linalg.norm(ut) < 1e-09:
        ut = u.copy()
    segment_clearance = _point_segment_distance(c, p, t)
    direct_safe = bool(segment_clearance >= safe_r and r >= safe_r - float(radial_tol) and (rt >= safe_r - float(radial_tol)))
    angle_err = _signed_angle(u, ut)
    angle_tol = math.radians(float(arc_angle_tol_deg))
    if direct_safe:
        waypoint = t
        mode = 'approach'
    elif r < route_r - float(radial_tol):
        waypoint = c + route_r * u
        mode = 'radial_out'
    elif abs(angle_err) > angle_tol:
        step = math.radians(float(arc_lookahead_deg))
        step = min(step, abs(angle_err)) * (1.0 if angle_err > 0.0 else -1.0)
        turn_r = max(route_r, r)
        waypoint = c + turn_r * _rotate(u, step)
        mode = 'arc'
    else:
        waypoint = t
        mode = 'approach'
    delta = waypoint - p
    d = float(np.linalg.norm(delta))
    if d <= 1e-09:
        action = np.zeros(2, dtype=np.float32)
    else:
        speed = float(max_action)
        if target_dist < float(approach_slow_radius):
            speed = max(float(min_action), float(max_action) * target_dist / max(1e-06, float(approach_slow_radius)))
        speed = float(np.clip(speed, float(min_action), float(max_action)))
        action = (speed * unit(delta)).astype(np.float32)
    return (action, mode, {'safe_radius': safe_r, 'route_radius': route_r, 'segment_clearance': float(segment_clearance), 'target_distance': target_dist, 'dog_main_radius': r, 'anchor_main_radius': rt, 'angle_error_deg': float(math.degrees(angle_err))})

class SafePolarStagingController(SurvivorStagingController):

    def __init__(self, collect_checkpoint: str, drive_checkpoint: str, *, planner_cfg: Optional[PlannerConfig]=None, deterministic: bool=True, device: str='auto', canonicalize_right: bool=True, canonicalize_collect: bool=True, collect_easy_side_sign: int=+1, collect_side_deadband: float=1.0, enable_survivor_staging: bool=True, staging_enter_distance: float=75.0, staging_exit_distance: float=20.0, staging_exit_hold_primitives: int=2, staging_max_action: float=1.0, staging_min_action: float=0.22, staging_slow_radius: float=35.0, staging_require_single_outlier: bool=True, staging_nav_clearance: float=88.0, staging_arc_extra: float=12.0, staging_arc_angle_tol_deg: float=18.0, staging_radial_tol: float=8.0, staging_arc_lookahead_deg: float=22.0):
        super().__init__(collect_checkpoint, drive_checkpoint, planner_cfg=planner_cfg, deterministic=deterministic, device=device, canonicalize_right=canonicalize_right, canonicalize_collect=canonicalize_collect, collect_easy_side_sign=collect_easy_side_sign, collect_side_deadband=collect_side_deadband, enable_survivor_staging=enable_survivor_staging, staging_enter_distance=staging_enter_distance, staging_exit_distance=staging_exit_distance, staging_exit_hold_primitives=staging_exit_hold_primitives, staging_max_action=staging_max_action, staging_min_action=staging_min_action, staging_slow_radius=staging_slow_radius, staging_require_single_outlier=staging_require_single_outlier)
        self.staging_nav_clearance = float(staging_nav_clearance)
        self.staging_arc_extra = float(staging_arc_extra)
        self.staging_arc_angle_tol_deg = float(staging_arc_angle_tol_deg)
        self.staging_radial_tol = float(staging_radial_tol)
        self.staging_arc_lookahead_deg = float(staging_arc_lookahead_deg)
        if self.staging_nav_clearance <= 0.0:
            raise ValueError('staging_nav_clearance must be > 0')
        self.reset_staging(0)

    def reset_staging(self, num_dogs: int) -> None:
        super().reset_staging(num_dogs)
        m = max(0, int(num_dogs))
        self._stage_prev_pos = np.full((m, 2), np.nan, dtype=np.float64)
        self._stage_path_length = np.zeros(m, dtype=np.float64)
        self._stage_min_main_center_distance = np.full(m, np.inf, dtype=np.float64)
        self._stage_min_any_sheep_distance = np.full(m, np.inf, dtype=np.float64)
        self._stage_start_global_goal_distance = np.full(m, np.nan, dtype=np.float64)
        self._stage_end_global_goal_distance = np.full(m, np.nan, dtype=np.float64)
        self._stage_start_main_goal_distance = np.full(m, np.nan, dtype=np.float64)
        self._stage_end_main_goal_distance = np.full(m, np.nan, dtype=np.float64)
        self._stage_mode_counts = {'radial_out': 0, 'arc': 0, 'approach': 0, 'hold': 0}
        self._stage_segment_clearance_min = float('inf')
        self._stage_safe_radius_last = 0.0
        self._stage_route_radius_last = 0.0
        self._stage_target_dist_last = 0.0

    def _ensure_stage_shape(self, num_dogs: int) -> None:
        if len(self._stage_active) != int(num_dogs):
            self.reset_staging(int(num_dogs))

    @staticmethod
    def _main_geometry(env, target: int) -> Tuple[np.ndarray, float, np.ndarray]:
        idx = collect_main_component_indices(env, int(target))
        sheep = env.sheep_pos.astype(np.float64)
        main = sheep[idx] if len(idx) else sheep
        c = np.mean(main, axis=0)
        hull_r = float(np.max(np.linalg.norm(main - c[None, :], axis=1))) if len(main) else 0.0
        return (c, hull_r, np.asarray(idx, dtype=np.int64))

    def _note_stage_position(self, env, j: int, target: int) -> Tuple[np.ndarray, float]:
        p = env.dog_pos[j].astype(np.float64)
        c, hull_r, _ = self._main_geometry(env, target)
        prev = self._stage_prev_pos[j]
        if np.all(np.isfinite(prev)):
            self._stage_path_length[j] += float(np.linalg.norm(p - prev))
        self._stage_prev_pos[j] = p.copy()
        self._stage_min_main_center_distance[j] = min(self._stage_min_main_center_distance[j], float(np.linalg.norm(p - c)))
        sheep_d = np.linalg.norm(env.sheep_pos.astype(np.float64) - p[None, :], axis=1)
        if len(sheep_d):
            self._stage_min_any_sheep_distance[j] = min(self._stage_min_any_sheep_distance[j], float(np.min(sheep_d)))
        return (c, hull_r)

    def _start_stage_diag(self, env, j: int, target: int) -> None:
        c, _, _ = self._main_geometry(env, target)
        self._stage_prev_pos[j] = env.dog_pos[j].astype(np.float64).copy()
        self._stage_path_length[j] = 0.0
        self._stage_min_main_center_distance[j] = float(np.linalg.norm(env.dog_pos[j].astype(np.float64) - c))
        sheep_d = np.linalg.norm(env.sheep_pos.astype(np.float64) - env.dog_pos[j].astype(np.float64)[None, :], axis=1)
        self._stage_min_any_sheep_distance[j] = float(np.min(sheep_d)) if len(sheep_d) else float('inf')
        self._stage_start_global_goal_distance[j] = float(env.metrics()['centroid_goal_distance'])
        self._stage_start_main_goal_distance[j] = float(np.linalg.norm(c - env.goal.astype(np.float64)))

    def _finish_stage_diag(self, env, j: int, target: int) -> None:
        c, _, _ = self._main_geometry(env, target)
        self._stage_end_global_goal_distance[j] = float(env.metrics()['centroid_goal_distance'])
        self._stage_end_main_goal_distance[j] = float(np.linalg.norm(c - env.goal.astype(np.float64)))

    def _safe_polar_staging_action(self, env, j: int, target: int, anchor: np.ndarray) -> np.ndarray:
        c, hull_r = self._note_stage_position(env, j, target)
        safe_r = hull_r + self.staging_nav_clearance
        a, mode, d = safe_polar_collect_step(env.dog_pos[j], anchor, c, safe_r, arc_extra=self.staging_arc_extra, arc_angle_tol_deg=self.staging_arc_angle_tol_deg, radial_tol=self.staging_radial_tol, arc_lookahead_deg=self.staging_arc_lookahead_deg, approach_slow_radius=self.staging_slow_radius, min_action=self.staging_min_action, max_action=self.staging_max_action)
        self._stage_mode_counts[mode] = self._stage_mode_counts.get(mode, 0) + 1
        self._stage_segment_clearance_min = min(self._stage_segment_clearance_min, float(d['segment_clearance']))
        self._stage_safe_radius_last = float(d['safe_radius'])
        self._stage_route_radius_last = float(d['route_radius'])
        self._stage_target_dist_last = float(d['target_distance'])
        self._stage_last_distance = float(d['target_distance'])
        return a

    def staging_diagnostics(self) -> dict:
        base = super().staging_diagnostics()
        finite_path = self._stage_path_length[np.isfinite(self._stage_path_length) & (self._stage_path_length > 0)]
        finite_main = self._stage_min_main_center_distance[np.isfinite(self._stage_min_main_center_distance)]
        finite_sheep = self._stage_min_any_sheep_distance[np.isfinite(self._stage_min_any_sheep_distance)]
        sg = self._stage_start_global_goal_distance[np.isfinite(self._stage_start_global_goal_distance)]
        eg = self._stage_end_global_goal_distance[np.isfinite(self._stage_end_global_goal_distance)]
        sm = self._stage_start_main_goal_distance[np.isfinite(self._stage_start_main_goal_distance)]
        em = self._stage_end_main_goal_distance[np.isfinite(self._stage_end_main_goal_distance)]
        denom = max(1, sum(self._stage_mode_counts.values()))
        base.update({'staging_navigation': 'safe_polar', 'staging_path_length_mean': float(np.mean(finite_path)) if len(finite_path) else 0.0, 'staging_min_dog_main_center_distance': float(np.min(finite_main)) if len(finite_main) else 0.0, 'staging_min_dog_any_sheep_distance': float(np.min(finite_sheep)) if len(finite_sheep) else 0.0, 'staging_global_goal_distance_start_mean': float(np.mean(sg)) if len(sg) else 0.0, 'staging_global_goal_distance_end_mean': float(np.mean(eg)) if len(eg) else 0.0, 'staging_global_goal_distance_delta_mean': float(np.mean(eg) - np.mean(sg)) if len(sg) and len(eg) else 0.0, 'staging_main_goal_distance_start_mean': float(np.mean(sm)) if len(sm) else 0.0, 'staging_main_goal_distance_end_mean': float(np.mean(em)) if len(em) else 0.0, 'staging_main_goal_distance_delta_mean': float(np.mean(em) - np.mean(sm)) if len(sm) and len(em) else 0.0, 'staging_nav_radial_out_fraction': float(self._stage_mode_counts.get('radial_out', 0) / denom), 'staging_nav_arc_fraction': float(self._stage_mode_counts.get('arc', 0) / denom), 'staging_nav_approach_fraction': float(self._stage_mode_counts.get('approach', 0) / denom), 'staging_segment_clearance_min': float(self._stage_segment_clearance_min) if np.isfinite(self._stage_segment_clearance_min) else 0.0, 'staging_safe_radius_last': float(self._stage_safe_radius_last), 'staging_route_radius_last': float(self._stage_route_radius_last), 'staging_target_distance_last': float(self._stage_target_dist_last)})
        return base

    def actions(self, env, action: HighLevelAction, task_targets: np.ndarray, dog_anchors: np.ndarray, *, return_diagnostics: bool=False):
        self._ensure_stage_shape(env.num_dogs)
        roles = np.asarray(action.roles, dtype=np.int64).reshape(env.num_dogs)
        targets = np.asarray(action.targets, dtype=np.int64).reshape(env.num_dogs)
        task_targets = np.asarray(task_targets, dtype=np.float32).reshape(env.num_dogs, 2)
        dog_anchors = np.asarray(dog_anchors, dtype=np.float32).reshape(env.num_dogs, 2)
        out = np.zeros((env.num_dogs, 2), dtype=np.float32)
        diags: List[DogActionDiagnostic] = []
        for j in range(env.num_dogs):
            role = int(roles[j])
            target = int(targets[j])
            if role == int(Role.COLLECT) and self._stage_armed[j] and (not self._stage_active[j]):
                d = float(np.linalg.norm(env.dog_pos[j].astype(np.float64) - dog_anchors[j].astype(np.float64)))
                if d > self.staging_enter_distance:
                    self._stage_active[j] = True
                    self._stage_armed[j] = False
                    self._stage_hold[j] = 0
                    self._stage_trigger_count += 1
                    self._stage_start_distances.append(d)
                    self._start_stage_diag(env, j, target)
                else:
                    self._stage_armed[j] = False
                    self._stage_target[j] = -1
                    self._stage_skipped_near_count += 1
            if role == int(Role.COLLECT) and self._stage_active[j] and (target == int(self._stage_target[j])):
                d = float(np.linalg.norm(env.dog_pos[j].astype(np.float64) - dog_anchors[j].astype(np.float64)))
                self._stage_last_distance = d
                if d <= self.staging_exit_distance:
                    self._stage_hold[j] += 1
                else:
                    self._stage_hold[j] = 0
                if self._stage_hold[j] >= self.staging_exit_hold_primitives:
                    self._note_stage_position(env, j, target)
                    self._finish_stage_diag(env, j, target)
                    self._stage_active[j] = False
                    self._stage_target[j] = -1
                    self._stage_hold[j] = 0
                    self._stage_completed_count += 1
                    self._stage_completion_distances.append(d)
                else:
                    a = self._clip_action(self._safe_polar_staging_action(env, j, target, dog_anchors[j]))
                    self._stage_primitive_count += 1
                    out[j] = a
                    diags.append(DogActionDiagnostic(dog_index=j, role=role, role_name='collect_stage_safe_polar', canonicalized=False, action_gap=0.0, action_cosine=1.0, action_norm=float(np.linalg.norm(a))))
                    continue
            obs = self.obs_builder.build(env, j, task_targets[j], dog_anchors[j])
            if role == int(Role.COLLECT):
                raw, cd = self.collect.predict(env, obs, target, deterministic=self.det, return_diagnostics=True)
                a = self._clip_action(raw)
                self._collect_predictions += 1
                self._collect_canonicalized += int(cd.canonicalized)
                self._collect_positive_side += int(cd.signed_lateral > 0.0)
                self._collect_negative_side += int(cd.signed_lateral < 0.0)
                self._collect_gap_sum += float(cd.action_gap)
                self._collect_cos_sum += float(cd.action_cosine)
                diag = DogActionDiagnostic(dog_index=j, role=role, role_name='collect', canonicalized=bool(cd.canonicalized), action_gap=float(cd.action_gap), action_cosine=float(cd.action_cosine), action_norm=float(np.linalg.norm(a)))
            elif role in ROLE_TO_DRIVE_NAME:
                name = ROLE_TO_DRIVE_NAME[role]
                raw, dd = self.drive.predict(obs, name, deterministic=self.det, return_diagnostics=True)
                a = self._clip_action(raw)
                diag = DogActionDiagnostic(dog_index=j, role=role, role_name=name, canonicalized=bool(dd.canonicalized), action_gap=float(dd.action_gap), action_cosine=float(dd.action_cosine), action_norm=float(np.linalg.norm(a)))
            else:
                raise ValueError(f'Unknown low-level role id {role}')
            out[j] = a
            diags.append(diag)
        return (out, diags) if return_diagnostics else out

def task_frame(center: np.ndarray, goal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    c = np.asarray(center, dtype=np.float64).reshape(2)
    q = np.asarray(goal, dtype=np.float64).reshape(2)
    g = unit(q - c)
    if np.linalg.norm(g) <= 1e-09:
        g = np.asarray([1.0, 0.0], dtype=np.float64)
    s = np.asarray([-g[1], g[0]], dtype=np.float64)
    return (g, s)

def rear_barrier_project_action(ppo_action: np.ndarray, goal_axis: np.ndarray, lateral_axis: np.ndarray, *, rear_long: float, target_long: float, dog_lateral: float, anchor_lateral: float, restore_slow_distance: float=45.0, min_rear_action: float=0.25, max_rear_action: float=1.0, max_lateral_action: float=0.45, lateral_anchor_blend: float=0.3, lateral_slow_distance: float=45.0) -> Tuple[np.ndarray, dict]:
    a = np.asarray(ppo_action, dtype=np.float64).reshape(2)
    g = unit(np.asarray(goal_axis, dtype=np.float64).reshape(2))
    s = unit(np.asarray(lateral_axis, dtype=np.float64).reshape(2))
    if np.linalg.norm(g) <= 1e-09:
        g = np.asarray([1.0, 0.0], dtype=np.float64)
    if np.linalg.norm(s) <= 1e-09:
        s = np.asarray([-g[1], g[0]], dtype=np.float64)
    ppo_long = float(np.dot(a, g))
    ppo_lat = float(np.dot(a, s))
    gap = max(0.0, float(rear_long) - float(target_long))
    if gap > 0.0:
        mag = gap / max(1e-06, float(restore_slow_distance))
        mag = float(np.clip(mag, float(min_rear_action), float(max_rear_action)))
        cmd_long = -mag
    else:
        cmd_long = min(0.0, ppo_long)
    lat_cap = abs(float(max_lateral_action))
    ppo_lat_bounded = float(np.clip(ppo_lat, -lat_cap, lat_cap))
    lat_err = float(anchor_lateral) - float(dog_lateral)
    anchor_lat_cmd = float(np.clip(lat_err / max(1e-06, float(lateral_slow_distance)), -lat_cap, lat_cap))
    blend = float(np.clip(lateral_anchor_blend, 0.0, 1.0))
    cmd_lat = (1.0 - blend) * ppo_lat_bounded + blend * anchor_lat_cmd
    cmd_lat = float(np.clip(cmd_lat, -lat_cap, lat_cap))
    out = cmd_long * g + cmd_lat * s
    n = float(np.linalg.norm(out))
    if n > 1.0:
        out = out / max(n, 1e-09)
    diag = {'ppo_long': ppo_long, 'ppo_lateral': ppo_lat, 'cmd_long': float(np.dot(out, g)), 'cmd_lateral': float(np.dot(out, s)), 'rear_gap': float(gap), 'lateral_error': float(lat_err), 'action_gap': float(np.linalg.norm(out - a)), 'action_cosine': float(np.dot(out, a) / max(1e-09, np.linalg.norm(out) * np.linalg.norm(a))) if np.linalg.norm(out) > 1e-09 and np.linalg.norm(a) > 1e-09 else 1.0}
    return (out.astype(np.float32), diag)

class FrozenBASCController(SafePolarStagingController):

    def __init__(self, collect_checkpoint: str, drive_checkpoint: str, *, planner_cfg: Optional[PlannerConfig]=None, deterministic: bool=True, device: str='auto', canonicalize_right: bool=True, canonicalize_collect: bool=True, collect_easy_side_sign: int=+1, collect_side_deadband: float=1.0, enable_survivor_staging: bool=True, staging_enter_distance: float=75.0, staging_exit_distance: float=20.0, staging_exit_hold_primitives: int=2, staging_max_action: float=1.0, staging_min_action: float=0.22, staging_slow_radius: float=35.0, staging_require_single_outlier: bool=True, staging_nav_clearance: float=88.0, staging_arc_extra: float=12.0, staging_arc_angle_tol_deg: float=18.0, staging_radial_tol: float=8.0, staging_arc_lookahead_deg: float=22.0, enable_rear_barrier: bool=True, rear_barrier_enter_long: float=-10.0, rear_barrier_release_long: float=-55.0, rear_barrier_min_anchor_distance: float=55.0, rear_barrier_release_hold_primitives: int=2, rear_barrier_restore_slow_distance: float=45.0, rear_barrier_min_rear_action: float=0.25, rear_barrier_max_rear_action: float=1.0, rear_barrier_max_lateral_action: float=0.45, rear_barrier_lateral_anchor_blend: float=0.3, rear_barrier_lateral_slow_distance: float=45.0):
        super().__init__(collect_checkpoint, drive_checkpoint, planner_cfg=planner_cfg, deterministic=deterministic, device=device, canonicalize_right=canonicalize_right, canonicalize_collect=canonicalize_collect, collect_easy_side_sign=collect_easy_side_sign, collect_side_deadband=collect_side_deadband, enable_survivor_staging=enable_survivor_staging, staging_enter_distance=staging_enter_distance, staging_exit_distance=staging_exit_distance, staging_exit_hold_primitives=staging_exit_hold_primitives, staging_max_action=staging_max_action, staging_min_action=staging_min_action, staging_slow_radius=staging_slow_radius, staging_require_single_outlier=staging_require_single_outlier, staging_nav_clearance=staging_nav_clearance, staging_arc_extra=staging_arc_extra, staging_arc_angle_tol_deg=staging_arc_angle_tol_deg, staging_radial_tol=staging_radial_tol, staging_arc_lookahead_deg=staging_arc_lookahead_deg)
        self.enable_rear_barrier = bool(enable_rear_barrier)
        self.rear_barrier_enter_long = float(rear_barrier_enter_long)
        self.rear_barrier_release_long = float(rear_barrier_release_long)
        self.rear_barrier_min_anchor_distance = float(rear_barrier_min_anchor_distance)
        self.rear_barrier_release_hold_primitives = max(1, int(rear_barrier_release_hold_primitives))
        self.rear_barrier_restore_slow_distance = max(1e-06, float(rear_barrier_restore_slow_distance))
        self.rear_barrier_min_rear_action = float(rear_barrier_min_rear_action)
        self.rear_barrier_max_rear_action = float(rear_barrier_max_rear_action)
        self.rear_barrier_max_lateral_action = abs(float(rear_barrier_max_lateral_action))
        self.rear_barrier_lateral_anchor_blend = float(rear_barrier_lateral_anchor_blend)
        self.rear_barrier_lateral_slow_distance = max(1e-06, float(rear_barrier_lateral_slow_distance))
        if self.rear_barrier_release_long >= self.rear_barrier_enter_long:
            raise ValueError('rear_barrier_release_long must be more rearward than rear_barrier_enter_long')
        if not 0.0 < self.rear_barrier_min_rear_action <= self.rear_barrier_max_rear_action <= 1.0:
            raise ValueError('require 0 < min_rear_action <= max_rear_action <= 1')
        if not 0.0 <= self.rear_barrier_lateral_anchor_blend <= 1.0:
            raise ValueError('rear_barrier_lateral_anchor_blend must be in [0,1]')
        self.reset_rear_barrier(0)

    def reset_rear_barrier(self, num_dogs: int) -> None:
        m = max(0, int(num_dogs))
        self._rear_barrier_active = np.zeros(m, dtype=bool)
        self._rear_barrier_hold = np.zeros(m, dtype=np.int64)
        self._rear_barrier_drive_context = False
        self._rear_barrier_start_primitive = np.full(m, -1, dtype=np.int64)
        self._rear_barrier_trigger_count = 0
        self._rear_barrier_completed_count = 0
        self._rear_barrier_abort_count = 0
        self._rear_barrier_primitive_count = 0
        self._rear_barrier_start_longs: List[float] = []
        self._rear_barrier_release_longs: List[float] = []
        self._rear_barrier_durations: List[int] = []
        self._rear_barrier_action_gaps: List[float] = []
        self._rear_barrier_action_cosines: List[float] = []
        self._rear_barrier_cmd_longs: List[float] = []
        self._rear_barrier_cmd_lats: List[float] = []
        self._rear_barrier_min_sheep_distance = float('inf')
        self._rear_barrier_max_front_long = float('-inf')
        self._rear_barrier_last_long = float('nan')
        self._rear_barrier_last_anchor_distance = float('nan')
        self._rear_barrier_last_lateral_error = float('nan')
        self._rear_barrier_last_cmd_long = 0.0
        self._rear_barrier_last_cmd_lateral = 0.0

    def _ensure_barrier_shape(self, num_dogs: int) -> None:
        if len(self._rear_barrier_active) != int(num_dogs):
            self.reset_rear_barrier(int(num_dogs))

    def begin_high_level_option(self, env, action: HighLevelAction, *, coordination_phase: str='UNKNOWN') -> None:
        super().begin_high_level_option(env, action, coordination_phase=coordination_phase)
        self._ensure_barrier_shape(env.num_dogs)
        roles = np.asarray(action.roles, dtype=np.int64).reshape(env.num_dogs)
        self._rear_barrier_drive_context = str(coordination_phase).upper() == 'DRIVE'
        for j in range(env.num_dogs):
            if not self._rear_barrier_active[j]:
                continue
            if not self._rear_barrier_drive_context or int(roles[j]) != int(Role.REAR_DRIVE):
                self._rear_barrier_active[j] = False
                self._rear_barrier_hold[j] = 0
                self._rear_barrier_abort_count += 1

    @staticmethod
    def _flock_frame(env):
        sheep = env.sheep_pos.astype(np.float64)
        c = np.mean(sheep, axis=0)
        g, s = task_frame(c, env.goal.astype(np.float64))
        return (c, g, s)

    def rear_barrier_diagnostics(self) -> dict:
        return {'rear_barrier_enabled': int(self.enable_rear_barrier), 'rear_barrier_trigger_count': int(self._rear_barrier_trigger_count), 'rear_barrier_completed_count': int(self._rear_barrier_completed_count), 'rear_barrier_abort_count': int(self._rear_barrier_abort_count), 'rear_barrier_primitive_count': int(self._rear_barrier_primitive_count), 'rear_barrier_episode': int(self._rear_barrier_trigger_count > 0), 'rear_barrier_active_end': int(np.sum(self._rear_barrier_active)) if len(self._rear_barrier_active) else 0, 'rear_barrier_start_long_mean': float(np.mean(self._rear_barrier_start_longs)) if self._rear_barrier_start_longs else 0.0, 'rear_barrier_release_long_mean': float(np.mean(self._rear_barrier_release_longs)) if self._rear_barrier_release_longs else 0.0, 'rear_barrier_duration_mean': float(np.mean(self._rear_barrier_durations)) if self._rear_barrier_durations else 0.0, 'rear_barrier_duration_max': int(np.max(self._rear_barrier_durations)) if self._rear_barrier_durations else 0, 'rear_barrier_min_dog_sheep_distance': float(self._rear_barrier_min_sheep_distance) if np.isfinite(self._rear_barrier_min_sheep_distance) else 0.0, 'rear_barrier_max_front_long': float(self._rear_barrier_max_front_long) if np.isfinite(self._rear_barrier_max_front_long) else 0.0, 'rear_barrier_action_gap_mean': float(np.mean(self._rear_barrier_action_gaps)) if self._rear_barrier_action_gaps else 0.0, 'rear_barrier_action_cosine_mean': float(np.mean(self._rear_barrier_action_cosines)) if self._rear_barrier_action_cosines else 1.0, 'rear_barrier_cmd_long_mean': float(np.mean(self._rear_barrier_cmd_longs)) if self._rear_barrier_cmd_longs else 0.0, 'rear_barrier_cmd_lateral_abs_mean': float(np.mean(np.abs(self._rear_barrier_cmd_lats))) if self._rear_barrier_cmd_lats else 0.0, 'rear_barrier_last_long': float(self._rear_barrier_last_long) if np.isfinite(self._rear_barrier_last_long) else 0.0, 'rear_barrier_last_anchor_distance': float(self._rear_barrier_last_anchor_distance) if np.isfinite(self._rear_barrier_last_anchor_distance) else 0.0, 'rear_barrier_last_lateral_error': float(self._rear_barrier_last_lateral_error) if np.isfinite(self._rear_barrier_last_lateral_error) else 0.0, 'rear_barrier_last_cmd_long': float(self._rear_barrier_last_cmd_long), 'rear_barrier_last_cmd_lateral': float(self._rear_barrier_last_cmd_lateral), 'rear_barrier_target_long': float(self.rear_barrier_release_long)}

    def actions(self, env, action: HighLevelAction, task_targets: np.ndarray, dog_anchors: np.ndarray, *, return_diagnostics: bool=False):
        out, diags = SafePolarStagingController.actions(self, env, action, task_targets, dog_anchors, return_diagnostics=True)
        self._ensure_barrier_shape(env.num_dogs)
        if not self.enable_rear_barrier or not self._rear_barrier_drive_context:
            return (out, diags) if return_diagnostics else out
        roles = np.asarray(action.roles, dtype=np.int64).reshape(env.num_dogs)
        anchors = np.asarray(dog_anchors, dtype=np.float64).reshape(env.num_dogs, 2)
        rear_idx = np.flatnonzero(roles == int(Role.REAR_DRIVE))
        if len(rear_idx) != 1:
            return (out, diags) if return_diagnostics else out
        j = int(rear_idx[0])
        c, g, s = self._flock_frame(env)
        p = env.dog_pos[j].astype(np.float64)
        a_anchor = anchors[j]
        rear_long = float(np.dot(p - c, g))
        dog_lat = float(np.dot(p - c, s))
        anchor_lat = float(np.dot(a_anchor - c, s))
        anchor_dist = float(np.linalg.norm(p - a_anchor))
        self._rear_barrier_last_long = rear_long
        self._rear_barrier_last_anchor_distance = anchor_dist
        self._rear_barrier_last_lateral_error = float(anchor_lat - dog_lat)
        if not self._rear_barrier_active[j]:
            should_enter = rear_long >= self.rear_barrier_enter_long and anchor_dist >= self.rear_barrier_min_anchor_distance
            if should_enter:
                self._rear_barrier_active[j] = True
                self._rear_barrier_hold[j] = 0
                self._rear_barrier_start_primitive[j] = int(env.primitive_step)
                self._rear_barrier_trigger_count += 1
                self._rear_barrier_start_longs.append(rear_long)
        if self._rear_barrier_active[j]:
            self._rear_barrier_max_front_long = max(self._rear_barrier_max_front_long, rear_long)
            sheep_d = np.linalg.norm(env.sheep_pos.astype(np.float64) - p[None, :], axis=1)
            if len(sheep_d):
                self._rear_barrier_min_sheep_distance = min(self._rear_barrier_min_sheep_distance, float(np.min(sheep_d)))
            if rear_long <= self.rear_barrier_release_long:
                self._rear_barrier_hold[j] += 1
            else:
                self._rear_barrier_hold[j] = 0
            if self._rear_barrier_hold[j] >= self.rear_barrier_release_hold_primitives:
                self._rear_barrier_active[j] = False
                self._rear_barrier_hold[j] = 0
                self._rear_barrier_completed_count += 1
                self._rear_barrier_release_longs.append(rear_long)
                start = int(self._rear_barrier_start_primitive[j])
                if start >= 0:
                    self._rear_barrier_durations.append(max(0, int(env.primitive_step) - start))
                self._rear_barrier_start_primitive[j] = -1
            else:
                ppo = np.asarray(out[j], dtype=np.float32).copy()
                cmd, dd = rear_barrier_project_action(ppo, g, s, rear_long=rear_long, target_long=self.rear_barrier_release_long, dog_lateral=dog_lat, anchor_lateral=anchor_lat, restore_slow_distance=self.rear_barrier_restore_slow_distance, min_rear_action=self.rear_barrier_min_rear_action, max_rear_action=self.rear_barrier_max_rear_action, max_lateral_action=self.rear_barrier_max_lateral_action, lateral_anchor_blend=self.rear_barrier_lateral_anchor_blend, lateral_slow_distance=self.rear_barrier_lateral_slow_distance)
                out[j] = self._clip_action(cmd)
                self._rear_barrier_primitive_count += 1
                self._rear_barrier_action_gaps.append(float(dd['action_gap']))
                self._rear_barrier_action_cosines.append(float(dd['action_cosine']))
                self._rear_barrier_cmd_longs.append(float(dd['cmd_long']))
                self._rear_barrier_cmd_lats.append(float(dd['cmd_lateral']))
                self._rear_barrier_last_cmd_long = float(dd['cmd_long'])
                self._rear_barrier_last_cmd_lateral = float(dd['cmd_lateral'])
                repl = DogActionDiagnostic(dog_index=j, role=int(Role.REAR_DRIVE), role_name='rear_barrier_guard', canonicalized=False, action_gap=float(dd['action_gap']), action_cosine=float(dd['action_cosine']), action_norm=float(np.linalg.norm(out[j])))
                replaced = False
                for k, diag in enumerate(diags):
                    if int(diag.dog_index) == j:
                        diags[k] = repl
                        replaced = True
                        break
                if not replaced:
                    diags.append(repl)
        return (out, diags) if return_diagnostics else out

class FrozenBASCExecutor(StagingExecutor):

    def reset(self, num_dogs: int) -> None:
        super().reset(num_dogs)
        if hasattr(self.low_level, 'reset_rear_barrier'):
            self.low_level.reset_rear_barrier(num_dogs)
EXECUTION_DEFAULTS = {'drive_segment_steps': 60, 'drive_cap_to_goal': False, 'staging_enter_distance': 75.0, 'staging_exit_distance': 20.0, 'staging_exit_hold_primitives': 2, 'staging_max_action': 1.0, 'staging_min_action': 0.22, 'staging_slow_radius': 35.0, 'staging_nav_clearance': 88.0, 'staging_arc_extra': 12.0, 'staging_arc_angle_tol_deg': 18.0, 'staging_radial_tol': 8.0, 'staging_arc_lookahead_deg': 22.0, 'rear_barrier_enter_long': -10.0, 'rear_barrier_release_long': -55.0, 'rear_barrier_min_anchor_distance': 55.0, 'rear_barrier_release_hold_primitives': 2, 'rear_barrier_restore_slow_distance': 45.0, 'rear_barrier_min_rear_action': 0.25, 'rear_barrier_max_rear_action': 1.0, 'rear_barrier_max_lateral_action': 0.45, 'rear_barrier_lateral_anchor_blend': 0.3, 'rear_barrier_lateral_slow_distance': 45.0, 'safety_soft_radius': 42.0, 'safety_hard_radius': 20.0, 'safety_separation_speed': 2.5, 'safety_prediction_horizon': 0.75}

def build_frozen_execution(collect_checkpoint: str, drive_checkpoint: str, planner_cfg: Optional[PlannerConfig]=None, *, device: str='auto') -> Tuple[FrozenBASCController, FrozenBASCExecutor]:
    pc = planner_cfg if planner_cfg is not None else PlannerConfig()
    d = EXECUTION_DEFAULTS
    controller = FrozenBASCController(collect_checkpoint, drive_checkpoint, planner_cfg=pc, deterministic=True, device=device, canonicalize_right=True, canonicalize_collect=True, enable_survivor_staging=True, staging_enter_distance=d['staging_enter_distance'], staging_exit_distance=d['staging_exit_distance'], staging_exit_hold_primitives=d['staging_exit_hold_primitives'], staging_max_action=d['staging_max_action'], staging_min_action=d['staging_min_action'], staging_slow_radius=d['staging_slow_radius'], staging_require_single_outlier=True, staging_nav_clearance=d['staging_nav_clearance'], staging_arc_extra=d['staging_arc_extra'], staging_arc_angle_tol_deg=d['staging_arc_angle_tol_deg'], staging_radial_tol=d['staging_radial_tol'], staging_arc_lookahead_deg=d['staging_arc_lookahead_deg'], enable_rear_barrier=True, rear_barrier_enter_long=d['rear_barrier_enter_long'], rear_barrier_release_long=d['rear_barrier_release_long'], rear_barrier_min_anchor_distance=d['rear_barrier_min_anchor_distance'], rear_barrier_release_hold_primitives=d['rear_barrier_release_hold_primitives'], rear_barrier_restore_slow_distance=d['rear_barrier_restore_slow_distance'], rear_barrier_min_rear_action=d['rear_barrier_min_rear_action'], rear_barrier_max_rear_action=d['rear_barrier_max_rear_action'], rear_barrier_max_lateral_action=d['rear_barrier_max_lateral_action'], rear_barrier_lateral_anchor_blend=d['rear_barrier_lateral_anchor_blend'], rear_barrier_lateral_slow_distance=d['rear_barrier_lateral_slow_distance'])
    executor = FrozenBASCExecutor(controller, pc, drive_segment_steps=d['drive_segment_steps'], drive_cap_to_goal=d['drive_cap_to_goal'], use_safety_shield=True, safety_soft_radius=d['safety_soft_radius'], safety_hard_radius=d['safety_hard_radius'], safety_separation_speed=d['safety_separation_speed'], safety_prediction_horizon=d['safety_prediction_horizon'])
    return (controller, executor)
