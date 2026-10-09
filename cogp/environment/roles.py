"""Shared task-to-low-level role interface."""

from __future__ import annotations
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from cogp.environment.config import PlannerConfig
from cogp.environment.geometry import unit, perp
from cogp.environment.role_geometry import boundary_relative_anchor, collect_recovery_target

class Role(IntEnum):
    COLLECT = 0
    REAR_DRIVE = 1
    LEFT_FLANK = 2
    RIGHT_FLANK = 3
ROLE_NAMES = {int(Role.COLLECT): 'Collect', int(Role.REAR_DRIVE): 'RearDrive', int(Role.LEFT_FLANK): 'LeftFlank', int(Role.RIGHT_FLANK): 'RightFlank'}

@dataclass
class HighLevelAction:
    roles: np.ndarray
    targets: np.ndarray
    offsets: np.ndarray

class RoleWaypointDecoder:

    def __init__(self, cfg: Optional[PlannerConfig]=None):
        self.cfg = cfg if cfg is not None else PlannerConfig()

    def decode_with_task_targets(self, env, action: HighLevelAction) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        c = env.flock_center().astype(np.float64)
        goal = env.goal.astype(np.float64)
        g = unit(goal - c)
        if np.linalg.norm(g) < 1e-09:
            g = np.array([0.0, -1.0], dtype=np.float64)
        roles = np.asarray(action.roles, dtype=np.int64)
        collecting = bool(np.any(roles == int(Role.COLLECT)))
        mode = 'Collecting' if collecting else 'Driving'
        lookahead = self.cfg.collect_lookahead if collecting else self.cfg.drive_lookahead
        flock_pb = np.clip(c + lookahead * g, 15.0, env.cfg.world_size - 15.0).astype(np.float32)
        task_targets = np.repeat(flock_pb[None, :], env.num_dogs, axis=0).astype(np.float32)
        anchors = np.zeros((env.num_dogs, 2), dtype=np.float32)
        collect_targets = [int(np.clip(action.targets[j], 0, env.num_sheep - 1)) for j in range(env.num_dogs) if int(roles[j]) == int(Role.COLLECT)]
        for j in range(env.num_dogs):
            role = int(roles[j])
            residual = np.asarray(action.offsets[j], dtype=np.float64)
            if role == int(Role.COLLECT):
                q = int(np.clip(action.targets[j], 0, env.num_sheep - 1))
                exclude_other = [x for x in collect_targets if x != q]
                recovery = collect_recovery_target(env, q, exclude_indices=exclude_other)
                task_targets[j] = recovery
                anchors[j] = boundary_relative_anchor(env, 'collect', self.cfg, flock_pb=recovery, target_index=q, residual=residual, collect_exclude_indices=exclude_other)
            elif role == int(Role.REAR_DRIVE):
                anchors[j] = boundary_relative_anchor(env, 'rear', self.cfg, flock_pb=flock_pb, residual=residual)
            elif role == int(Role.LEFT_FLANK):
                anchors[j] = boundary_relative_anchor(env, 'left', self.cfg, flock_pb=flock_pb, residual=residual)
            elif role == int(Role.RIGHT_FLANK):
                anchors[j] = boundary_relative_anchor(env, 'right', self.cfg, flock_pb=flock_pb, residual=residual)
            else:
                raise ValueError(f'Unknown role {role}')
        return (flock_pb, task_targets, anchors, mode)

    def decode(self, env, action: HighLevelAction) -> Tuple[np.ndarray, np.ndarray, str]:
        flock_pb, _, anchors, mode = self.decode_with_task_targets(env, action)
        return (flock_pb, anchors, mode)
