"""Shared fixed-reference Drive segment state."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from cogp.environment.config import PlannerConfig
from cogp.environment.geometry import unit

@dataclass
class SharedDriveSegment:
    index: int
    pb: np.ndarray
    start_primitive_step: int
    age: int = 0

class SharedDriveSegmentManager:

    def __init__(self, planner_cfg: PlannerConfig, *, segment_steps: int=60, cap_to_goal: bool=False):
        self.cfg = planner_cfg
        self.segment_steps = max(1, int(segment_steps))
        self.cap_to_goal = bool(cap_to_goal)
        self.current: Optional[SharedDriveSegment] = None
        self.next_index = 0
        self.started_count = 0
        self.completed_count = 0
        self.interrupted_count = 0
        self.last_interrupt_reason: Optional[str] = None

    def reset(self) -> None:
        self.current = None
        self.next_index = 0
        self.started_count = 0
        self.completed_count = 0
        self.interrupted_count = 0
        self.last_interrupt_reason = None

    def _compute_pb(self, env) -> np.ndarray:
        c = env.flock_center().astype(np.float64)
        goal = env.goal.astype(np.float64)
        delta = goal - c
        d = float(np.linalg.norm(delta))
        g = unit(delta)
        if np.linalg.norm(g) < 1e-09:
            g = np.array([0.0, -1.0], dtype=np.float64)
        lookahead = float(self.cfg.drive_lookahead)
        if self.cap_to_goal:
            lookahead = min(lookahead, d)
        pb = c + lookahead * g
        return np.clip(pb, 15.0, env.cfg.world_size - 15.0).astype(np.float32)

    def ensure(self, env) -> Tuple[np.ndarray, bool]:
        if self.current is None:
            seg = SharedDriveSegment(index=int(self.next_index), pb=self._compute_pb(env), start_primitive_step=int(env.primitive_step), age=0)
            self.current = seg
            self.next_index += 1
            self.started_count += 1
            return (seg.pb.copy(), True)
        return (self.current.pb.copy(), False)

    def after_primitive(self) -> bool:
        if self.current is None:
            return False
        self.current.age += 1
        if self.current.age >= self.segment_steps:
            self.completed_count += 1
            self.current = None
            return True
        return False

    def interrupt(self, reason: str) -> bool:
        if self.current is None:
            return False
        self.current = None
        self.interrupted_count += 1
        self.last_interrupt_reason = str(reason)
        return True

    @property
    def age(self) -> int:
        return 0 if self.current is None else int(self.current.age)

    @property
    def index(self) -> int:
        return -1 if self.current is None else int(self.current.index)
