"""Decision context and shared frozen-coordinator state."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from cogp.environment import PlannerConfig
from cogp.environment.roles import HighLevelAction, Role
from cogp.control.acquisition import AcquisitionConfig
PHASE_RECOVER=0
PHASE_DRIVE=1
def phase_to_int(phase: str) -> int:
    return PHASE_DRIVE if str(phase).upper() == "DRIVE" else PHASE_RECOVER


@dataclass
class DecisionContext:
    previous_phase: str
    phase: str
    candidates: np.ndarray
    lcc: float
    transitioned: bool

    @property
    def phase_id(self) -> int:
        return phase_to_int(self.phase)


class FrozenSuccessorCoordinator:
    """Policy-facing coordinator preserving the validated successor protocol."""

    def __init__(
        self,
        executor,
        planner_cfg: PlannerConfig,
        *,
        gamma: float,
        phase_gate: Optional[object] = None,
        acquisition_cfg: Optional[AcquisitionConfig] = None,
        drive_cap_to_goal: bool = False,
    ):
        self.executor = executor
        self.pc = planner_cfg
        self.gamma = float(gamma)
        self.phase_gate = phase_gate
        if self.phase_gate is None:
            raise ValueError('A task phase gate is required')
        self.acq_cfg = acquisition_cfg if acquisition_cfg is not None else AcquisitionConfig()
        self.drive_cap_to_goal = bool(drive_cap_to_goal)
        self.last_roles: Optional[np.ndarray] = None
        self.acquire_started_total = 0
        self.acquire_completed_total = 0
        self.acquire_aborted_total = 0
        self.acquire_timeout_total = 0

    def reset(self, num_dogs: int) -> None:
        self.executor.reset(num_dogs)
        self.phase_gate.reset()
        self.last_roles = np.full(int(num_dogs), -1, dtype=np.int64)
        self.acquire_started_total = 0
        self.acquire_completed_total = 0
        self.acquire_aborted_total = 0
        self.acquire_timeout_total = 0

    def prepare(self, env) -> DecisionContext:
        return self.phase_gate.prepare(env)

    @staticmethod
    def action_gains(action: HighLevelAction, ctx: DecisionContext) -> np.ndarray:
        roles = np.asarray(action.roles, dtype=np.int64)
        gains = np.ones(len(roles), dtype=np.float32)
        recovering_with_candidates = ctx.phase == "RECOVER" and len(ctx.candidates) > 0
        recovery_hold = ctx.phase == "RECOVER" and len(ctx.candidates) == 0
        for j, r in enumerate(roles):
            if int(r) == int(Role.COLLECT):
                gains[j] = 1.0
            elif int(r) == int(Role.REAR_DRIVE):
                gains[j] = 0.38 if recovering_with_candidates else (0.55 if recovery_hold else 1.0)
            else:
                gains[j] = 0.20 if recovering_with_candidates else (0.25 if recovery_hold else 0.55)
        return gains

    def _accumulate(self, total: float, discount: float, reward: float) -> Tuple[float, float]:
        total += discount * float(reward)
        discount *= self.gamma
        return total, discount

