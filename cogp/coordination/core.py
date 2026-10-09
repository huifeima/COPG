"""Topology tasks, feasible assignments, HCCG actor and commitment execution.

The neural actor is used through ranking.py, which adds symmetric selected-pair
scoring and candidate competition. Its tensor names remain checkpoint-compatible.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import permutations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from cogp.environment import EnvConfig, PlannerConfig, GraphObservationBuilder
from cogp.environment.geometry import unit, perp
from cogp.environment.roles import HighLevelAction, Role, RoleWaypointDecoder
from cogp.environment.role_geometry import boundary_relative_anchor
from cogp.control.acquisition import AcquisitionConfig, AcquisitionSession, compute_drive_pb
from .context import DecisionContext, FrozenSuccessorCoordinator
from .value import RunningValueNorm, _orthogonal_init
TASK_COLLECT=0
TASK_REAR=1
TASK_LEFT=2
TASK_RIGHT=3
TASK_RESERVE=4
NUM_TASK_TYPES=5
PHASE_RECOVER=0
PHASE_DRIVE=1
TASK_RAW_DIM=16
DOG_TASK_GEOM_DIM=12
PAIRWISE_GEOM_DIM=10
COMP_RAW_DIM=12
COMP_EDGE_DIM=5
TRAIN_SCENARIOS=("compact", "single_outlier", "double_outlier", "split")
def parse_scenario_probs(x: Sequence[float] | str) -> np.ndarray:
    if isinstance(x, str):
        vals = [float(v.strip()) for v in x.split(",") if v.strip()]
    else:
        vals = [float(v) for v in x]
    if len(vals) != 4:
        raise ValueError("mixed probs must be compact,single_outlier,double_outlier,split")
    p = np.asarray(vals, dtype=np.float64)
    if np.any(p < 0) or float(p.sum()) <= 0:
        raise ValueError("scenario probabilities must be nonnegative and sum > 0")
    return p / p.sum()


def _components_from_mask(mask: np.ndarray) -> List[Tuple[int, ...]]:
    A = np.asarray(mask, dtype=bool)
    n = int(A.shape[0])
    seen = np.zeros(n, dtype=bool)
    comps: List[Tuple[int, ...]] = []
    for root in range(n):
        if seen[root]:
            continue
        stack = [root]
        seen[root] = True
        comp: List[int] = []
        while stack:
            i = int(stack.pop())
            comp.append(i)
            for j in np.where(A[i])[0].tolist():
                j = int(j)
                if not seen[j]:
                    seen[j] = True
                    stack.append(j)
        comps.append(tuple(sorted(comp)))
    comps.sort(key=lambda c: (-len(c), c[0] if c else -1))
    return comps


def _social_graph_np(pos_norm: np.ndarray, valid_n: int, radius_norm: float, max_neighbors: int) -> np.ndarray:
    """Reconstruct the environment social graph exactly from normalized positions."""
    P = np.asarray(pos_norm[:valid_n], dtype=np.float64)
    n = len(P)
    if n <= 1:
        return np.zeros((n, n), dtype=bool)
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    raw = d <= float(radius_norm)
    k = min(int(max_neighbors), n - 1)
    if k >= n - 1:
        return raw
    directed = np.zeros_like(raw)
    for i in range(n):
        cand = np.where(raw[i])[0]
        if cand.size:
            keep = cand[np.argsort(d[i, cand])[:k]]
            directed[i, keep] = True
    return np.logical_or(directed, directed.T)


def _parse_global_np(g: np.ndarray, max_sheep: int, max_dogs: int):
    x = np.asarray(g, dtype=np.float32)
    S, D = int(max_sheep), int(max_dogs)
    k = 0
    sheep = x[k:k + S * 4].reshape(S, 4); k += S * 4
    sm = x[k:k + S]; k += S
    dogs = x[k:k + D * 4].reshape(D, 4); k += D * 4
    dm = x[k:k + D]; k += D
    goal = x[k:k + 2]; k += 2
    extras = x[k:k + 4]
    return sheep, sm, dogs, dm, goal, extras


def _parse_global_torch(g: torch.Tensor, max_sheep: int, max_dogs: int):
    S, D = int(max_sheep), int(max_dogs)
    k = 0
    sheep = g[k:k + S * 4].reshape(S, 4); k += S * 4
    sm = g[k:k + S]; k += S
    dogs = g[k:k + D * 4].reshape(D, 4); k += D * 4
    dm = g[k:k + D]; k += D
    goal = g[k:k + 2]; k += 2
    extras = g[k:k + 4]
    return sheep, sm, dogs, dm, goal, extras


def _frame_basis_np(sheep_pos: np.ndarray, goal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = np.mean(sheep_pos, axis=0) if len(sheep_pos) else np.zeros(2, dtype=np.float64)
    a = np.asarray(goal, dtype=np.float64) - c
    n = float(np.linalg.norm(a))
    if n < 1e-9:
        a = np.asarray([0.0, -1.0], dtype=np.float64)
    else:
        a = a / n
    l = np.asarray([-a[1], a[0]], dtype=np.float64)
    return c, a, l


def _triplet_np(v: np.ndarray, axis: np.ndarray, lateral: np.ndarray) -> np.ndarray:
    vv = np.asarray(v, dtype=np.float64)
    return np.asarray([np.dot(vv, axis), np.dot(vv, lateral), np.linalg.norm(vv)], dtype=np.float32)


@dataclass(frozen=True)
class PressureTask:
    task_type: int
    component_members: Tuple[int, ...] = ()
    component_rank: int = -1
    target: int = -1
    pressure_rank: int = 0
    pressure_count: int = 1
    pressure_lateral: float = 0.0
    reserve_rank: int = 0

    @property
    def component_size(self) -> int:
        return len(self.component_members)

    def key(self) -> Tuple:
        return (
            int(self.task_type), tuple(self.component_members), int(self.component_rank),
            int(self.target), int(self.pressure_rank), int(self.pressure_count), int(self.reserve_rank),
        )

    def short(self) -> str:
        if self.task_type == TASK_COLLECT:
            return f"C{self.target}[K{self.component_rank}:P{self.pressure_rank}/{self.pressure_count}]"
        if self.task_type == TASK_REAR:
            return "Rear"
        if self.task_type == TASK_LEFT:
            return "Left"
        if self.task_type == TASK_RIGHT:
            return "Right"
        return f"Hold{self.reserve_rank}"


@dataclass(frozen=True)
class HCCGAssignmentSpec:
    tasks_by_dog: Tuple[PressureTask, ...]

    def signature(self) -> str:
        return "|".join(t.short() for t in self.tasks_by_dog)

    @property
    def collect_count(self) -> int:
        return sum(int(t.task_type) == TASK_COLLECT for t in self.tasks_by_dog)

    @property
    def reserve_count(self) -> int:
        return sum(int(t.task_type) == TASK_RESERVE for t in self.tasks_by_dog)


@dataclass
class HCCGStructure:
    active_sheep: int
    active_dogs: int
    components: Tuple[Tuple[int, ...], ...]
    main_component: Tuple[int, ...]
    social_graph: np.ndarray
    collect_tasks: Tuple[PressureTask, ...]
    center: np.ndarray
    axis: np.ndarray
    lateral: np.ndarray

    @property
    def lcc_fraction(self) -> float:
        if self.active_sheep <= 0:
            return 1.0
        return float(len(self.main_component) / max(1, self.active_sheep))

    @property
    def partition_signature(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(sorted(self.components, key=lambda c: (-len(c), c)))


class HCCGTaskBuilder:
    """Graph -> components -> stable pressure-task nodes.

    Task identity is component membership + pressure slot, not sheep ID.  The
    target sheep is an execution-time actuation point attached to that stable
    component task.
    """

    def __init__(
        self,
        *,
        world_size: float,
        social_radius: float,
        sheep_max_neighbors: int = 7,
        sheep_per_pressure_slot: int = 4,
        max_pressure_slots_per_component: int = 3,
        outlier_radius: float = 75.0,
    ):
        self.world_size = float(world_size)
        self.radius_norm = float(social_radius) / max(self.world_size, 1e-8)
        self.sheep_max_neighbors = int(sheep_max_neighbors)
        self.sheep_per_pressure_slot = max(1, int(sheep_per_pressure_slot))
        self.max_pressure_slots_per_component = max(1, int(max_pressure_slots_per_component))
        self.outlier_radius = float(outlier_radius)
        self.outlier_radius_norm = self.outlier_radius / max(self.world_size, 1e-8)

    @staticmethod
    def _select_representatives(P: np.ndarray, comp: Sequence[int], main_center: np.ndarray, k: int) -> List[int]:
        ids = [int(i) for i in comp]
        if not ids or k <= 0:
            return []
        dmain = np.linalg.norm(P[ids] - main_center[None, :], axis=1)
        first = ids[int(np.argmax(dmain))]
        chosen = [first]
        remaining = [i for i in ids if i != first]
        while remaining and len(chosen) < int(k):
            best, best_key = None, None
            for q in remaining:
                sep = min(float(np.linalg.norm(P[q] - P[z])) for z in chosen)
                away = float(np.linalg.norm(P[q] - main_center))
                key = (sep, away, -q)
                if best_key is None or key > best_key:
                    best_key, best = key, q
            chosen.append(int(best)); remaining.remove(int(best))
        return chosen

    def build_from_global(self, g: np.ndarray, max_sheep: int, max_dogs: int) -> HCCGStructure:
        sheep, sm, dogs, dm, goal, _ = _parse_global_np(g, max_sheep, max_dogs)
        n = int(np.sum(sm > 0.5)); m = int(np.sum(dm > 0.5))
        P = sheep[:n, :2].astype(np.float64)
        A = _social_graph_np(P, n, self.radius_norm, self.sheep_max_neighbors)
        comps = _components_from_mask(A)
        main = comps[0] if comps else tuple()
        main_center = np.mean(P[list(main)], axis=0) if main else (np.mean(P, axis=0) if n else np.zeros(2))
        center, axis, lateral = _frame_basis_np(P, goal)
        tasks: List[PressureTask] = []
        for rank, comp in enumerate(comps[1:], start=1):
            nc = len(comp)
            k = min(
                nc,
                self.max_pressure_slots_per_component,
                max(1, int(math.ceil(nc / float(self.sheep_per_pressure_slot)))),
            )
            reps = self._select_representatives(P, comp, main_center, k)
            cc = np.mean(P[list(comp)], axis=0)
            toward = main_center - cc
            nt = float(np.linalg.norm(toward))
            if nt < 1e-9:
                toward = -axis
            else:
                toward = toward / nt
            side = np.asarray([-toward[1], toward[0]], dtype=np.float64)
            radius = max(1e-6, max(float(np.linalg.norm(P[q] - cc)) for q in comp))
            # Sort pressure points laterally in the component->main frame.  This
            # makes pressure_rank stable geometrically rather than by sheep ID.
            reps = sorted(reps, key=lambda q: float(np.dot(P[q] - cc, side)))
            for r, q in enumerate(reps):
                lat = float(np.dot(P[q] - cc, side) / radius)
                tasks.append(PressureTask(
                    task_type=TASK_COLLECT,
                    component_members=tuple(comp), component_rank=rank,
                    target=int(q), pressure_rank=int(r), pressure_count=int(k),
                    pressure_lateral=float(np.clip(lat, -1.0, 1.0)),
                ))
        # Connected-but-radial-outlier fallback. This preserves the validated
        # phase semantics without making sheep ID the task identity.
        if len(comps) <= 1 and n > 1:
            radial = np.linalg.norm(P - center[None, :], axis=1)
            for rr, q in enumerate(np.where(radial > self.outlier_radius_norm)[0].tolist(), start=1):
                tasks.append(PressureTask(TASK_COLLECT, (int(q),), rr, int(q), 0, 1, 0.0, 0))
        return HCCGStructure(
            active_sheep=n, active_dogs=m, components=tuple(comps), main_component=tuple(main),
            social_graph=A, collect_tasks=tuple(tasks), center=center.astype(np.float32),
            axis=axis.astype(np.float32), lateral=lateral.astype(np.float32),
        )

    def build_from_env(self, env) -> HCCGStructure:
        # Build directly from exact env graph for phase/execution logic.
        A = np.asarray(env.social_graph_mask(), dtype=bool)
        comps = _components_from_mask(A)
        main = comps[0] if comps else tuple()
        P = env.sheep_pos.astype(np.float64)
        main_center = np.mean(P[list(main)], axis=0) if main else env.flock_center().astype(np.float64)
        center, axis, lateral = _frame_basis_np(P, env.goal.astype(np.float64))
        tasks: List[PressureTask] = []
        for rank, comp in enumerate(comps[1:], start=1):
            nc = len(comp)
            k = min(nc, self.max_pressure_slots_per_component,
                    max(1, int(math.ceil(nc / float(self.sheep_per_pressure_slot)))))
            reps = self._select_representatives(P, comp, main_center, k)
            cc = np.mean(P[list(comp)], axis=0)
            toward = main_center - cc
            nt = float(np.linalg.norm(toward))
            toward = (-axis if nt < 1e-9 else toward / nt)
            side = np.asarray([-toward[1], toward[0]], dtype=np.float64)
            radius = max(1e-6, max(float(np.linalg.norm(P[q] - cc)) for q in comp))
            reps = sorted(reps, key=lambda q: float(np.dot(P[q] - cc, side)))
            for r, q in enumerate(reps):
                lat = float(np.dot(P[q] - cc, side) / radius)
                tasks.append(PressureTask(
                    TASK_COLLECT, tuple(comp), rank, int(q), int(r), int(k),
                    float(np.clip(lat, -1.0, 1.0)), 0,
                ))
        if len(comps) <= 1 and env.num_sheep > 1:
            radial_only = np.where(env.outlier_mask())[0].tolist()
            for rr, q in enumerate(radial_only, start=1):
                tasks.append(PressureTask(TASK_COLLECT, (int(q),), rr, int(q), 0, 1, 0.0, 0))
        return HCCGStructure(
            env.num_sheep, env.num_dogs, tuple(comps), tuple(main), A, tuple(tasks),
            center.astype(np.float32), axis.astype(np.float32), lateral.astype(np.float32),
        )


class HCCGPhaseGate:
    """Component-native RECOVER/DRIVE hysteresis."""
    def __init__(self, task_builder: HCCGTaskBuilder,
                 recover_trigger_lcc: float = 0.86,
                 recover_exit_lcc: float = 0.90,
                 recover_exit_hold: int = 4):
        self.task_builder = task_builder
        self.recover_trigger_lcc = float(recover_trigger_lcc)
        self.recover_exit_lcc = float(recover_exit_lcc)
        self.recover_exit_hold = max(1, int(recover_exit_hold))
        self.reset()

    def reset(self):
        self.phase = "DRIVE"
        self.connected_hold = 0
        self.last_structure: Optional[HCCGStructure] = None

    def force_recover(self):
        self.phase = "RECOVER"; self.connected_hold = 0

    def prepare(self, env) -> DecisionContext:
        prev = self.phase
        st = self.task_builder.build_from_env(env)
        has_tasks = len(st.collect_tasks) > 0
        lcc = st.lcc_fraction
        if self.phase == "DRIVE":
            if has_tasks or lcc < self.recover_trigger_lcc:
                self.phase = "RECOVER"; self.connected_hold = 0
        else:
            if (not has_tasks) and lcc >= self.recover_exit_lcc:
                self.connected_hold += 1
            else:
                self.connected_hold = 0
            if self.connected_hold >= self.recover_exit_hold:
                self.phase = "DRIVE"; self.connected_hold = 0
        self.last_structure = st
        # candidates are only diagnostics / compatibility with existing session.
        cands = np.asarray([t.target for t in st.collect_tasks], dtype=np.int64)
        return DecisionContext(prev, self.phase, cands, float(lcc), bool(prev != self.phase))

    def collect_candidates(self, env) -> np.ndarray:
        st = self.task_builder.build_from_env(env)
        return np.asarray([t.target for t in st.collect_tasks], dtype=np.int64)

    def graph_lcc_fraction(self, env) -> float:
        return self.task_builder.build_from_env(env).lcc_fraction


class HCCGActionSpace:
    """Structured component coverage + learnable intra-component pressure count.

    Hard invariant:
      * if there are >= M detached components, all M dogs Collect distinct components;
      * if detached components < M, every detached component receives at least one
        Collect dog before support/extra pressure is considered.

    Learned choice:
      * for a large component exposing multiple pressure slots, the actor may use
        one, two, ... extra Collect dogs or leave the remaining dogs as support.

    This preserves the validated inter-component coverage semantics while giving
    the graph a genuinely richer coordination decision inside each component.
    """
    def __init__(self, task_builder: HCCGTaskBuilder):
        self.task_builder = task_builder

    @staticmethod
    def support_pool(M: int) -> List[PressureTask]:
        return [PressureTask(TASK_REAR), PressureTask(TASK_LEFT), PressureTask(TASK_RIGHT)] + [PressureTask(TASK_RESERVE,reserve_rank=i) for i in range(max(0,int(M)))]

    @staticmethod
    def _component_id(t: PressureTask) -> Tuple[int,...]:
        return tuple(t.component_members) if t.task_type==TASK_COLLECT else tuple()

    @classmethod
    def is_prefix_extendable(
        cls,
        tasks: Sequence[PressureTask],
        structure: HCCGStructure,
        phase_id: int,
    ) -> bool:
        """Return whether a partial assignment can still be completed feasibly.

        This is the prefix analogue of :meth:`is_valid`. It is intended for
        beam search: prefixes that already violate a hard invariant, or that
        no longer have enough remaining shepherd slots to satisfy one, are
        removed before score-based truncation.

        The feasible *final* assignment set is unchanged.
        """
        M = int(structure.active_dogs)
        k = len(tasks)
        if k > M:
            return False

        # Concrete task nodes are single-assignment, including indexed Reserve.
        keys = [t.key() for t in tasks]
        if len(set(keys)) != len(keys):
            return False

        collect = [t for t in tasks if t.task_type == TASK_COLLECT]
        rear = sum(t.task_type == TASK_REAR for t in tasks)
        left = sum(t.task_type == TASK_LEFT for t in tasks)
        right = sum(t.task_type == TASK_RIGHT for t in tasks)
        if rear > 1 or left > 1 or right > 1:
            return False

        remaining = M - k

        # Drive (or Recover with no collection demand): no Collect is allowed
        # and the completed support formation must contain exactly one Rear.
        if int(phase_id) == PHASE_DRIVE or len(structure.collect_tasks) == 0:
            if collect:
                return False
            if k == M:
                return rear == 1
            # If Rear has not yet been selected, at least one slot must remain.
            return rear == 1 or remaining >= 1

        comps = {tuple(t.component_members) for t in structure.collect_tasks}
        covered = {tuple(t.component_members) for t in collect}
        nc = len(comps)

        # Limited team capacity: every shepherd must Collect a distinct
        # component. Any support task or repeated component is irrecoverable.
        if nc >= M:
            if len(collect) != k:
                return False
            if len(covered) != len(collect):
                return False
            if k == M:
                return cls.is_valid(tasks, structure, phase_id)
            return True

        # Enough shepherds to cover all detached components. The remaining
        # slots must still be sufficient to cover every currently missing
        # component. Extra Collect pressure on an already covered component is
        # allowed, as in is_valid().
        missing = comps.difference(covered)
        if len(missing) > remaining:
            return False

        support_count = k - len(collect)

        # Once any support task has been selected, the final support subset
        # must contain Rear. If Rear is still absent, reserve one remaining
        # slot for it in addition to all missing component coverage.
        if support_count > 0 and rear == 0:
            if remaining < len(missing) + 1:
                return False

        if k == M:
            return cls.is_valid(tasks, structure, phase_id)
        return True

    @classmethod
    def is_valid(cls, tasks: Sequence[PressureTask], structure: HCCGStructure, phase_id: int) -> bool:
        M=int(structure.active_dogs)
        if len(tasks)!=M: return False
        # Every concrete task node is single-assignment. Reserve nodes are unique by rank.
        keys=[t.key() for t in tasks]
        if len(set(keys))!=len(keys): return False
        collect=[t for t in tasks if t.task_type==TASK_COLLECT]
        rear=sum(t.task_type==TASK_REAR for t in tasks)
        left=sum(t.task_type==TASK_LEFT for t in tasks); right=sum(t.task_type==TASK_RIGHT for t in tasks)
        if left>1 or right>1 or rear>1: return False
        if int(phase_id)==PHASE_DRIVE or len(structure.collect_tasks)==0:
            return len(collect)==0 and rear==1
        comps={tuple(t.component_members) for t in structure.collect_tasks}
        covered={tuple(t.component_members) for t in collect}
        nc=len(comps)
        if nc>=M:
            # Limited shepherd capacity: every dog services a different component.
            return len(collect)==M and len(covered)==M
        # Enough dogs to cover all detached components. At least one pressure task
        # per component is mandatory; additional dogs may add pressure or support.
        if not comps.issubset(covered): return False
        if len(collect)<1 or len(collect)>min(M,len(structure.collect_tasks)): return False
        support=M-len(collect)
        if support==0: return rear==0 and left==0 and right==0
        # Any support formation keeps exactly one Rear. Other support dogs can be
        # flank or Reserve; Reserve lets additional shepherds remain non-productive.
        return rear==1

    def enumerate_exact(self, structure: HCCGStructure, phase_id: int) -> List[HCCGAssignmentSpec]:
        M=int(structure.active_dogs)
        C=list(structure.collect_tasks) if int(phase_id)==PHASE_RECOVER else []
        pool=C+self.support_pool(M)
        # Exact enumeration is intended for the 2-dog training distribution and
        # small diagnostics. Larger zero-shot teams use graph-aware beam search.
        if M>3 and len(pool)>10:
            raise RuntimeError('exact HCCG enumeration intentionally limited; use coordination.ranking.inference for fixed native beam search')
        out=[]; seen=set()
        for p in permutations(pool,M):
            if not self.is_valid(p,structure,phase_id): continue
            k=tuple(t.key() for t in p)
            if k in seen: continue
            seen.add(k); out.append(HCCGAssignmentSpec(tuple(p)))
        if not out: raise RuntimeError('no feasible HCCG assignment')
        return out


class SumSqrtMessage(nn.Module):
    """Relation message with sum/sqrt(degree) aggregation.

    Unlike pure mean aggregation, this retains a cardinality signal while still
    keeping scale growth sublinear for zero-shot larger neighborhoods.
    """
    def __init__(self, hidden: int, edge_dim: int):
        super().__init__()
        h = int(hidden)
        self.msg = nn.Sequential(nn.Linear(h + int(edge_dim), h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.upd = nn.Sequential(nn.Linear(2 * h + 1, h), nn.Tanh(), nn.Linear(h, h))
        self.norm = nn.LayerNorm(h)
        self.apply(_orthogonal_init)

    def forward(self, h: torch.Tensor, adj: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        n, hd = h.shape
        if n == 0:
            return h
        src = h.unsqueeze(0).expand(n, n, hd)
        msg = self.msg(torch.cat([src, edge], dim=-1).reshape(n*n, -1)).reshape(n, n, hd)
        w = adj.to(dtype=h.dtype).unsqueeze(-1)
        deg = w.sum(1)
        agg = (msg * w).sum(1) / torch.sqrt(deg.clamp_min(1.0))
        logdeg = torch.log1p(deg.squeeze(-1)).unsqueeze(-1)
        delta = self.upd(torch.cat([h, agg, logdeg], dim=-1))
        return self.norm(h + delta)


class HCCGPolicy(nn.Module):
    """Graph-native actor with component supernodes and coordination factors."""
    def __init__(
        self,
        max_sheep: int,
        max_dogs: int,
        *,
        hidden: int = 128,
        world_size: float = 800.0,
        social_radius: float = 42.0,
        sheep_max_neighbors: int = 7,
        sheep_per_pressure_slot: int = 4,
        max_pressure_slots_per_component: int = 3,
        message_layers: int = 2,
        use_component_messages: bool = True,
        pairwise_weight: float = 0.50,
        admissibility_score_weight: float = 0.10,
    ):
        super().__init__()
        self.max_sheep = int(max_sheep); self.max_dogs = int(max_dogs)
        self.hidden = int(hidden); self.world_size = float(world_size)
        self.social_radius = float(social_radius); self.radius_norm = self.social_radius / max(self.world_size, 1e-8)
        self.sheep_max_neighbors = int(sheep_max_neighbors)
        self.message_layers = int(message_layers)
        self.use_component_messages = bool(use_component_messages)
        self.pairwise_weight = float(pairwise_weight)
        self.admissibility_score_weight = float(admissibility_score_weight)
        self.task_builder = HCCGTaskBuilder(
            world_size=world_size, social_radius=social_radius,
            sheep_max_neighbors=sheep_max_neighbors,
            sheep_per_pressure_slot=sheep_per_pressure_slot,
            max_pressure_slots_per_component=max_pressure_slots_per_component,
        )
        self.space = HCCGActionSpace(self.task_builder)
        h = self.hidden
        self.sheep_phi = nn.Sequential(nn.Linear(5, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.dog_phi = nn.Sequential(nn.Linear(4, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.ss_layers = nn.ModuleList([SumSqrtMessage(h, 3) for _ in range(max(1, self.message_layers))])
        self.comp_init = nn.Sequential(nn.Linear(2*h + COMP_RAW_DIM, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.comp_layers = nn.ModuleList([SumSqrtMessage(h, COMP_EDGE_DIM) for _ in range(max(1, self.message_layers))])
        self.dog_comp_msg = nn.Sequential(nn.Linear(h + 3, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.dog_update = nn.Sequential(nn.Linear(2*h + 1, h), nn.Tanh(), nn.Linear(h, h))
        self.dog_norm = nn.LayerNorm(h)
        self.context_net = nn.Sequential(nn.Linear(5*h + 8, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.task_net = nn.Sequential(nn.Linear(h + TASK_RAW_DIM, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        unary_in = 3*h + DOG_TASK_GEOM_DIM
        self.unary_net = nn.Sequential(nn.Linear(unary_in, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))
        self.admissibility_net = nn.Sequential(nn.Linear(unary_in, h), nn.Tanh(), nn.Linear(h, 1))
        pair_in = 4*h + PAIRWISE_GEOM_DIM
        self.pair_net = nn.Sequential(nn.Linear(pair_in, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))
        self.apply(_orthogonal_init)
        _orthogonal_init(self.unary_net[-1], gain=0.01)
        _orthogonal_init(self.pair_net[-1], gain=0.01)
        # Starts as 0.5 competence for every edge -> constant log term across
        # assignments, so it cannot perturb the initial policy preference.
        nn.init.zeros_(self.admissibility_net[-1].weight)
        nn.init.zeros_(self.admissibility_net[-1].bias)

    @staticmethod
    def _edge_triplet_t(receiver: torch.Tensor, sender: torch.Tensor, axis: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
        v = sender - receiver
        return torch.stack([(v*axis).sum(-1), (v*lat).sum(-1), torch.linalg.vector_norm(v, dim=-1)], dim=-1)

    def _active_tensors(self, g: torch.Tensor):
        sheep, sm, dogs, dm, goal, extras = _parse_global_torch(g, self.max_sheep, self.max_dogs)
        n = int((sm > 0.5).sum().item()); m = int((dm > 0.5).sum().item())
        sp = sheep[:n, :2]; dp = dogs[:m, :2]
        center = sp.mean(0) if n else torch.zeros(2, dtype=g.dtype, device=g.device)
        axis = goal - center; axis = axis / torch.linalg.vector_norm(axis).clamp_min(1e-6)
        lat = torch.stack([-axis[1], axis[0]])
        return sheep, sm, dogs, dm, goal, extras, n, m, sp, dp, center, axis, lat

    def _graph_encode(self, g: torch.Tensor, structure: HCCGStructure):
        sheep, sm, dogs, dm, goal, extras, n, m, sp, dp, center, axis, lat = self._active_tensors(g)
        # Candidate flag is component pressure-task participation, not a legacy
        # representative sheep ID mask.
        cand = torch.zeros(n, dtype=g.dtype, device=g.device)
        for t in structure.collect_tasks:
            for q in t.component_members:
                if 0 <= int(q) < n:
                    cand[int(q)] = 1.0
        sh = self.sheep_phi(torch.cat([sheep[:n], cand.unsqueeze(-1)], dim=-1))
        dh = self.dog_phi(dogs[:m])

        # Exact social adjacency from the structural decomposition.
        A = torch.as_tensor(structure.social_graph, dtype=torch.bool, device=g.device)
        ss_edge = self._edge_triplet_t(sp[:, None, :], sp[None, :, :], axis, lat)
        if self.use_component_messages:
            for layer in self.ss_layers:
                sh = layer(sh, A, ss_edge)

        comp_h: List[torch.Tensor] = []
        comp_centers: List[torch.Tensor] = []
        comp_sizes: List[int] = []
        main_c = sp[list(structure.main_component)].mean(0) if structure.main_component else center
        for rank, comp in enumerate(structure.components):
            idx = torch.as_tensor(list(comp), dtype=torch.long, device=g.device)
            hs = sh[idx]
            mean_h = hs.mean(0)
            sum_h = hs.sum(0) / math.sqrt(max(1, len(comp)))
            cp = sp[idx].mean(0); cv = sheep[idx, 2:4].mean(0)
            rr = torch.linalg.vector_norm(sp[idx] - cp.unsqueeze(0), dim=-1)
            rel_f = self._edge_triplet_t(center, cp, axis, lat)
            rel_m = self._edge_triplet_t(main_c, cp, axis, lat)
            raw = torch.cat([
                torch.tensor([len(comp)/max(1,n), math.log1p(len(comp))/math.log1p(max(2,n)), 1.0 if rank==0 else 0.0], dtype=g.dtype, device=g.device),
                rel_f, rel_m,
                cv,
                torch.stack([rr.mean() if len(rr) else torch.tensor(0.,device=g.device), rr.max() if len(rr) else torch.tensor(0.,device=g.device)]),
            ], dim=0)
            # raw length 3+3+3+2+2 = 13; compress deterministically to 12 by
            # dropping duplicate main-component distance radius when rank=0.
            raw = raw[:COMP_RAW_DIM]
            comp_h.append(self.comp_init(torch.cat([mean_h, sum_h, raw], dim=0)))
            comp_centers.append(cp); comp_sizes.append(len(comp))
        if comp_h:
            ch = torch.stack(comp_h, dim=0); cp = torch.stack(comp_centers, dim=0)
            C = len(comp_h)
            adj = ~torch.eye(C, dtype=torch.bool, device=g.device)
            rel = self._edge_triplet_t(cp[:,None,:], cp[None,:,:], axis, lat)
            size = torch.as_tensor(comp_sizes, dtype=g.dtype, device=g.device)
            sr = (size[None,:] / size[:,None].clamp_min(1.0)).unsqueeze(-1)
            same_main = torch.zeros((C,C,1), dtype=g.dtype, device=g.device)
            same_main[:,0,0] = 1.0
            ce = torch.cat([rel, torch.log1p(sr), same_main], dim=-1)
            if self.use_component_messages:
                for layer in self.comp_layers:
                    ch = layer(ch, adj, ce)
        else:
            ch = torch.zeros((0,self.hidden), dtype=g.dtype, device=g.device)
            cp = torch.zeros((0,2), dtype=g.dtype, device=g.device)

        # Every dog can read every component supernode.  This is where a dog
        # learns which *component* it should service, not just which sheep ID.
        if self.use_component_messages and m and len(ch):
            d2c = self._edge_triplet_t(dp[:,None,:], cp[None,:,:], axis, lat)  # dog x comp
            src = ch.unsqueeze(0).expand(m, len(ch), self.hidden)
            msg = self.dog_comp_msg(torch.cat([src, d2c], dim=-1).reshape(m*len(ch), -1)).reshape(m, len(ch), self.hidden)
            agg = msg.sum(1) / math.sqrt(max(1, len(ch)))
            logc = torch.full((m,1), math.log1p(len(ch)), dtype=g.dtype, device=g.device)
            dh = self.dog_norm(dh + self.dog_update(torch.cat([dh, agg, logc], dim=-1)))

        sheep_mean = sh.mean(0) if n else torch.zeros(self.hidden, dtype=g.dtype, device=g.device)
        sheep_sum = sh.sum(0)/math.sqrt(max(1,n)) if n else torch.zeros(self.hidden, dtype=g.dtype, device=g.device)
        dog_mean = dh.mean(0) if m else torch.zeros(self.hidden, dtype=g.dtype, device=g.device)
        comp_mean = ch.mean(0) if len(ch) else torch.zeros(self.hidden, dtype=g.dtype, device=g.device)
        comp_sum = ch.sum(0)/math.sqrt(max(1,len(ch))) if len(ch) else torch.zeros(self.hidden, dtype=g.dtype, device=g.device)
        phase_stub = torch.tensor([len(structure.components)/max(1,n), math.log1p(m)/math.log(7.0)], dtype=g.dtype, device=g.device)
        ctx = self.context_net(torch.cat([sheep_mean, sheep_sum, dog_mean, comp_mean, comp_sum, goal, extras, phase_stub], dim=0))
        return ctx, sh, dh, ch, cp, center, axis, lat, sheep, dogs, goal, extras

    def _component_index(self, structure: HCCGStructure, task: PressureTask) -> int:
        if task.task_type != TASK_COLLECT:
            return -1
        for i, c in enumerate(structure.components):
            if tuple(c) == tuple(task.component_members):
                return i
        return -1

    def _task_position(self, task: PressureTask, structure: HCCGStructure,
                       sheep: torch.Tensor, center: torch.Tensor, axis: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
        if task.task_type == TASK_COLLECT and 0 <= task.target < structure.active_sheep:
            return sheep[int(task.target), :2]
        if task.task_type == TASK_REAR:
            return center - 0.14 * axis
        if task.task_type == TASK_LEFT:
            return center - 0.07 * axis + 0.14 * lat
        if task.task_type == TASK_RIGHT:
            return center - 0.07 * axis - 0.14 * lat
        side = 1.0 if int(task.reserve_rank) % 2 == 0 else -1.0
        level = 0.18 + 0.03 * (int(task.reserve_rank)//2)
        return center - level * axis + side * 0.20 * lat

    def _task_raw(self, task: PressureTask, structure: HCCGStructure,
                  sheep: torch.Tensor, center: torch.Tensor, axis: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
        z = torch.zeros(TASK_RAW_DIM, dtype=sheep.dtype, device=sheep.device)
        z[int(task.task_type)] = 1.0
        tp = self._task_position(task, structure, sheep, center, axis, lat)
        z[5:8] = self._edge_triplet_t(center, tp, axis, lat)
        if task.task_type == TASK_COLLECT:
            ci = self._component_index(structure, task)
            comp = task.component_members
            cp = sheep[torch.as_tensor(list(comp), dtype=torch.long, device=sheep.device), :2].mean(0)
            main = structure.main_component
            mp = sheep[torch.as_tensor(list(main), dtype=torch.long, device=sheep.device), :2].mean(0) if main else center
            z[8:11] = self._edge_triplet_t(mp, cp, axis, lat)
            z[11] = float(len(comp)/max(1,structure.active_sheep))
            z[12] = float(task.pressure_rank/max(1,task.pressure_count-1)) if task.pressure_count>1 else 0.5
            z[13] = float(task.pressure_lateral)
            z[14] = float(task.pressure_count/max(1,self.task_builder.max_pressure_slots_per_component))
            z[15] = float(ci/max(1,len(structure.components)-1)) if ci>=0 else 0.0
        else:
            z[11] = 0.0; z[12] = 0.0; z[13] = float(task.reserve_rank)
        return z













class CardinalityValue(nn.Module):
    """Set critic whose parameter shapes do not depend on entity count."""
    def __init__(self, max_sheep: int, max_dogs: int, hidden: int=128):
        super().__init__(); self.max_sheep=int(max_sheep); self.max_dogs=int(max_dogs); h=int(hidden)
        self.sphi=nn.Sequential(nn.Linear(4,h),nn.Tanh(),nn.Linear(h,h),nn.Tanh())
        self.dphi=nn.Sequential(nn.Linear(4,h),nn.Tanh(),nn.Linear(h,h),nn.Tanh())
        self.net=nn.Sequential(nn.Linear(4*h+6,h),nn.Tanh(),nn.Linear(h,h),nn.Tanh(),nn.Linear(h,1))
        self.apply(_orthogonal_init); _orthogonal_init(self.net[-1],gain=1.0)
    def forward(self,g:torch.Tensor)->torch.Tensor:
        if g.ndim==1: g=g.unsqueeze(0)
        vals=[]
        for x in g:
            s,sm,d,dm,goal,extras=_parse_global_torch(x,self.max_sheep,self.max_dogs)
            n=(sm>0.5); m=(dm>0.5)
            hdim=self.sphi[-2].out_features
            sh=self.sphi(s[n]) if bool(n.any()) else torch.zeros((0,hdim),device=x.device)
            dh=self.dphi(d[m]) if bool(m.any()) else torch.zeros((0,hdim),device=x.device)
            smn=sh.mean(0) if len(sh) else torch.zeros(hdim,device=x.device)
            ssu=sh.sum(0)/math.sqrt(max(1,len(sh))) if len(sh) else torch.zeros(hdim,device=x.device)
            dmn=dh.mean(0) if len(dh) else torch.zeros(hdim,device=x.device)
            dsu=dh.sum(0)/math.sqrt(max(1,len(dh))) if len(dh) else torch.zeros(hdim,device=x.device)
            vals.append(self.net(torch.cat([smn,ssu,dmn,dsu,goal,extras],dim=0)).squeeze(-1))
        return torch.stack(vals)


class ComponentAwareRoleWaypointDecoder(RoleWaypointDecoder):
    """Same BASC interface, but Collect P_task is the main-component centroid.

    This prevents a large detached component from becoming its own recovery set
    when several pressure slots target different boundary sheep in that component.
    """
    def __init__(self,cfg:Optional[PlannerConfig]=None):
        super().__init__(cfg); self.assignment:Optional[HCCGAssignmentSpec]=None
    def set_assignment(self,spec:HCCGAssignmentSpec): self.assignment=spec
    def decode_with_task_targets(self,env,action:HighLevelAction):
        if self.assignment is None:
            return super().decode_with_task_targets(env,action)
        c=env.flock_center().astype(np.float64); goal=env.goal.astype(np.float64); g=unit(goal-c)
        if np.linalg.norm(g)<1e-9: g=np.asarray([0.0,-1.0],dtype=np.float64)
        roles=np.asarray(action.roles,dtype=np.int64); collecting=bool(np.any(roles==int(Role.COLLECT)))
        lookahead=self.cfg.collect_lookahead if collecting else self.cfg.drive_lookahead
        flock_pb=np.clip(c+float(lookahead)*g,15.0,env.cfg.world_size-15.0).astype(np.float32)
        task_targets=np.repeat(flock_pb[None,:],env.num_dogs,axis=0).astype(np.float32)
        anchors=np.zeros((env.num_dogs,2),dtype=np.float32)
        comps=_components_from_mask(env.social_graph_mask()); main=comps[0] if comps else tuple(range(env.num_sheep))
        main_set=set(main)
        pseudo_exclude=set()
        for tt in self.assignment.tasks_by_dog:
            if tt.task_type==TASK_COLLECT and set(tt.component_members).issubset(main_set) and len(tt.component_members)<len(main_set):
                pseudo_exclude.update(tt.component_members)
        recovery_ids=[i for i in main if i not in pseudo_exclude]
        main_center=env.sheep_pos[recovery_ids].astype(np.float64).mean(0) if recovery_ids else (env.sheep_pos[list(main)].astype(np.float64).mean(0) if main else c)
        for j in range(env.num_dogs):
            task=self.assignment.tasks_by_dog[j] if j<len(self.assignment.tasks_by_dog) else PressureTask(TASK_RESERVE)
            residual=np.asarray(action.offsets[j],dtype=np.float64)
            role=int(roles[j])
            if role==int(Role.COLLECT):
                q=int(np.clip(action.targets[j],0,env.num_sheep-1))
                recovery=np.clip(main_center,10.0,env.cfg.world_size-10.0).astype(np.float32)
                task_targets[j]=recovery
                anchors[j]=boundary_relative_anchor(env,"collect",self.cfg,flock_pb=recovery,target_index=q,residual=residual)
            elif role==int(Role.REAR_DRIVE):
                anchors[j]=boundary_relative_anchor(env,"rear",self.cfg,flock_pb=flock_pb,residual=residual)
            elif role==int(Role.LEFT_FLANK):
                anchors[j]=boundary_relative_anchor(env,"left",self.cfg,flock_pb=flock_pb,residual=residual)
            elif role==int(Role.RIGHT_FLANK):
                anchors[j]=boundary_relative_anchor(env,"right",self.cfg,flock_pb=flock_pb,residual=residual)
            else: raise ValueError(role)
        return flock_pb,task_targets,anchors,("CollectingComponent" if collecting else "Driving")


@dataclass
class HCCGExecutionAssignment:
    spec:HCCGAssignmentSpec
    action:HighLevelAction
    action_gains:np.ndarray


class HCCGCommittedCoordinator(FrozenSuccessorCoordinator):
    """Component-stable semi-MDP execution with logical Reserve support."""
    def __init__(self,executor,planner_cfg:PlannerConfig,task_builder:HCCGTaskBuilder,*,gamma:float,max_option_high_steps:int=64):
        self.task_builder=task_builder
        gate=HCCGPhaseGate(task_builder)
        super().__init__(executor,planner_cfg,gamma=gamma,phase_gate=gate,acquisition_cfg=AcquisitionConfig(),drive_cap_to_goal=False)
        self.max_option_high_steps=int(max_option_high_steps)
        self.component_decoder=ComponentAwareRoleWaypointDecoder(planner_cfg)
        self.executor.decoder=self.component_decoder

    def _execution(self,env,ctx:DecisionContext,spec:HCCGAssignmentSpec)->HCCGExecutionAssignment:
        M=env.num_dogs; roles=np.zeros(M,dtype=np.int64); targets=np.zeros(M,dtype=np.int64); offsets=np.zeros((M,2),dtype=np.float32); gains=np.ones(M,dtype=np.float32)
        recovering=ctx.phase=="RECOVER" and spec.collect_count>0
        for j,t in enumerate(spec.tasks_by_dog):
            if t.task_type==TASK_COLLECT:
                roles[j]=int(Role.COLLECT); targets[j]=int(t.target); offsets[j]=np.asarray([0.0,0.65*float(t.pressure_lateral)],dtype=np.float32); gains[j]=1.0
            elif t.task_type==TASK_REAR:
                roles[j]=int(Role.REAR_DRIVE); gains[j]=0.38 if recovering else (0.55 if ctx.phase=="RECOVER" else 1.0)
            elif t.task_type==TASK_LEFT:
                roles[j]=int(Role.LEFT_FLANK); gains[j]=0.20 if recovering else (0.25 if ctx.phase=="RECOVER" else 0.55)
            elif t.task_type==TASK_RIGHT:
                roles[j]=int(Role.RIGHT_FLANK); gains[j]=0.20 if recovering else (0.25 if ctx.phase=="RECOVER" else 0.55)
            elif t.task_type==TASK_RESERVE:
                # Reserve is deliberately non-productive: map through an existing
                # flank decoder to preserve BASC dimensions, but use a far/rear
                # residual and very low gain.  Extra dogs therefore do not have
                # to add pressure merely because they exist.
                side=+1.0 if (int(t.reserve_rank)%2==0) else -1.0
                roles[j]=int(Role.LEFT_FLANK if side>0 else Role.RIGHT_FLANK)
                offsets[j]=np.asarray([-1.0,side],dtype=np.float32); gains[j]=0.05
            else: raise ValueError(t.task_type)
        act=HighLevelAction(roles,targets,offsets); self.component_decoder.set_assignment(spec)
        return HCCGExecutionAssignment(spec,act,gains)

    def _step_custom(self,env,ctx:DecisionContext,exe:HCCGExecutionAssignment):
        action=exe.action; gains=exe.action_gains
        total=0.; raw_total=0.; discount=1.; duration=0; last_info={}; done=False
        if ctx.previous_phase=="RECOVER" and ctx.phase=="DRIVE":
            pb=compute_drive_pb(env,self.pc,cap_to_goal=self.drive_cap_to_goal)
            session=AcquisitionSession(self.pc,action,pb,cfg=self.acq_cfg)
            self.executor.drive_segments.interrupt("task_enter_acquire"); self.acquire_started_total+=1
            max_chunks=int(np.ceil(self.acq_cfg.timeout_primitives/max(1,self.pc.option_horizon)))+4
            for _ in range(max_chunks):
                r,done,info=session.execute_high_level(env,option_horizon=self.pc.option_horizon,shield=self.executor.shield if self.executor.use_safety_shield else None)
                total+=discount*float(r); discount*=self.gamma; raw_total+=float(r); duration+=1; last_info=dict(info)
                if done or session.status!=AcquisitionSession.STATUS_ACTIVE: break
            if session.status==AcquisitionSession.STATUS_READY and not done:
                self.executor.prime_drive_segment(env,session.pb.copy()); self.acquire_completed_total+=1
                r,done,info=self.executor.execute_role(env,action,action_gains=gains,coordination_phase="DRIVE")
                total+=discount*float(r); raw_total+=float(r); duration+=1; last_info=dict(info)
            elif session.status==AcquisitionSession.STATUS_ABORT_TO_RECOVER:
                self.acquire_aborted_total+=1; self.phase_gate.force_recover()
            elif session.status==AcquisitionSession.STATUS_TIMEOUT:
                self.acquire_timeout_total+=1; self.phase_gate.force_recover()
        else:
            r,done,info=self.executor.execute_role(env,action,action_gains=gains,coordination_phase=ctx.phase)
            total+=float(r); raw_total+=float(r); duration=1; last_info=dict(info)
        return float(total),bool(done),last_info,int(duration),float(raw_total)

    def step_committed(self,env,ctx:DecisionContext,spec:HCCGAssignmentSpec):
        start_phase=str(ctx.phase); start_partition=self.task_builder.build_from_env(env).partition_signature
        start_seg=int(self.executor.drive_segments.completed_count); start_abort=int(self.acquire_aborted_total+self.acquire_timeout_total)
        exe=self._execution(env,ctx,spec)
        total=0.; raw=0.; duration=0; disc=1.; done=False; info={}; pending=None; chunks=0; collision_any=0
        cur=ctx
        while not done and chunks<self.max_option_high_steps and env.primitive_step<env.cfg.max_steps:
            r,done,inf,dur,rraw=self._step_custom(env,cur,exe)
            total+=disc*r; disc*=self.gamma**max(1,dur); raw+=rraw; duration+=dur; chunks+=max(1,dur); info=dict(inf)
            collision_any=max(collision_any,int(float(inf.get("dog_collision_count",0.0))>0),int(float(inf.get("collision_primitive_fraction",0.0))>0))
            if done: break
            if int(self.acquire_aborted_total+self.acquire_timeout_total)>start_abort and cur.phase=="DRIVE":
                st=self.task_builder.build_from_env(env); pending=DecisionContext("DRIVE","RECOVER",np.asarray([t.target for t in st.collect_tasks],dtype=np.int64),st.lcc_fraction,True); break
            nxt=self.phase_gate.prepare(env)
            if nxt.transitioned: pending=nxt; break
            if start_phase=="RECOVER":
                if self.task_builder.build_from_env(env).partition_signature!=start_partition:
                    pending=nxt; break
            else:
                if int(self.executor.drive_segments.completed_count)>start_seg:
                    pending=nxt; break
            cur=nxt
        info.update({
            "task_assignment_signature":spec.signature(),"task_collect_count":float(spec.collect_count),"task_reserve_count":float(spec.reserve_count),
            "task_collision_any":float(collision_any),"task_option_duration_high":float(duration),"task_partition_changed":float(self.task_builder.build_from_env(env).partition_signature!=start_partition),
        })
        return float(total),bool(done),info,int(duration),float(raw),pending

    def collect_admissibility_labels(self,env_before_structure:HCCGStructure,spec:HCCGAssignmentSpec,env_after,collision_any:int=0):
        after=self.task_builder.build_from_env(env_after); main=set(after.main_component)
        y=np.zeros(len(spec.tasks_by_dog),dtype=np.float32); mask=np.zeros_like(y)
        for j,t in enumerate(spec.tasks_by_dog):
            if t.task_type==TASK_COLLECT:
                mask[j]=1.0
                members=set(t.component_members)
                frac=len(members & main)/max(1,len(members))
                y[j]=1.0 if frac>=0.5 else 0.0
        return y,mask


