"""Adapted edge-competition and conditional agent-DAG allocation models."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .core import (HCCGTaskBuilder, HCCGActionSpace, HCCGAssignmentSpec,
    HCCGStructure, PressureTask, TASK_COLLECT, TASK_REAR, TASK_LEFT,
    TASK_RIGHT, TASK_RESERVE, PHASE_RECOVER, PHASE_DRIVE, _parse_global_torch)
from .value import _orthogonal_init
DOG_FEAT_DIM=10
TASK_FEAT_DIM=22
EDGE_FEAT_DIM=10
GLOBAL_FEAT_DIM=12
def _mlp(inp: int, hidden: int, out: Optional[int] = None) -> nn.Sequential:
    h = int(hidden)
    o = int(out if out is not None else h)
    return nn.Sequential(
        nn.Linear(int(inp), h),
        nn.Tanh(),
        nn.Linear(h, o),
        nn.Tanh() if o == h else nn.Identity(),
    )


def _task_key(t: PressureTask) -> Tuple:
    return t.key()


def _task_order_key(t: PressureTask) -> Tuple:
    return (
        int(t.task_type),
        tuple(t.component_members),
        int(t.component_rank),
        int(t.pressure_rank),
        int(t.reserve_rank),
        int(t.target),
    )


def _safe_mean(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    if x.numel() == 0:
        shape = list(x.shape)
        shape.pop(dim)
        return torch.zeros(shape, dtype=x.dtype, device=x.device)
    return x.mean(dim=dim)


@dataclass
class EncodedAssignmentState:
    structure: HCCGStructure
    phase_id: int
    tasks: Tuple[PressureTask, ...]
    task_index: Dict[Tuple, int]
    dog_raw: torch.Tensor
    task_raw: torch.Tensor
    edge_raw: torch.Tensor
    global_raw: torch.Tensor
    dog_emb: torch.Tensor
    task_emb: torch.Tensor
    global_emb: torch.Tensor
    edge_emb: torch.Tensor
    edge_logits: torch.Tensor
    edge_admiss: torch.Tensor


class ExternalAssignmentPolicyBase(nn.Module):
    family_name = "external"

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
        outlier_radius: float = 75.0,
        admissibility_score_weight: float = 0.10,
    ):
        super().__init__()
        self.max_sheep = int(max_sheep)
        self.max_dogs = int(max_dogs)
        self.hidden = int(hidden)
        self.world_size = float(world_size)
        self.social_radius = float(social_radius)
        self.sheep_max_neighbors = int(sheep_max_neighbors)
        self.message_layers = int(message_layers)
        self.outlier_radius = float(outlier_radius)
        self.admissibility_score_weight = float(admissibility_score_weight)

        self.task_builder = HCCGTaskBuilder(
            world_size=world_size,
            social_radius=social_radius,
            sheep_max_neighbors=sheep_max_neighbors,
            sheep_per_pressure_slot=sheep_per_pressure_slot,
            max_pressure_slots_per_component=max_pressure_slots_per_component,
            outlier_radius=outlier_radius,
        )
        self.space = HCCGActionSpace(self.task_builder)

        h = self.hidden
        self.dog_encoder = _mlp(DOG_FEAT_DIM, h)
        self.task_encoder = _mlp(TASK_FEAT_DIM, h)
        self.global_encoder = _mlp(GLOBAL_FEAT_DIM, h)
        self.base_edge_encoder = _mlp(2 * h + h + EDGE_FEAT_DIM, h)
        self.apply(_orthogonal_init)

    # ------------------------------------------------------------------
    # Shared deterministic feature construction.
    # ------------------------------------------------------------------
    def _candidate_tasks(
        self,
        st: HCCGStructure,
        phase_id: int,
    ) -> Tuple[PressureTask, ...]:
        collect = list(st.collect_tasks) if int(phase_id) == PHASE_RECOVER else []
        pool = collect + self.space.support_pool(int(st.active_dogs))
        seen = set()
        out = []
        for t in pool:
            k = _task_key(t)
            if k not in seen:
                seen.add(k)
                out.append(t)
        return tuple(out)

    def _build_raw_features(
        self,
        g: torch.Tensor,
        phase_id: int,
        st: HCCGStructure,
        tasks: Sequence[PressureTask],
    ):
        sheep, sm, dogs, dm, goal, extras = _parse_global_torch(
            g, self.max_sheep, self.max_dogs
        )
        n = int(st.active_sheep)
        m = int(st.active_dogs)
        dev = g.device
        dtype = g.dtype

        sheep_pos = sheep[:n, :2]
        dog_state = dogs[:m]
        if n > 0:
            flock_center = sheep_pos.mean(0)
        else:
            flock_center = torch.zeros(2, device=dev, dtype=dtype)

        axis_np = np.asarray(st.axis, dtype=np.float32)
        lat_np = np.asarray(st.lateral, dtype=np.float32)
        center_np = np.asarray(st.center, dtype=np.float32)
        axis = torch.as_tensor(axis_np, device=dev, dtype=dtype)
        lateral = torch.as_tensor(lat_np, device=dev, dtype=dtype)
        center = torch.as_tensor(center_np, device=dev, dtype=dtype)

        # Dog features: native 4D + relative-to-flock + relative-to-goal.
        dog_feat = []
        for i in range(m):
            p = dog_state[i, :2]
            dc = p - flock_center
            dg = goal - p
            dog_feat.append(torch.cat([
                dog_state[i],
                dc,
                torch.linalg.vector_norm(dc).view(1),
                dg[:2],
                torch.linalg.vector_norm(dg[:2]).view(1),
            ], dim=0))
        dog_raw = (
            torch.stack(dog_feat, 0)
            if dog_feat else
            torch.zeros((0, DOG_FEAT_DIM), device=dev, dtype=dtype)
        )

        # Task features use the same topology objects, but do not reuse HCCG's
        # learned component/task encoders.
        task_feat = []
        anchors = []
        for t in tasks:
            typ = torch.zeros(5, device=dev, dtype=dtype)
            typ[int(t.task_type)] = 1.0

            members = [int(q) for q in t.component_members if 0 <= int(q) < n]
            if members:
                idx = torch.as_tensor(members, device=dev, dtype=torch.long)
                cp = sheep_pos.index_select(0, idx).mean(0)
                spread = torch.linalg.vector_norm(
                    sheep_pos.index_select(0, idx) - cp[None, :], dim=-1
                ).mean()
            else:
                cp = center
                spread = torch.zeros((), device=dev, dtype=dtype)

            # Representation-only support anchors.
            if int(t.task_type) == TASK_COLLECT and 0 <= int(t.target) < n:
                anchor = sheep[int(t.target), :2]
            elif int(t.task_type) == TASK_REAR:
                anchor = center - 0.08 * axis
            elif int(t.task_type) == TASK_LEFT:
                anchor = center + 0.08 * lateral
            elif int(t.task_type) == TASK_RIGHT:
                anchor = center - 0.08 * lateral
            else:
                rr = float(t.reserve_rank)
                anchor = center - 0.12 * axis + (0.03 * ((rr % 3.0) - 1.0)) * lateral

            anchors.append(anchor)

            rel_goal = goal[:2] - cp
            comp_size = torch.tensor(
                [float(len(members)) / max(1.0, float(self.max_sheep))],
                device=dev, dtype=dtype
            )
            comp_rank = torch.tensor(
                [max(-1.0, float(t.component_rank)) / max(1.0, float(self.max_sheep))],
                device=dev, dtype=dtype
            )
            pr = torch.tensor(
                [float(t.pressure_rank) / 3.0, float(t.pressure_count) / 3.0,
                 float(t.pressure_lateral)],
                device=dev, dtype=dtype
            )
            reserve = torch.tensor(
                [float(t.reserve_rank) / max(1.0, float(self.max_dogs))],
                device=dev, dtype=dtype
            )
            scalars = torch.cat([
                comp_size,
                comp_rank,
                pr,
                reserve,
                spread.view(1),
            ], dim=0)  # 7

            # 5 + 7 + 2 + 2 + 3 + 3 = 22
            feat = torch.cat([
                typ,                       # 5
                scalars,                   # 7 -> 12
                anchor,                    # 2 -> 14
                cp,                        # 2 -> 16
                rel_goal,                  # 2
                torch.linalg.vector_norm(rel_goal).view(1),  # ->19
                anchor - cp,               # 2
                torch.linalg.vector_norm(anchor - cp).view(1),  # ->22
            ], dim=0)
            task_feat.append(feat)

        task_raw = (
            torch.stack(task_feat, 0)
            if task_feat else
            torch.zeros((0, TASK_FEAT_DIM), device=dev, dtype=dtype)
        )
        anchor_tensor = (
            torch.stack(anchors, 0)
            if anchors else
            torch.zeros((0, 2), device=dev, dtype=dtype)
        )

        # Edge features: dog-task geometry.
        edge_raw = torch.zeros(
            (m, len(tasks), EDGE_FEAT_DIM), device=dev, dtype=dtype
        )
        for i in range(m):
            dp = dog_state[i, :2]
            for j, t in enumerate(tasks):
                a = anchor_tensor[j]
                da = a - dp
                dg = goal[:2] - dp
                # dx,dy,dist to task; projections in topology frame;
                # dx,dy,dist to goal; collect indicator; support indicator.
                edge_raw[i, j] = torch.cat([
                    da,
                    torch.linalg.vector_norm(da).view(1),
                    torch.stack([torch.dot(da, axis), torch.dot(da, lateral)]),
                    dg,
                    torch.linalg.vector_norm(dg).view(1),
                    torch.tensor([
                        1.0 if int(t.task_type) == TASK_COLLECT else 0.0,
                        0.0 if int(t.task_type) == TASK_COLLECT else 1.0,
                    ], device=dev, dtype=dtype),
                ], dim=0)

        lcc = float(st.lcc_fraction)
        global_raw = torch.cat([
            torch.tensor([
                float(phase_id),
                float(n) / max(1.0, float(self.max_sheep)),
                float(m) / max(1.0, float(self.max_dogs)),
                lcc,
            ], device=dev, dtype=dtype),
            goal[:2],
            flock_center,
            extras[:4],
        ], dim=0)
        assert global_raw.numel() == GLOBAL_FEAT_DIM

        return dog_raw, task_raw, edge_raw, global_raw

    def _encode_base(
        self,
        dog_raw: torch.Tensor,
        task_raw: torch.Tensor,
        edge_raw: torch.Tensor,
        global_raw: torch.Tensor,
    ):
        dog = self.dog_encoder(dog_raw)
        task = self.task_encoder(task_raw)
        glob = self.global_encoder(global_raw)
        m, t = edge_raw.shape[:2]
        if m == 0 or t == 0:
            edge = torch.zeros((m, t, self.hidden), device=global_raw.device)
        else:
            d = dog[:, None, :].expand(m, t, self.hidden)
            q = task[None, :, :].expand(m, t, self.hidden)
            c = glob[None, None, :].expand(m, t, self.hidden)
            edge = self.base_edge_encoder(
                torch.cat([d, q, c, edge_raw], dim=-1).reshape(m * t, -1)
            ).reshape(m, t, self.hidden)
        return dog, task, glob, edge


    def architecture_forward(
        self,
        dog: torch.Tensor,
        task: torch.Tensor,
        glob: torch.Tensor,
        edge: torch.Tensor,
        edge_raw: torch.Tensor,
        tasks: Sequence[PressureTask],
    ):
        raise NotImplementedError

    def score_spec(
        self,
        state: EncodedAssignmentState,
        spec: HCCGAssignmentSpec,
    ) -> torch.Tensor:
        vals = []
        for i, t in enumerate(spec.tasks_by_dog):
            j = state.task_index[_task_key(t)]
            v = state.edge_logits[i, j]
            if int(t.task_type) == TASK_COLLECT:
                v = v + self.admissibility_score_weight * F.logsigmoid(
                    state.edge_admiss[i, j]
                )
            vals.append(v)
        return torch.stack(vals).mean()

    def admissibility_for_spec(
        self,
        state: EncodedAssignmentState,
        spec: HCCGAssignmentSpec,
    ) -> torch.Tensor:
        out = torch.zeros(
            self.max_dogs,
            device=state.global_raw.device,
            dtype=state.global_raw.dtype,
        )
        for i, t in enumerate(spec.tasks_by_dog):
            if i >= self.max_dogs:
                break
            j = state.task_index[_task_key(t)]
            out[i] = state.edge_admiss[i, j]
        return out

    def score_specs(
        self,
        state: EncodedAssignmentState,
        specs: Sequence[HCCGAssignmentSpec],
    ) -> torch.Tensor:
        return torch.stack([self.score_spec(state, s) for s in specs], dim=0)





    def partial_surrogate(
        self,
        state: EncodedAssignmentState,
        partial: Sequence[PressureTask],
    ) -> float:
        vals = []
        for i, t in enumerate(partial):
            j = state.task_index[_task_key(t)]
            v = state.edge_logits[i, j]
            if int(t.task_type) == TASK_COLLECT:
                v = v + self.admissibility_score_weight * F.logsigmoid(
                    state.edge_admiss[i, j]
                )
            vals.append(float(v.detach().cpu().item()))
        return float(np.mean(vals)) if vals else 0.0



class EdgeGNNPolicy(ExternalAssignmentPolicyBase):
    family_name = "magnet_edge_gnn"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        h = self.hidden
        L = max(1, int(self.message_layers))
        self.same_dog_msg = nn.ModuleList([
            _mlp(h, h) for _ in range(L)
        ])
        self.same_task_msg = nn.ModuleList([
            _mlp(h, h) for _ in range(L)
        ])
        self.line_update = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3 * h + 2, h),
                nn.Tanh(),
                nn.Linear(h, h),
            )
            for _ in range(L)
        ])
        self.line_norm = nn.ModuleList([nn.LayerNorm(h) for _ in range(L)])
        self.edge_score = nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))
        self.edge_adm = nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))
        self.apply(_orthogonal_init)

    def architecture_forward(self, dog, task, glob, edge, edge_raw, tasks):
        h = edge
        M, T, H = h.shape
        for l in range(len(self.line_update)):
            same_dog = self.same_dog_msg[l](h.reshape(M * T, H)).reshape(M, T, H)
            dog_agg = same_dog.sum(dim=1, keepdim=True) - same_dog
            dog_deg = max(1, T - 1)
            dog_agg = dog_agg / math.sqrt(float(dog_deg))

            same_task = self.same_task_msg[l](h.reshape(M * T, H)).reshape(M, T, H)
            task_agg = same_task.sum(dim=0, keepdim=True) - same_task
            task_deg = max(1, M - 1)
            task_agg = task_agg / math.sqrt(float(task_deg))

            dd = torch.full(
                (M, T, 1), math.log1p(float(dog_deg)),
                device=h.device, dtype=h.dtype
            )
            td = torch.full(
                (M, T, 1), math.log1p(float(task_deg)),
                device=h.device, dtype=h.dtype
            )
            delta = self.line_update[l](
                torch.cat([h, dog_agg, task_agg, dd, td], dim=-1)
            )
            h = self.line_norm[l](h + delta)

        logits = self.edge_score(h).squeeze(-1)
        adm = self.edge_adm(h).squeeze(-1)
        return h, logits, adm


class AgentDAGPolicy(ExternalAssignmentPolicyBase):
    family_name = "gcs_agent_dag"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        h = self.hidden
        L = max(1, int(self.message_layers))
        self.dep_gate = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3 * h, h),
                nn.Tanh(),
                nn.Linear(h, 1),
            )
            for _ in range(L)
        ])
        self.dep_msg = nn.ModuleList([_mlp(h, h) for _ in range(L)])
        self.dep_update = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * h + 1, h),
                nn.Tanh(),
                nn.Linear(h, h),
            )
            for _ in range(L)
        ])
        self.dep_norm = nn.ModuleList([nn.LayerNorm(h) for _ in range(L)])
        self.edge_rebuild = _mlp(3 * h + EDGE_FEAT_DIM, h)
        self.score_head = nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))
        self.adm_head2 = nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))
        # Action-dependency path: a shepherd's candidate task score is
        # conditioned on the tasks already assigned to its DAG predecessors.
        self.action_msg = _mlp(2 * h, h)
        self.conditional_head = nn.Sequential(
            nn.Linear(3 * h, h),
            nn.Tanh(),
            nn.Linear(h, 1),
        )
        self.apply(_orthogonal_init)

    @staticmethod
    def _dag_order(edge_raw: torch.Tensor) -> List[int]:
        # Deterministic state-derived ordering used only to make the learned
        # dependency graph acyclic.  It does not depend on shepherd identity.
        # Edge feature 7 is shepherd->goal distance; feature 2 is
        # shepherd->task-anchor distance.  The latter is averaged over tasks.
        M, T = edge_raw.shape[:2]
        if T <= 0:
            return list(range(M))
        goal_dist = edge_raw[:, 0, 7].detach().cpu().numpy()
        task_dist = edge_raw[:, :, 2].mean(dim=1).detach().cpu().numpy()
        # continuous geometric keys make exact ties rare; index is only a
        # deterministic final tie-break and is not fed to the network.
        return sorted(
            range(M),
            key=lambda i: (
                float(goal_dist[i]),
                float(task_dist[i]),
                int(i),
            ),
        )

    def architecture_forward(self, dog, task, glob, edge, edge_raw, tasks):
        d = dog
        M = d.shape[0]
        order = self._dag_order(edge_raw)
        rank = {dog_i: k for k, dog_i in enumerate(order)}

        for l in range(len(self.dep_update)):
            new = []
            for i in range(M):
                preds = [j for j in range(M) if rank[j] < rank[i]]
                if not preds:
                    agg = torch.zeros_like(d[i])
                    deg = torch.zeros(1, device=d.device, dtype=d.dtype)
                else:
                    msgs = []
                    gates = []
                    for j in preds:
                        gate_in = torch.cat([d[j], d[i], glob], dim=-1)
                        a = torch.sigmoid(self.dep_gate[l](gate_in)).squeeze()
                        msgs.append(self.dep_msg[l](d[j]) * a)
                        gates.append(a)
                    agg = torch.stack(msgs, 0).sum(0) / math.sqrt(float(len(preds)))
                    deg = torch.stack(gates).sum().view(1)
                delta = self.dep_update[l](
                    torch.cat([d[i], agg, torch.log1p(deg)], dim=-1)
                )
                new.append(self.dep_norm[l](d[i] + delta))
            d = torch.stack(new, 0)

        M, T = edge_raw.shape[:2]
        if M == 0 or T == 0:
            e = edge
        else:
            di = d[:, None, :].expand(M, T, self.hidden)
            tj = task[None, :, :].expand(M, T, self.hidden)
            gg = glob[None, None, :].expand(M, T, self.hidden)
            rebuilt = self.edge_rebuild(
                torch.cat([di, tj, gg, edge_raw], dim=-1).reshape(M*T, -1)
            ).reshape(M, T, self.hidden)
            e = edge + rebuilt
        logits = self.score_head(e).squeeze(-1)
        adm = self.adm_head2(e).squeeze(-1)
        return e, logits, adm

    def _conditional_score_tasks(
        self,
        state: EncodedAssignmentState,
        tasks_by_dog: Sequence[PressureTask],
    ) -> torch.Tensor:
        n_assigned = len(tasks_by_dog)
        if n_assigned == 0:
            return torch.zeros(
                (), device=state.global_raw.device, dtype=state.global_raw.dtype
            )
        order = self._dag_order(state.edge_raw)
        rank = {dog_i: k for k, dog_i in enumerate(order)}
        vals = []
        for i, t in enumerate(tasks_by_dog):
            ti = state.task_index[_task_key(t)]
            preds = [
                j for j in range(n_assigned)
                if j != i and rank[j] < rank[i]
            ]
            if preds:
                msgs = []
                for j in preds:
                    tj = state.task_index[_task_key(tasks_by_dog[j])]
                    gate = torch.sigmoid(
                        self.dep_gate[-1](
                            torch.cat([
                                state.dog_emb[j],
                                state.dog_emb[i],
                                state.global_emb,
                            ], dim=-1)
                        )
                    ).squeeze()
                    msg = self.action_msg(
                        torch.cat([
                            state.dog_emb[j],
                            state.task_emb[tj],
                        ], dim=-1)
                    )
                    msgs.append(gate * msg)
                act_ctx = torch.stack(msgs, 0).sum(0) / math.sqrt(
                    float(len(preds))
                )
            else:
                act_ctx = torch.zeros_like(state.global_emb)

            cond = self.conditional_head(
                torch.cat([
                    state.edge_emb[i, ti],
                    act_ctx,
                    state.global_emb,
                ], dim=-1)
            ).squeeze()
            v = state.edge_logits[i, ti] + cond
            if int(t.task_type) == TASK_COLLECT:
                v = v + self.admissibility_score_weight * F.logsigmoid(
                    state.edge_admiss[i, ti]
                )
            vals.append(v)
        return torch.stack(vals).mean()

    def score_spec(self, state, spec):
        return self._conditional_score_tasks(state, spec.tasks_by_dog)

    def partial_surrogate(self, state, partial):
        return float(
            self._conditional_score_tasks(state, partial).detach().cpu().item()
        )


def make_external_policy(family, **kwargs):
    if family == 'magnet_edge_gnn': return EdgeGNNPolicy(**kwargs)
    if family == 'gcs_agent_dag': return AgentDAGPolicy(**kwargs)
    raise ValueError('Unsupported allocation family: '+str(family))
