"""Differentiable native external factors over authoritative task snapshots.

Used by from-scratch PPO training and frozen evaluation. Learned conditional
scores are retained. Only inference helpers are decorated with no_grad.

"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn.functional as F


def encode_external(policy,g,st,phase):
    from .baselines import EncodedAssignmentState,_task_key
    tasks=policy._candidate_tasks(st,phase)
    raw=policy._build_raw_features(g,phase,st,tasks)
    d,t,glob,e=policy._encode_base(*raw)
    e2,logits,adm=policy.architecture_forward(d,t,glob,e,raw[2],tasks)
    return EncodedAssignmentState(st,int(phase),tuple(tasks),{_task_key(t):i for i,t in enumerate(tasks)},
         *raw,d,t,glob,e2,logits,adm)


class ExternalScores:
    def __init__(self,policy,g,st,phase):
        self.policy=policy;self.state=encode_external(policy,g,st,phase)
        self.pool=self.state.tasks;self.M=st.active_dogs;self.K=len(self.pool)
        self.device=g.device;self.family=policy.family_name
        collect=torch.tensor([t.task_type==0 for t in self.pool],device=g.device)
        # Do not depend on the numeric task enum.
        from .core import TASK_COLLECT
        collect=torch.tensor([t.task_type==TASK_COLLECT for t in self.pool],device=g.device)
        self.collect=collect
        self.u=self.state.edge_logits + policy.admissibility_score_weight*F.logsigmoid(self.state.edge_admiss)*collect[None,:]
        if self.family=='gcs_agent_dag':
            s=self.state;p=self.policy;M=self.M;K=self.K
            self.order=p._dag_order(s.edge_raw);rank={i:j for j,i in enumerate(self.order)}
            self.rank=rank
            self.messages=p.action_msg(torch.cat([s.dog_emb[:,None,:].expand(M,K,-1),s.task_emb[None,:,:].expand(M,K,-1)],-1))
            gates=[]
            for j in range(M):
                row=[]
                for i in range(M):
                    row.append(torch.sigmoid(p.dep_gate[-1](torch.cat([s.dog_emb[j],s.dog_emb[i],s.global_emb],-1))).squeeze())
                gates.append(torch.stack(row))
            self.gates=torch.stack(gates)

    def _gcs(self,ix):
        s=self.state;p=self.policy;B,n=ix.shape;dev=ix.device
        dogids=torch.arange(n,device=dev)
        selected=self.messages[dogids[None,:],ix]
        contexts=[]
        for i in range(n):
            preds=[j for j in range(n) if j!=i and self.rank[j]<self.rank[i]]
            if preds:
                msgs=selected[:,preds,:]*self.gates[preds,i][None,:,None]
                ctx=msgs.sum(1)/math.sqrt(len(preds))
            else:ctx=torch.zeros((B,p.hidden),device=dev,dtype=s.global_emb.dtype)
            contexts.append(ctx)
        ctx=torch.stack(contexts,1)
        conditional=p.conditional_head(torch.cat([s.edge_emb[dogids[None,:],ix],ctx,
            s.global_emb[None,None,:].expand(B,n,-1)],-1)).squeeze(-1)
        return (self.u[dogids[None,:],ix]+conditional).mean(1)


    def score_indices(self,indices,*,partial=False):
        x=torch.as_tensor(indices,device=self.device,dtype=torch.long)
        if x.ndim!=2 or len(x)==0:raise ValueError('Empty/invalid candidates')
        ys=[]
        for ix in x.split(256):
            if self.family=='gcs_agent_dag':v=self._gcs(ix)
            else:v=self.u[torch.arange(ix.shape[1],device=ix.device)[None,:],ix].mean(1)
            ys.append(v)
        return torch.cat(ys)

    def indices(self,specs):
        return [[self.state.task_index[t.key()] for t in s.tasks_by_dog] for s in specs]

    def native_scores(self,specs):
        return self.policy.score_specs(self.state,specs)


def table_for(policy,state):
    if not hasattr(policy,'family_name'):
        raise TypeError('Native external scoring accepts only the two external coordinator families')
    g=torch.as_tensor(state.global_np,device=next(policy.parameters()).device)
    return ExternalScores(policy,g,state.structure,state.phase)


def reference_check(policy,states):
    """Compare real loaded checkpoint to native scalar formulas, no outcome selection."""
    from .core import HCCGAssignmentSpec
    checks=[]
    with torch.no_grad():
        for state in states:
            tb=table_for(policy,state)
            if state.structure.active_dogs<=3:specs=policy.space.enumerate_exact(state.structure,state.phase)
            else:
                # Deterministic legal sample, independent of scores and outcomes.
                beam=[()]
                for i in range(state.structure.active_dogs):
                    cand=[s+(k,) for s in beam for k in range(tb.K) if policy.space.is_prefix_extendable(tuple(tb.pool[q] for q in s+(k,)),state.structure,state.phase)]
                    beam=cand[:48]
                specs=[HCCGAssignmentSpec(tuple(tb.pool[k] for k in s)) for s in beam if policy.space.is_valid(tuple(tb.pool[k] for k in s),state.structure,state.phase)]
            specs=specs[:32]
            if not specs:raise RuntimeError('No reference candidates')
            indices=tb.indices(specs);fast=tb.score_indices(indices);native=tb.native_scores(specs)
            if not torch.isfinite(fast).all() or not torch.allclose(fast,native,atol=3e-5,rtol=1e-5):
                raise RuntimeError(f'Candidate scorer differs from native formula: {tb.family}; maxerror={float((fast-native).abs().max())}')
            partial_error=0.
            if isinstance(tb,ExternalScores):
                for length in range(1,state.structure.active_dogs+1):
                    pts=[s.tasks_by_dog[:length] for s in specs[:8]]
                    ix=[r[:length] for r in indices[:8]]
                    v=tb.score_indices(ix,partial=True)
                    expected=torch.tensor([policy.partial_surrogate(tb.state,s) for s in pts],device=v.device)
                    if not torch.allclose(v,expected,atol=3e-5,rtol=1e-5):raise RuntimeError('Partial-score native reference mismatch')
                    partial_error=max(partial_error,float((v-expected).abs().max()))
            checks.append({'dogs':state.structure.active_dogs,'phase':state.phase,'candidates':len(specs),
                           'max_score_error':float((fast-native).abs().max()),'partial_error':partial_error})
    return checks


@torch.no_grad()
def choose(policy,state):
    from . import core
    tb=table_for(policy,state);M=state.structure.active_dogs
    if M<=3:
        specs=state.specs;y=tb.score_indices(tb.indices(specs))
        if not torch.isfinite(y).all():raise RuntimeError('Nonfinite scores')
        # Scalar final scoring on all exact candidates avoids a batch-dependent tie decision.
        vals=tb.native_scores(specs)
        k=int(vals.argmax())
        return specs[k],{'search':'exact','candidates':len(specs),'truncated':False}
    beams=[()];expanded=0;truncated=False
    for j in range(M):
        nxt=[]
        for part in beams:
            used=set(part)
            for k in range(tb.K):
                if k in used:continue
                ix=part+(k,)
                if policy.space.is_prefix_extendable(tuple(tb.pool[q] for q in ix),state.structure,state.phase):nxt.append(ix)
        if not nxt:raise RuntimeError('No extendable beam; no fallback')
        expanded+=len(nxt)
        vals=tb.score_indices(nxt,partial=True).cpu().double().numpy()
        if not np.isfinite(vals).all():raise RuntimeError('Nonfinite beam scores')
        order=np.argsort(-vals,kind='stable');truncated|=len(nxt)>4096
        beams=[nxt[int(i)] for i in order[:4096]]
    specs=[core.HCCGAssignmentSpec(tuple(tb.pool[k] for k in part)) for part in beams
           if policy.space.is_valid(tuple(tb.pool[k] for k in part),state.structure,state.phase)]
    if not specs:raise RuntimeError('No feasible final beam')
    vals=tb.score_indices(tb.indices(specs))
    # Close final scores are reranked with the native formula, stable original order.
    best=vals.max();near=torch.nonzero(vals>=best-(3e-5+1e-5*best.abs())).flatten().cpu().tolist()
    nv=tb.native_scores([specs[i] for i in near]);pick=near[int(nv.argmax())]
    return specs[pick],{'search':'fixed4096_native_order','candidates':len(specs),'expanded':expanded,'truncated':bool(truncated)}
