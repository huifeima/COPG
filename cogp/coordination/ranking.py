"""Symmetric HCCG factors and native assignment ranking.

Unlike the earlier diagnostic wrapper, the tensor computations here retain
parameter gradients. Each state's graph is encoded once per forward call, never
cached across policy updates or physical states. A reference-equivalence test is
mandatory before server training.
"""
from __future__ import annotations
from dataclasses import dataclass
from contextlib import contextmanager
import itertools
import math
import numpy as np
import torch
import torch.nn.functional as F

@dataclass
class Prepared:
    snapshot: object
    global_np: np.ndarray
    structure: object
    phase: int
    specs: list
    candidate_hash: str
    keys: tuple

def prepare(encoded, training=False):
    from . import core
    from .snapshot import TaskSnapshot,candidate_digest,task_key
    snap=TaskSnapshot(encoded) if isinstance(encoded,str) else encoded
    g,st,phase=snap.tensors(core)
    specs=core.HCCGActionSpace(None).enumerate_exact(st,phase) if st.active_dogs<=3 else []
    result=Prepared(snap,g.numpy(),st,phase,specs,candidate_digest(specs) if specs else '',
                    tuple(task_key(x) for x in specs))
    if training and st.active_dogs>3:
        from .geometry import geometry_candidates
        from cogp.learning.parent.protocol import TRAIN_M4_CANDIDATES
        specs,info=geometry_candidates(result,width=TRAIN_M4_CANDIDATES)
        result.specs=specs;result.candidate_hash=candidate_digest(specs)
        result.keys=tuple(task_key(x) for x in specs)
        result.training_search=info
    return result

def pair_scale_factor(policy,M):
    mode=getattr(policy,'pair_scale_mode','constant')
    if mode=='constant': return 1.0
    raise ValueError('Unknown pair scale mode: '+str(mode))

class Factors:
    def __init__(self,policy,g,st,phase,core):
        self.policy,self.g,self.st,self.phase,self.core=policy,g,st,int(phase),core
        self.M=int(st.active_dogs)
        self.pool=(list(st.collect_tasks) if phase==core.PHASE_RECOVER else [])+policy.space.support_pool(self.M)
        self.index={t.key():i for i,t in enumerate(self.pool)}
        if len(self.index)!=len(self.pool):raise RuntimeError('Duplicate task identity')
        self.K=len(self.pool)
        ctx,_,dh,ch,_,center,axis,lat,sheep,dogs,_,_=policy._graph_encode(g,st)
        self.ctx,self.dh,self.ch,self.center,self.axis,self.lat,self.sheep,self.dogs=ctx,dh,ch,center,axis,lat,sheep,dogs
        raw=torch.stack([policy._task_raw(t,st,sheep,center,axis,lat) for t in self.pool])
        comp=[]
        for t in self.pool:
            ci=policy._component_index(st,t)
            comp.append(ch[ci] if 0<=ci<len(ch) else torch.zeros(policy.hidden,device=g.device))
        self.th=policy.task_net(torch.cat([torch.stack(comp),raw],dim=-1))
        self.positions=torch.stack([policy._task_position(t,st,sheep,center,axis,lat) for t in self.pool])
        dogc=policy._edge_triplet_t(center,dogs[:self.M,:2],axis,lat)
        taskc=policy._edge_triplet_t(center,self.positions,axis,lat)
        td=policy._edge_triplet_t(dogs[:self.M,None,:2],self.positions[None,:,:],axis,lat)
        extra=[]
        for k,t in enumerate(self.pool):
            extra.append(torch.stack([raw[k,11],raw[k,13],raw[k,10]]) if t.task_type==core.TASK_COLLECT
                         else torch.zeros(3,device=g.device))
        geom=torch.cat([dogc[:,None,:].expand(-1,self.K,-1),taskc[None,:,:].expand(self.M,-1,-1),
                        td,torch.stack(extra)[None,:,:].expand(self.M,-1,-1)],dim=-1)
        x=torch.cat([ctx[None,None,:].expand(self.M,self.K,-1),dh[:,None,:].expand(-1,self.K,-1),
                     self.th[None,:,:].expand(self.M,-1,-1),geom],dim=-1)
        self.assignment_inputs=x  # graph-connected dog-task bindings; never CPU-cache learned vectors
        self.adm=policy.admissibility_net(x).squeeze(-1)
        self.unary=(policy.unary_net(x).squeeze(-1)+policy.admissibility_score_weight*F.logsigmoid(self.adm))
        if hasattr(policy,'competition_head') and getattr(policy,'residual_scale',1.)!=0:
            tm=torch.ones((1,self.K),dtype=torch.bool,device=g.device)
            self.unary=self.unary+float(getattr(policy,'residual_scale',1.))*policy.competition_head(x[None],tm)[0]
        self.task_edges=policy._edge_triplet_t(self.positions[:,None,:],self.positions[None,:,:],axis,lat)
        flags=[]
        for ti in self.pool:
            row=[]
            for tj in self.pool:
                both=ti.task_type==core.TASK_COLLECT and tj.task_type==core.TASK_COLLECT
                row.append([float(both and ti.component_members==tj.component_members),float(both),
                            float(ti.task_type==tj.task_type)])
            flags.append(row)
        self.flags=torch.tensor(flags,dtype=g.dtype,device=g.device)
        self.pairs={}

    def pair_block(self,i,j):
        key=(i,j)
        if key not in self.pairs:
            p=self.policy;K=self.K
            dg=p._edge_triplet_t(self.dogs[i,:2],self.dogs[j,:2],self.axis,self.lat)
            dr=p._edge_triplet_t(self.dogs[j,:2],self.dogs[i,:2],self.axis,self.lat)
            sep=self.task_edges[...,2:3]
            gij=torch.cat([dg.expand(K,K,3),self.task_edges,self.flags,sep],dim=-1)
            gji=torch.cat([dr.expand(K,K,3),self.task_edges.transpose(0,1),self.flags.transpose(0,1),sep],dim=-1)
            hi=self.th[:,None,:].expand(K,K,-1);hj=self.th[None,:,:].expand(K,K,-1)
            x=torch.cat([self.dh[i].expand(K,K,-1),self.dh[j].expand(K,K,-1),hi,hj,gij],dim=-1)
            r=torch.cat([self.dh[j].expand(K,K,-1),self.dh[i].expand(K,K,-1),hj,hi,gji],dim=-1)
            x=torch.stack([x,r],dim=2).reshape(K*K*2,-1)
            values=torch.cat([p.pair_net(y).squeeze(-1) for y in x.split(2048)])
            self.pairs[key]=values.reshape(K,K,2).mean(-1)
        return self.pairs[key]


    def scores(self,specs):
        ix=torch.tensor([[self.index[t.key()] for t in a.tasks_by_dog] for a in specs],
                        dtype=torch.long,device=self.g.device)
        dogs=torch.arange(self.M,device=self.g.device)[None,:]
        scores=self.unary[dogs,ix].mean(-1)
        if self.M>1 and self.policy.pairwise_weight!=0:
            pair=[]
            for i in range(self.M):
                for j in range(i+1,self.M):pair.append(self.pair_block(i,j)[ix[:,i],ix[:,j]])
            scores=scores+self.policy.pairwise_weight*pair_scale_factor(self.policy,self.M)*torch.stack(pair).mean(0)
        return scores,self.adm[dogs,ix]

@contextmanager
def actor_state_context(policy,state):
    """Bind this immutable decision packet for a scalar augmented-actor call."""
    if not hasattr(policy,'side_variant'):
        yield
        return
    from cogp.controls.actor import rows_from_state
    previous=policy._bound_actor_side
    policy._bound_actor_side=rows_from_state(policy,state)
    try:yield
    finally:policy._bound_actor_side=previous


def exact_forward(policy,state):
    with actor_state_context(policy,state):return _exact_forward(policy,state)


def _exact_forward(policy,state):
    from . import core
    if not state.specs:raise ValueError('Forward requires a nonempty prepared feasible support')
    if hasattr(policy,'family_name'):
        from .external_scores import table_for
        table=table_for(policy,state)
        indices=torch.as_tensor(table.indices(state.specs),device=table.device,dtype=torch.long)
        dogs=torch.arange(state.structure.active_dogs,device=table.device)[None,:]
        return table.score_indices(indices),table.state.edge_admiss[dogs,indices]
    g=torch.as_tensor(state.global_np,device=next(policy.parameters()).device)
    return Factors(policy,g,state.structure,state.phase,core).scores(state.specs)

@torch.no_grad()
def inference(policy,state):
    with actor_state_context(policy,state):return _inference(policy,state)


@torch.no_grad()
def _inference(policy,state):
    """Exact small-team inference and fixed4096 original-index beam.

    All methods use the same prefix order; finite-beam ordering is not claimed
    invariant to arbitrary dog relabeling. Exact score invariance is tested.
    """
    from . import core
    if hasattr(policy,'family_name'):
        from .external_scores import choose
        return choose(policy,state)
    if state.structure.active_dogs<=3:
        y,_=exact_forward(policy,state)
        if not torch.isfinite(y).all():raise RuntimeError('Nonfinite inference score')
        k=int(y.argmax())
        return state.specs[k],{'search':'exact','candidates':len(y),'truncated':False,'score':float(y[k]),
            'native_preserved':True}
    M=state.structure.active_dogs
    g=torch.as_tensor(state.global_np,device=next(policy.parameters()).device)
    S,D=policy.max_sheep,policy.max_dogs
    rows=state.global_np[5*S:5*S+4*D].reshape(D,4)[:M]
    order=tuple(range(M))
    pg=g  # original-index order is the identity; no reordering is performed
    tab=Factors(policy,pg,state.structure,state.phase,core)
    unary=tab.unary.cpu().double().numpy()
    pairs={(i,j):tab.pair_block(i,j).cpu().double().numpy() for i in range(M) for j in range(i+1,M)} if policy.pairwise_weight else {}
    if not np.isfinite(unary).all() or not all(np.isfinite(x).all() for x in pairs.values()):
        raise RuntimeError('Nonfinite factor table')
    scale=policy.pairwise_weight*pair_scale_factor(policy,M)*M/(M*(M-1)/2)
    beams=[(0.,(),frozenset())];expanded=0;truncated=False
    for j in range(M):
        nxt=[]
        for raw,ks,used in beams:
            for k,t in enumerate(tab.pool):
                if k in used:continue
                newks=ks+(k,)
                tasks=tuple(tab.pool[q] for q in newks)
                if not policy.space.is_prefix_extendable(tasks,state.structure,state.phase):continue
                val=raw+float(unary[j,k])+sum(scale*float(pairs[i,j][ki,k]) for i,ki in enumerate(ks)) if scale else raw+float(unary[j,k])
                nxt.append((val,newks,used|{k}));expanded+=1
        if not nxt:raise RuntimeError('No extendable beam; no widening/retry')
        nxt.sort(key=lambda x:x[0],reverse=True)
        truncated|=len(nxt)>4096
        beams=nxt[:4096]
    survivors=[b for b in beams if policy.space.is_valid(tuple(tab.pool[k] for k in b[1]),state.structure,state.phase)]
    if not survivors:raise RuntimeError('No feasible final beam')
    best=survivors[0]
    out=[None]*M
    for new,old in enumerate(order):out[old]=tab.pool[best[1][new]]
    return core.HCCGAssignmentSpec(tuple(out)),{'search':'beam4096_common_index_order',
        'expanded':expanded,'truncated':bool(truncated),'score':best[0]/M,'dog_order':list(order),
        'complete_candidates':len(survivors),'native_preserved':True}

def make_models(ec, pc, method='HCCG'):
    from . import core
    from .competition import CandidateCompetition
    from .interfaces import builder_for
    from cogp.learning.parent.protocol import METHODS
    cfg=METHODS[method]
    kw=dict(max_sheep=ec.max_sheep,max_dogs=ec.max_dogs,hidden=cfg['hidden'],
        world_size=ec.world_size,social_radius=ec.sheep_social_radius,
        sheep_max_neighbors=ec.sheep_max_neighbors,message_layers=2,
        sheep_per_pressure_slot=4,max_pressure_slots_per_component=3,
        admissibility_score_weight=0.1)
    if method=='HCCG':
        policy=core.HCCGPolicy(**kw,use_component_messages=True,pairwise_weight=0.5)
    else:
        from .baselines import make_external_policy
        policy=make_external_policy(cfg['family'],**kw,outlier_radius=75.0)
    policy.task_builder=builder_for(method,ec)
    policy.space=core.HCCGActionSpace(policy.task_builder)
    policy.mainline_method=method
    policy.residual_scale=1.0
    policy.pair_scale_mode='constant'
    if method=='HCCG':
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(torch.initial_seed()+16437)
            policy.competition_head=CandidateCompetition(3*cfg['hidden']+12,64,2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(torch.initial_seed()+31337)
        critic=core.CardinalityValue(ec.max_sheep,ec.max_dogs,pc.hidden_dim)
    return policy,critic

def score_assignments(policy, state, assignments):
    with actor_state_context(policy,state):return _score_assignments(policy,state,assignments)


def _score_assignments(policy, state, assignments):
    """Differentiable scores for explicitly ordered, feasible full assignments."""
    if hasattr(policy,'family_name'):
        from .external_scores import table_for
        table=table_for(policy,state)
        return table.score_indices(table.indices(assignments))
    g=torch.as_tensor(state.global_np,device=next(policy.parameters()).device)
    from . import core
    return Factors(policy,g,state.structure,state.phase,core).scores(assignments)[0]

@torch.no_grad()
def rank_assignments(policy,state,assignments):
    """Stable descending generator score, preserving input order at ties."""
    values=score_assignments(policy,state,assignments).detach().cpu().double().numpy()
    if not np.isfinite(values).all(): raise RuntimeError('Nonfinite allocation scores')
    order=np.argsort(-values,kind='stable')
    return [assignments[int(i)] for i in order],values[order]
