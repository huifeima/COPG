"""Feasible native-first proposal lists with explicit stable ordering."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from cogp.coordination.ranking import prepare,inference,rank_assignments
from cogp.coordination.geometry import geometry_candidates
from cogp.coordination.snapshot import task_key,candidate_digest

@dataclass
class CandidateSet:
    specs:list
    scores:np.ndarray
    native:object
    info:dict


def unique(specs):
    result=[];seen=set()
    for spec in specs:
        key=task_key(spec)
        if key not in seen:seen.add(key);result.append(spec)
    return result


def build_candidates(session,K=4,proposal='native'):
    if type(K) is not int or K<1:raise ValueError('K must be a positive integer')
    state=prepare(session.snapshot)
    if session.method=='Geometry':
        candidates,info=geometry_candidates(state,width=min(4096,max(256,16*K)))
        specs=candidates[:K]
        # Geometry chooses minimum total assigned travel; scores have opposite sign.
        from cogp.coordination.geometry import geometry_table
        pool,cost=geometry_table(state);ix={t.key():i for i,t in enumerate(pool)}
        values=np.asarray([-sum(cost[j,ix[t.key()]] for j,t in enumerate(s.tasks_by_dog)) for s in specs])
        return CandidateSet(specs,values,specs[0],{**info,'effective_K':len(specs),'candidate_hash':candidate_digest(specs)})
    native,native_info=inference(session.policy,state)
    if proposal not in ('native','original'):
        from cogp.refinements import expanded_support,select_refinement
        support=expanded_support(session)
        path=None
        if proposal in ('return_rank','winner'):
            path=session.registry.resolve(f'proposal.{session.seed}.{session.method}.{proposal}')
        specs,scores,info=select_refinement(session,support,support.scores,proposal,K,path)
        return CandidateSet(specs,np.asarray(scores),native,{**info,'effective_K':len(specs),'candidate_hash':candidate_digest(specs)})
    if state.structure.active_dogs<=3:
        candidates=state.specs
        info={'search':'exact_feasible_support'}
    else:
        geo,info=geometry_candidates(state,width=min(4096,max(256,16*K)))
        candidates=unique([native,*geo])
        info={**info,'native_search':native_info}
    ranked,scores=rank_assignments(session.policy,state,candidates)
    nk=task_key(native);index=next(i for i,s in enumerate(ranked) if task_key(s)==nk)
    specs=[native]+[s for s in ranked if task_key(s)!=nk][:K-1]
    bykey={task_key(s):float(v) for s,v in zip(ranked,scores)}
    values=np.asarray([bykey[task_key(s)] for s in specs],np.float64)
    return CandidateSet(specs,values,native,{**info,'effective_K':len(specs),'native_retained':True,'candidate_hash':candidate_digest(specs)})
