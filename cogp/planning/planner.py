"""Candidate-only execution-aware model rollouts and endpoint scoring."""
from __future__ import annotations
from dataclasses import dataclass
import math
import time
import numpy as np
from .proposals import build_candidates
from .branching import rollout_branch,forbid_true_forces,state_digest
from cogp.coordination.snapshot import task_key

REWARD_SCALE=.10

def score_components(reward,duration,done,value,gamma=.99,prior=0.,log_probability=0.):
    if duration<0 or not all(math.isfinite(float(x)) for x in (reward,value,gamma,prior,log_probability)):
        raise ValueError('Invalid rollout score component')
    reward_term=REWARD_SCALE*float(reward)
    tail_term=0. if done else float(gamma)**int(duration)*float(value)
    prior_term=float(prior)*float(log_probability)
    return dict(reward_term=reward_term,tail_term=tail_term,prior_term=prior_term,score=reward_term+tail_term+prior_term)

@dataclass
class Selection:
    spec:object
    record:dict

class Planner:
    def __init__(self,session,H=64,E=64,K=4,prior=0.,memory='inherit',proposal='native'):
        if any(type(x) is not int or x<1 for x in (H,E,K)):raise ValueError('Positive integer H,E,K required')
        if memory not in ('inherit','full','reinitialize','reinitialized'):raise ValueError('Unknown memory intervention')
        self.session,self.H,self.E,self.K=session,H,E,K
        self.prior,self.memory,self.proposal=float(prior),memory,proposal
        self._check_execution_budget()
        if session.predictor is None:raise ValueError('Formal COGP selection requires a fitted acceleration model')

    def _check_execution_budget(self):
        """E describes the live executor, not the cloned prediction horizon.

        Recheck at use time because callers can mutate the coordinator between
        decisions. Prediction branches may independently use H, including H>E.
        """
        actual=getattr(self.session.co,'max_option_high_steps',None)
        if isinstance(actual,bool) or not isinstance(actual,(int,np.integer)) or actual<1:
            raise ValueError('Live coordinator requires a positive integer execution budget')
        if int(actual)!=self.E:
            raise ValueError(f'Planner E={self.E} does not match live coordinator execution budget {actual}')

    def evaluate(self,candidates):
        self._check_execution_budget()
        session=self.session;before=state_digest(session.env,session.co,session.ctx)
        logits=np.asarray(candidates.scores,np.float64)
        # Common additive constants do not affect candidate ordering.
        shift=logits-logits.max();log_probs=shift-np.log(np.exp(shift).sum())
        records=[]
        with forbid_true_forces(session.env):
            for order,(spec,logp) in enumerate(zip(candidates.specs,log_probs)):
                start=time.perf_counter()
                if session.arm.get('critic') in ('physical','execution'):
                    if self.H>self.E:raise ValueError('Matched unfinished-commitment critics require H<=E')
                    from .commitment import start as begin,advance
                    handle=begin(session,spec,E=self.E,predictor=session.predictor)
                    if self.memory in ('reinitialize','reinitialized'):
                        from .memory import reinitialize_history
                        # Memory controls use the default critic and one-shot path; avoid resetting after execution conversion.
                        raise ValueError('Matched-critic and history interventions are separate panels')
                    endpoint=advance(handle,self.H);local=endpoint.session
                    value=0. if endpoint.done else local.value(endpoint)
                else:
                    endpoint=rollout_branch(session.env,session.co,session.ctx,spec,session.predictor,self.H,
                        'full' if self.memory in ('inherit','full') else 'reinitialized')
                    local=session.clone_view(endpoint);value=local.value()
                components=score_components(endpoint.reward,endpoint.duration,endpoint.done,value,session.pc.gamma,self.prior,float(logp))
                records.append(dict(order=order,tasks=task_key(spec),predicted_reward=endpoint.reward,predicted_duration=endpoint.duration,
                    predicted_terminal=endpoint.done,predicted_value=value,primitive_steps=endpoint.primitive_steps,
                    seconds=time.perf_counter()-start,**components))
        if state_digest(session.env,session.co,session.ctx)!=before:raise RuntimeError('Planner modified real execution state')
        return records

    def select(self,candidates=None):
        self._check_execution_budget()
        start=time.perf_counter();candidates=candidates or build_candidates(self.session,self.K,self.proposal)
        rows=self.evaluate(candidates)
        selected=int(np.argmax([row['score'] for row in rows])) # First/native candidate wins exact ties.
        return Selection(candidates.specs[selected],dict(selected_index=selected,candidates=rows,
            candidate_info=candidates.info,H=self.H,E=self.E,K_requested=self.K,effective_K=len(rows),memory=self.memory,
            seconds=time.perf_counter()-start))
