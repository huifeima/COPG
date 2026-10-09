"""Strict parent loading and explicit historical tensor-key conversion."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math,re
from typing import Any
from .io import sha

# These are compatibility inputs only; new model files and tensor keys are
# always saved through the neutral package and schema.
ARCHIVED_METHODS={'M2__HCCG_Comp2':'HCCG','REF32_HCCG_Comp2':'HCCG',
                  'M2__EdgeGNN':'EdgeGNN','REF32_EdgeGNN':'EdgeGNN',
                  'M2__AgentDAG':'AgentDAG','REF32_AgentDAG':'AgentDAG'}
ARCHIVED_TENSOR_PREFIXES={}

def convert_tensor_keys(state_dict,prefix_map=None):
    """Explicit one-to-one conversion; collisions fail instead of losing weights."""
    mapping=ARCHIVED_TENSOR_PREFIXES if prefix_map is None else prefix_map
    converted={};changes=[]
    for key,value in state_dict.items():
        target=key
        if prefix_map is None:
            matched=re.fullmatch(r'[vV]\d+_(side_encoder|side_score)(\..+)',key)
            if matched:target=matched.group(1)+matched.group(2)
        for old,new in mapping.items():
            if key.startswith(old):target=new+key[len(old):];break
        if target in converted:raise ValueError('Tensor-key conversion collision: '+target)
        converted[target]=value
        if target!=key:changes.append({'source':key,'target':target})
    return converted,changes

@dataclass
class ParentModels:
    policy:Any
    critic:Any
    value_norm:Any
    env_cfg:Any
    planner_cfg:Any
    payload:dict
    checkpoint_sha256:str
    method:str

def build_policy(method,env_cfg,planner_cfg,device='cpu'):
    from cogp.coordination.ranking import make_models
    actor,_=make_models(env_cfg,planner_cfg,method)
    return actor.to(device)

def load_parent(path,*,method=None,device='cpu',expected_sha256=None,allow_archived=True):
    """Load complete actor/critic/normalizer with strict architecture checks.

    Archived model metadata is converted explicitly. No neural parameters are
    ignored, inferred from performance, or replaced on a shape mismatch.
    """
    import torch
    from cogp.environment import EnvConfig,PlannerConfig
    from cogp.coordination.ranking import make_models
    from cogp.coordination.value import RunningValueNorm
    path=Path(path)
    if path.is_dir():path=path/'final.pt'
    before=sha(path)
    if expected_sha256 and before!=expected_sha256:raise RuntimeError('Parent checkpoint SHA256 mismatch')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if sha(path)!=before:raise RuntimeError('Checkpoint changed while loading')
    if not isinstance(payload,dict):raise ValueError('Expected a parent checkpoint mapping')
    learner=payload.get('learner',payload)
    job=payload.get('job',{})
    source=learner.get('method',job.get('method'))
    source=ARCHIVED_METHODS.get(source,source) if allow_archived else source
    name=method or source
    if name not in ('HCCG','EdgeGNN','AgentDAG'):raise ValueError('Explicit supported parent method required')
    if source is not None and name!=source:raise ValueError('Checkpoint allocation method differs from request')
    ec=EnvConfig(**learner.get('env_cfg',job.get('env_cfg',{})))
    pc=PlannerConfig(**learner.get('planner_cfg',job.get('planner_cfg',{})))
    actor,critic=make_models(ec,pc,name)
    if job.get('actor_variant') is not None or job.get('variant') is not None:
        from cogp.controls.actor import attach_side_encoder
        attach_side_encoder(actor,job.get('actor_variant',job.get('variant')))
    actor_state,changes=convert_tensor_keys(learner['policy']) if allow_archived else (learner['policy'],[])
    for module,state in ((actor,actor_state),(critic,learner['critic'])):
        expected=module.state_dict()
        if set(expected)!=set(state):raise RuntimeError('Parent tensor keys do not exactly match the architecture')
        for key,tensor in state.items():
            if not isinstance(tensor,torch.Tensor) or tensor.shape!=expected[key].shape or tensor.dtype!=expected[key].dtype:
                raise RuntimeError('Parent tensor shape or dtype mismatch: '+key)
            if not torch.isfinite(tensor).all():raise RuntimeError('Nonfinite parent tensor: '+key)
    actor.load_state_dict(actor_state,strict=True);critic.load_state_dict(learner['critic'],strict=True)
    norm=RunningValueNorm();norm.load_state_dict(learner['value_norm'])
    if not all(math.isfinite(v) for v in (norm.mean,norm.var,norm.count)) or norm.var<=0 or norm.count<=0:
        raise ValueError('Invalid checkpoint value normalization')
    actor=actor.to(device).eval().requires_grad_(False);critic=critic.to(device).eval().requires_grad_(False)
    payload={**payload,'explicit_tensor_key_conversion':changes}
    return ParentModels(actor,critic,norm,ec,pc,payload,before,name)
