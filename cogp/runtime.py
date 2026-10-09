"""One self-contained stateful experiment interface for every coordinator."""
from __future__ import annotations
import copy
import dataclasses
import time
from functools import lru_cache
import numpy as np
import torch
from cogp.assets import AssetRegistry
from cogp.environment import EnvConfig,PlannerConfig,MultiDogHerdingEnv,GraphObservationBuilder
from cogp.environment.scenarios import reset_scaling_case
from cogp.control import build_frozen_execution
from cogp.coordination import core
from cogp.coordination.interfaces import builder_for
from cogp.coordination.snapshot import TaskSnapshot,validate_spec,task_key
from cogp.coordination.ranking import prepare,inference
from cogp.learning.parent.checkpoint import load_parent
from cogp.io import sha

@lru_cache(maxsize=24)
def _parent(path,method,device,checksum):
    return load_parent(path,method=method,device=device,expected_sha256=checksum)

@lru_cache(maxsize=24)
def _dynamics(path,device,checksum):
    from cogp.dynamics import load_model
    if sha(path)!=checksum:raise RuntimeError('Dynamics asset changed during loading')
    model=load_model(path,device=device)
    if model.checkpoint_sha256!=checksum:raise RuntimeError('Dynamics asset changed while loading')
    return model

METHODS={'hccg':'HCCG','edge_gnn':'EdgeGNN','agent_dag':'AgentDAG','geometry':'Geometry'}

class Session:
    def __init__(self,env,co,ob,policy,critic,value_norm,*,seed=0,method='HCCG',arm=None,predictor=None,asset_identity=None):
        self.env,self.co,self.ob=env,co,ob
        self.policy,self.critic,self.value_norm=policy,critic,value_norm
        self.seed,self.method=int(seed),method
        self.arm=dict(arm or {});self.predictor=predictor
        self.asset_identity=dict(asset_identity or {})
        self.ec,self.pc=env.cfg,co.pc
        self.tb=co.task_builder;self.executor=co.executor;self.controller=co.executor.low_level
        self.ctx=None;self.snapshot=None;self.done=False

    @classmethod
    def from_registry(cls,registry,seed,arm,device='cpu'):
        registry=registry if isinstance(registry,AssetRegistry) else AssetRegistry(registry)
        engineering=bool(arm.get('engineering_test',False))
        method=METHODS.get(arm['generator'],arm['generator'])
        parent_method='HCCG' if method=='Geometry' else method
        owner=parent_method if arm.get('critic')=='own' else 'HCCG'
        paths={}
        def asset(key):
            path=registry.resolve(key,require_sha=not engineering);paths[key]=sha(path);return path
        parent_key=f'parent.{seed}.{parent_method}'
        shared_key=f'parent.{seed}.{owner}'
        parent_path=asset(parent_key)
        parent=_parent(str(parent_path),parent_method,str(device),sha(parent_path))
        shared_path=asset(shared_key)
        shared=parent if owner==parent_method else _parent(str(shared_path),owner,str(device),sha(shared_path))
        if (shared.env_cfg.max_sheep,shared.env_cfg.max_dogs)!=(parent.env_cfg.max_sheep,parent.env_cfg.max_dogs) or shared.planner_cfg.gamma!=parent.planner_cfg.gamma:
            raise ValueError('Shared critic input capacities or discount differ from parent interface')
        for key,loaded in ((parent_key,parent),(shared_key,shared)):
            declared_seed=loaded.payload.get('job',{}).get('seed',loaded.payload.get('seed'))
            if declared_seed is not None and declared_seed!=seed:raise ValueError('Parent checkpoint training seed differs from asset key')
            if not engineering:
                from cogp.scientific import validate_parent
                validate_parent(loaded,seed=seed,receipts=registry.verified_receipts(key))
        if not engineering and (registry.data.get('test_only') or registry.data.get('scientific_policy') is False):
            raise ValueError('Engineering-test asset manifest cannot be used in a scientific panel')
        ec=copy.deepcopy(parent.env_cfg);pc=copy.deepcopy(parent.planner_cfg)
        if ec.max_sheep<20 or ec.max_dogs<6:
            raise ValueError('Evaluation requires parent observation capacities max_sheep>=20,max_dogs>=6; cannot alter trained tensor shapes.')
        controller,executor=build_frozen_execution(str(asset('basc.collect')),str(asset('basc.drive')),pc,device=device)
        if not engineering:
            for loaded in (parent,shared):
                for name in ('collect','drive'):
                    if loaded.payload['job']['basc_sha256'][name]!=paths['basc.'+name]:
                        raise ValueError('Frozen BASC asset differs from parent training identity: '+name)
        tb=builder_for(method,ec);co=core.HCCGCommittedCoordinator(executor,pc,tb,gamma=pc.gamma,max_option_high_steps=arm.get('E',64))
        ob=GraphObservationBuilder(ec.max_sheep,ec.max_dogs,pc.graph_radius,pc.dog_sheep_edge_radius,pc.use_ss_edges,pc.use_ds_edges,pc.use_dd_edges)
        predictor=None
        if arm.get('planner'):
            if arm.get('dynamics')=='true_diagnostic_only':
                raise ValueError('True simulator cannot be deployed as a learned dynamics asset')
            from cogp.dynamics import load_model
            dynamics_key=f"dynamics.{seed}.{arm.get('dynamics','graph')}"
            dynamics_path=asset(dynamics_key)
            predictor=_dynamics(str(dynamics_path),str(device),sha(dynamics_path))
            declared_seed=predictor.checkpoint_identity.get('seed',predictor.checkpoint_origin.get('source_training_seed'))
            if declared_seed is not None and declared_seed!=seed:raise ValueError('Dynamics checkpoint training seed differs from asset key')
            if predictor.kind!=arm.get('dynamics','graph'):raise ValueError('Dynamics checkpoint kind differs from requested arm')
            from cogp.dynamics.models import geometry_config
            if predictor.config!=geometry_config(ec):raise ValueError('Dynamics known physical scales differ from environment')
            if not engineering:
                from cogp.scientific import validate_dynamics
                validate_dynamics(predictor,seed=seed,kind=arm.get('dynamics','graph'),receipts=registry.verified_receipts(dynamics_key))
        policy=parent.policy
        if arm.get('actor','default')!='default':
            # Actor controls export a standalone, full-budget actor artifact.
            from cogp.controls.actor import load_actor
            from cogp.learning.parent.checkpoint import build_policy
            policy,actor_receipt=load_actor(asset(f"actor.{seed}.{arm['actor']}"),lambda:build_policy('HCCG',ec,pc),device=device)
            if actor_receipt['seed']!=seed or actor_receipt['variant']!=arm['actor']:raise ValueError('Actor checkpoint seed or feature variant differs from requested arm')
        critic=shared.critic;norm=shared.value_norm
        if arm.get('critic') in ('physical','execution'):
            from cogp.controls.critic import load_critic
            critic=load_critic(asset(f"critic.{seed}.{arm['critic']}"),device=device)
            if critic.payload['seed']!=seed or critic.arm!=arm['critic']:raise ValueError('Critic checkpoint seed or feature arm differs from requested arm')
            if critic.payload['final_update']!=5000 and not arm.get('engineering_test',False):raise ValueError('Critic checkpoint does not have the fixed 5000-update training budget')
        result=cls(MultiDogHerdingEnv(ec),co,ob,policy,critic,norm,seed=seed,method=method,arm=arm,predictor=predictor,asset_identity=paths)
        result.registry=registry
        return result

    def capture(self):
        snap=TaskSnapshot.capture(self.env,self.tb,self.ob,self.ctx.phase_id)
        if getattr(self.policy,'side_variant',None):
            from cogp.controls.actor import enrich_snapshot
            snap=enrich_snapshot(snap,self.env,self.co,self.ctx)
        self.snapshot=snap;return snap

    def reset(self,case):
        self.case=copy.deepcopy(case)
        # Restore the base coefficient rather than compounding stress variants.
        base=getattr(self,'_base_k_dog',self.ec.k_dog);self._base_k_dog=base
        self.ec.k_dog=base*float(case.get('physics_multiplier',1.0))
        self.env.reset(seed=int(case['seed']),scenario='compact',num_sheep=int(case['sheep']),num_dogs=int(case['dogs']))
        reset_scaling_case(self.env,int(case['seed']),str(case['scenario']))
        self.co.reset(self.env.num_dogs);self.ctx=self.co.prepare(self.env);self.done=False
        return self.capture()

    def native(self):
        if self.method=='Geometry':
            from cogp.coordination.geometry import geometry_candidates
            return geometry_candidates(prepare(self.snapshot),width=256)[0][0]
        return inference(self.policy,prepare(self.snapshot))[0]

    def step(self,spec):
        if self.done:raise RuntimeError('Cannot execute after episode termination')
        validate_spec(self.snapshot,self.env,self.tb,spec,core)
        start=self.env.primitive_step
        reward,done,info,duration,raw,pending=self.co.step_committed(self.env,self.ctx,spec)
        if duration<=0:raise RuntimeError('No progress in committed execution')
        self.done=bool(done or self.env.primitive_step>=self.ec.max_steps or self.env.metrics()['success']>.5)
        self.ctx=pending if pending is not None else self.ctx if self.done else self.co.prepare(self.env)
        self.capture()
        return dict(reward=float(reward),raw=float(raw),duration=int(duration),done=self.done,info=info,primitive_steps=int(self.env.primitive_step-start))

    def clone_view(self,rollout):
        result=copy.copy(self)
        result.env,result.co,result.ctx=rollout.env,rollout.co,rollout.ctx
        result.ec,result.pc=rollout.env.cfg,rollout.co.pc
        result.tb=rollout.co.task_builder;result.executor=rollout.co.executor;result.controller=result.executor.low_level
        result.done=rollout.done;result.capture()
        return result

    @torch.no_grad()
    def value(self,snapshot=None):
        if self.done:return 0.0
        if hasattr(self.critic,'value'):
            return float(self.critic.value(self,snapshot))
        g=torch.as_tensor(self.ob.global_state_vector(self.env),dtype=torch.float32,device=next(self.critic.parameters()).device)
        return float(self.value_norm.denormalize_scalar(float(self.critic(g))))


def run_episode(job,registry,device='cpu'):
    from cogp.planning.planner import Planner
    arm={**job['arm_config'],'engineering_test':bool(job.get('test_only',False))}
    session=Session.from_registry(registry,job['seed'],arm,device)
    session.reset(job['case'])
    planner=Planner(session,H=arm['H'],E=arm['E'],K=arm['K'],prior=arm.get('prior',0),memory=arm.get('memory','inherit'),proposal=arm.get('proposal','native')) if arm['planner'] else None
    cpu_start=time.process_time();wall_start=time.perf_counter();latencies=[];trace=[];reward=0.;discount=1.;decisions=0
    while not session.done:
        begin=time.perf_counter()
        choice=planner.select() if planner else None
        spec=choice.spec if choice else session.native()
        elapsed=time.perf_counter()-begin;latencies.append(elapsed)
        before=session.snapshot.sha256;step=session.step(spec)
        reward+=discount*.1*step['reward'];discount*=session.pc.gamma**step['duration']
        trace.append(dict(decision=decisions,snapshot_hash=before,selected_tasks=task_key(spec),selection_seconds=elapsed,
                          planner=choice.record if choice else None,step={k:v for k,v in step.items() if k!='info'}))
        decisions+=1
    result=dict(test_only=bool(job.get('test_only',False)),seed=job['seed'],arm=job['arm'],condition='|'.join(str(job['case'].get(k,'')) for k in ('sheep','dogs','kind','scenario','physics_multiplier')),
        world=job['case']['case_id'],success=int(session.env.metrics()['success']>.5),cpu_seconds=time.process_time()-cpu_start,
        wall_seconds=time.perf_counter()-wall_start,latency=latencies,decisions=decisions,primitive_steps=session.env.primitive_step,
        scaled_return=reward,assets=session.asset_identity,trace=trace,case=job['case'],environment=session.ec.to_dict(),execution=session.pc.to_dict())
    if job.get('panel')=='memory':
        # Separate shared-state diagnostic is produced by diagnostics; this trajectory alone is not a shared-candidate comparison.
        result['intervention_scope']='prediction branch designated history only'
    return result
