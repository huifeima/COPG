"""Standalone eight-stream PPO parent training and atomic resumable checkpoints."""
from __future__ import annotations
import argparse,json,time
from dataclasses import asdict
from pathlib import Path
from .io import configure_torch,atomic_json,atomic_torch,atomic_copy,sha,emit,lock,rng_state,restore_rng,tensor_digest
from .protocol import SCHEMA,PPO,TOTAL,BLOCK,ROLLOUT,METHODS,N_ENVS,SeedAddress,DEFAULT_ADDRESS,settings,digest

def configs():
    from cogp.environment import EnvConfig,PlannerConfig
    ec=EnvConfig(num_sheep=15,num_dogs=2,max_sheep=40,max_dogs=6,seed=0)
    pc=PlannerConfig(hidden_dim=128,option_horizon=6,rollout_steps=ROLLOUT,basc_blend=1.,
        actor_lr=PPO['actor_lr'],critic_lr=PPO['critic_lr'],clip_ratio=PPO['clip_ratio'],
        ppo_epochs=PPO['epochs'],gamma=PPO['gamma'],gae_lambda=PPO['gae_lambda'],max_grad_norm=PPO['max_grad_norm'])
    return ec,pc

def latest_payload(out):
    import torch
    out=Path(out);index=out/'latest.json'
    if not index.exists():return None
    row=json.loads(index.read_text());path=out/row['file']
    if sha(path)!=row['sha256']:raise RuntimeError('Committed checkpoint hash mismatch')
    result=torch.load(path,map_location='cpu',weights_only=False)
    if result['learner']['decisions']!=row['decisions']:raise RuntimeError('Checkpoint decision counter mismatch')
    report=out/result['block_report_file']
    if sha(report)!=result['block_report_sha256']:raise RuntimeError('Committed block report changed')
    return result

def train(out,method,seed,device,collect_checkpoint,drive_checkpoint,*,
          study_hash=None,seed_address=None,actor_variant=None,_test_limits=None):
    """Train from random initialisation; select the final budget checkpoint.

    _test_limits is explicitly recorded and prevents a smoke checkpoint being
    mistaken for the paper budget. Production budget is 32768 decisions.
    """
    from .learner import Learner
    from .workers import Pool
    if method not in METHODS:raise ValueError('Unsupported parent method')
    if actor_variant is not None and method!='HCCG':raise ValueError('Matched actor controls use HCCG')
    if actor_variant is not None and seed_address is None:
        # The matched actor protocol's address is recorded in its source.
        from cogp.controls.actor import episode_seed as actor_episode_seed
        address=SeedAddress(origin=actor_episode_seed(201,0,0,0))
    else:
        address=seed_address or DEFAULT_ADDRESS
    limits=dict(total=TOTAL,block=BLOCK,rollout=ROLLOUT)
    if _test_limits is not None:
        if set(_test_limits)!=set(limits):raise ValueError('All smoke limits required')
        limits={k:int(v) for k,v in _test_limits.items()}
        if not 0<limits['total']<TOTAL:raise ValueError('Smoke training must have a smaller budget')
    if limits['total']%limits['block'] or limits['block']%limits['rollout'] or limits['rollout']%N_ENVS:
        raise ValueError('Budget must partition into equal eight-stream rollouts')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    collect,drive=Path(collect_checkpoint).resolve(),Path(drive_checkpoint).resolve()
    for path in (collect,drive):
        if not path.is_file():raise FileNotFoundError(path)
    with lock(out/'.train.lock'):
        configure_torch(seed,device);ec,pc=configs();learner=Learner(ec,pc,device,method)
        if actor_variant is not None:
            from cogp.controls.actor import configure_learner
            configure_learner(learner,actor_variant)
        job=dict(schema=SCHEMA,method=method,seed=int(seed),device=str(device),
                 env_cfg=ec.to_dict(),planner_cfg=pc.to_dict(),protocol=settings(address),
                 basc_sha256={'collect':sha(collect),'drive':sha(drive)},
                 actor_variant=actor_variant,initialization='random',checkpoint_selection='final_budget',
                 limits=limits,test_only=_test_limits is not None,
                 initial_actor_digest=tensor_digest(learner.policy),initial_critic_digest=tensor_digest(learner.critic),
                 actor_parameters=sum(p.numel() for p in learner.policy.parameters()),
                 critic_parameters=sum(p.numel() for p in learner.critic.parameters()))
        job['study_hash']=study_hash or digest(job)
        jobpath=out/'job.json'
        if jobpath.exists() and json.loads(jobpath.read_text())!=job:raise RuntimeError('Training identity changed')
        if not jobpath.exists():atomic_json(jobpath,job)
        previous=latest_payload(out)
        if previous is not None:
            if previous['job']!=job:raise RuntimeError('Resume identity mismatch')
            learner.restore(previous['learner']);restore_rng(previous['rng'])
        if learner.decisions%limits['block'] or learner.decisions>limits['total']:raise RuntimeError('Invalid resume counter')
        started=time.perf_counter()
        if learner.decisions<limits['total']:
            with Pool(ec,pc,collect,drive,N_ENVS,method=method,seed_address=address,actor_variant=actor_variant) as pool:
                while learner.decisions<limits['total']:
                    block=learner.decisions//limits['block'];pool.reset_training(seed,block);rows=[];episodes=[]
                    for _ in range(limits['block']//limits['rollout']):
                        records,collection=learner.collect(pool,limits['rollout']);stats=learner.update(records)
                        rows.append(dict(decisions=learner.decisions,primitive_steps=learner.primitive_steps,
                                         high_steps=learner.high_steps,collection={k:v for k,v in collection.items() if k!='episodes'},**stats))
                        episodes.extend(collection['episodes'])
                        atomic_json(out/'progress.json',dict(status='TRAINING',seed=seed,method=method,last=rows[-1]))
                    report=dict(block=block,training=rows,episodes=episodes,
                                boundary_reset='unfinished segments bootstrapped; reset at every block boundary')
                    report_path=out/'blocks'/f'block_{block:03d}.json';atomic_json(report_path,report)
                    payload=dict(schema=SCHEMA,job=job,learner=learner.payload(),rng=rng_state(),
                                 block_report_file=str(report_path.relative_to(out)),block_report_sha256=sha(report_path))
                    checkpoint=out/'checkpoints'/f'step_{learner.decisions:06d}.pt';atomic_torch(checkpoint,payload)
                    atomic_json(out/'latest.json',dict(file=str(checkpoint.relative_to(out)),sha256=sha(checkpoint),decisions=learner.decisions))
                    emit('PARENT_BLOCK_COMMITTED',method=method,seed=seed,decisions=learner.decisions)
        index=json.loads((out/'latest.json').read_text());checkpoint=out/index['file']
        final=out/'final.pt'
        if final.exists() and sha(final)!=index['sha256']:raise RuntimeError('Different final checkpoint already exists')
        if not final.exists():atomic_copy(checkpoint,final)
        receipt=dict(status='COMPLETE',study_hash=job['study_hash'],method=method,seed=seed,
                     actor_variant=actor_variant,test_only=_test_limits is not None,
                     decisions=learner.decisions,primitive_steps=learner.primitive_steps,high_steps=learner.high_steps,
                     checkpoint_sha256=sha(final),actor_digest=tensor_digest(learner.policy),
                     seconds_this_process=time.perf_counter()-started,seed_address=asdict(address))
        if actor_variant is not None and _test_limits is None:
            from cogp.controls.actor import save_actor
            exported=save_actor(learner.policy,out/'actor.pt',seed=seed,decisions=learner.decisions,
                                metadata={'study_hash':job['study_hash'],'seed_address':asdict(address)})
            receipt['actor_export']=exported
        atomic_json(out/'complete.json',receipt)
        # A byte-bound per-checkpoint receipt travels with AssetRegistry.add.
        # complete.json remains the resumable training-directory receipt.
        atomic_json(final.with_suffix('.json'),receipt)
        (out/'complete.sha256').write_text(sha(out/'complete.json')+'\n')
        return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    p.add_argument('--method',choices=list(METHODS),required=True);p.add_argument('--seed',type=int,default=201)
    p.add_argument('--device',default='cpu');p.add_argument('--collect-checkpoint',required=True);p.add_argument('--drive-checkpoint',required=True)
    p.add_argument('--seed-address',help='JSON file containing explicit SeedAddress fields')
    a=p.parse_args();address=SeedAddress(**json.loads(Path(a.seed_address).read_text())) if a.seed_address else DEFAULT_ADDRESS
    train(a.out,a.method,a.seed,a.device,a.collect_checkpoint,a.drive_checkpoint,seed_address=address)
if __name__=='__main__':main()
