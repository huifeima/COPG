"""Fixed parent training protocol, with auditable configurable world addresses."""
from __future__ import annotations
from dataclasses import dataclass,asdict
import hashlib,json
SCHEMA='parent-training'
MODEL_SCHEMA='symmetric-allocation-ppo'
N_ENVS=8
ROLLOUT=1024
BLOCK=4096
TOTAL=32768
TRAIN_M4_CANDIDATES=256
TRAIN_BATCH_CANDIDATE_LIMIT=2048
TRAIN_SCENARIOS=('compact','single_outlier','double_outlier','split')
SEEDS=tuple(range(201,209))
PPO={'clip_ratio':.10,'value_clip':.20,'epochs':5,'reward_scale':.10,'entropy_coef':.01,
     'aux_coef':.10,'actor_microbatch':16,'critic_microbatch':256,'actor_lr':3e-4,
     'critic_lr':5e-4,'max_grad_norm':.5,'gamma':.99,'gae_lambda':.95}
METHODS={
    'HCCG':{'family':'hccg','hidden':128,'parameters':901700},
    'EdgeGNN':{'family':'magnet_edge_gnn','hidden':184,'parameters':862226},
    'AgentDAG':{'family':'gcs_agent_dag','hidden':152,'parameters':823845},
}
@dataclass(frozen=True)
class SeedAddress:
    origin:int=1_000_000_000_000
    seed_base:int=201
    seed_stride:int=10_000_000_000
    block_stride:int=1_000_000
    worker_stride:int=20_000
    scenario_offset:int=13_000_003
    def episode(self,seed,block,worker,episode):
        if not (seed>=self.seed_base and 0<=block<64 and 0<=worker<8 and 0<=episode<20000):
            raise ValueError('Training world address outside declared range')
        if self.worker_stride<20000 or self.block_stride<8*self.worker_stride or self.seed_stride<64*self.block_stride:
            raise ValueError('Seed addresses overlap')
        return self.origin+(seed-self.seed_base)*self.seed_stride+block*self.block_stride+worker*self.worker_stride+episode
DEFAULT_ADDRESS=SeedAddress()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def episode_seed(seed,block,wid,episode,address=None):return (address or DEFAULT_ADDRESS).episode(seed,block,wid,episode)
def quotas(rollout_index,total=ROLLOUT,n=N_ENVS):
    if total<n or total%n:raise ValueError('Equal logical worker quotas required')
    return [total//n]*n
def training_dogs(method,block,wid):return 2
def flat_mixture():return {s:.25 for s in TRAIN_SCENARIOS}
def settings(address=None):
    return dict(methods=METHODS,n_envs=N_ENVS,rollout=ROLLOUT,block=BLOCK,total=TOTAL,
                ppo=PPO,scenarios=flat_mixture(),seed_address=asdict(address or DEFAULT_ADDRESS))
