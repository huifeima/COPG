from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
import torch
import torch.nn.functional as F
from .io import emit,tensor_digest
from cogp.coordination.ranking import prepare,exact_forward,make_models
from .protocol import PPO,MODEL_SCHEMA,quotas
from cogp.coordination.batch import batch_forward

@dataclass
class Record:
    state: object
    selected: int
    selected_key: tuple
    candidate_hash: str
    old_logp: float
    value: float
    reward: float
    raw_reward: float
    duration: int
    primitive_duration: int
    done: bool
    y: np.ndarray
    mask: np.ndarray
    advantage: float=0.
    target: float=0.

def gae(rewards,values,durations,dones,last_value,gamma=.99,lam=.95):
    rewards=np.asarray(rewards,np.float64);values=np.asarray(values,np.float64)
    out=np.zeros(len(rewards),np.float64);carry=0.
    for t in reversed(range(len(rewards))):
        non=1.-float(dones[t]);dt=int(durations[t])
        if dt<1:raise ValueError('Nonpositive SMDP duration')
        nv=float(last_value) if t==len(rewards)-1 else values[t+1]
        delta=rewards[t]+gamma**dt*nv*non-values[t]
        carry=delta+(gamma*lam)**dt*non*carry;out[t]=carry
    return out.astype(np.float32),(out+values).astype(np.float32)

class Learner:
    def __init__(self,ec,pc,device,method="HCCG"):
        from cogp.coordination.value import RunningValueNorm
        self.ec,self.pc,self.device=ec,pc,torch.device(device)
        self.method=method
        self.policy,self.critic=make_models(ec,pc,method)
        self.policy=self.policy.to(self.device);self.critic=self.critic.to(self.device)
        self.policy.eval();self.critic.eval() # no Dropout/BatchNorm in audited core
        from .protocol import METHODS
        self.actor_opt=torch.optim.Adam([p for p in self.policy.parameters() if p.requires_grad],lr=PPO['actor_lr'],eps=1e-5)
        self.critic_opt=torch.optim.Adam(self.critic.parameters(),lr=PPO['critic_lr'],eps=1e-5)
        self.norm=RunningValueNorm();self.decisions=0;self.primitive_steps=0;self.high_steps=0
        self.episodes=0;self.successes=0
    @torch.no_grad()
    def values(self,states):
        g=torch.tensor(np.stack([s.global_np for s in states]),device=self.device)
        z=self.critic(g).detach().cpu().numpy()
        return np.asarray([self.norm.denormalize_scalar(float(v)) for v in z],np.float32)
    @torch.no_grad()
    def collect(self,pool,total=1024):
        q=quotas(self.decisions//1024,total,pool.n)
        per=[[] for _ in range(pool.n)];pending={};eps=[];raw_hist={};terms={}
        start=time.perf_counter();worker_seconds=0.;dispatched=0;completed=0;mark=256
        score_seconds=0.;base_sum=shape_sum=0.;dog_hist={};last_heartbeat=start
        def dispatch(indices):
            nonlocal dispatched,score_seconds
            if not indices:return
            tick=time.perf_counter();ss=[prepare(pool.states[i],training=True) for i in indices]
            scores,_,_=batch_forward(self.policy,ss)
            logps=torch.log_softmax(scores,1).cpu().numpy();vs=self.values(ss)
            score_seconds+=time.perf_counter()-tick
            for ii,(i,s) in enumerate(zip(indices,ss)):
                if len(per[i])>=q[i] or i in pending:raise RuntimeError('Bad async quota state')
                lp=logps[ii,:len(s.specs)].astype(np.float64);p=np.exp(lp);p/=p.sum()
                a=int(pool.action_rng[i].choice(len(p),p=p))
                pending[i]=(s,a,float(lp[a]),float(vs[ii]))
                pool.send(i,{'cmd':'step','spec':s.specs[a]});dispatched+=1
        dispatch(list(range(pool.n)))
        while pending:
            ready=pool.ready(pending);again=[]
            for i in ready:
                tr=pool.recv(i);s,a,lp,v=pending.pop(i)
                per[i].append(Record(s,a,s.keys[a],s.candidate_hash,lp,v,
                   PPO['reward_scale']*tr['reward'],tr['raw_reward'],tr['duration'],tr['primitive_duration'],
                   tr['done'],np.asarray(tr['y'],np.float32),np.asarray(tr['mask'],np.float32)))
                dog_hist[str(s.structure.active_dogs)]=dog_hist.get(str(s.structure.active_dogs),0)+1
                self.primitive_steps+=tr['primitive_duration'];self.high_steps+=tr['duration']
                worker_seconds+=tr['worker_step_seconds'];base_sum+=tr['base_reward'];shape_sum+=tr['shaping_applied']
                for k,value in tr['reward_terms'].items():terms[k]=terms.get(k,0.)+value
                name=pool.metas[i]['profile'];raw_hist[name]=raw_hist.get(name,0)+1
                if tr['episode'] is not None:
                    eps.append(tr['episode']);self.episodes+=1;self.successes+=tr['episode']['success']
                pool.states[i]=tr['next_encoded'];pool.metas[i]=tr['next_meta'];completed+=1
                if len(per[i])<q[i]:again.append(i)
            dispatch(again)
            if completed>=mark or time.perf_counter()-last_heartbeat>60:
                emit('ASYNC_ROLLOUT',completed=completed,total=total,in_flight=len(pending),quotas=q)
                mark=((completed//256)+1)*256;last_heartbeat=time.perf_counter()
        if completed!=total or dispatched!=total or [len(x) for x in per]!=q:raise RuntimeError('Unequal/missing rollout contributions')
        last=self.values([prepare(x,training=True) for x in pool.states])
        for i,rr in enumerate(per):
            av,ret=gae([r.reward for r in rr],[r.value for r in rr],[r.duration for r in rr],
                       [r.done for r in rr],last[i],PPO['gamma'],PPO['gae_lambda'])
            for r,a,t in zip(rr,av,ret):r.advantage=float(a);r.target=float(t)
        records=[r for rr in per for r in rr] # canonical worker/time order, not completion order
        av=np.array([r.advantage for r in records],np.float32);av=(av-av.mean())/(av.std()+1e-8)
        for r,a in zip(records,av):r.advantage=float(a)
        self.decisions+=total;elapsed=time.perf_counter()-start
        return records,{'seconds':elapsed,'episodes':eps,'decision_profile_counts':raw_hist,
           'decision_dog_counts':dog_hist,'quotas':q,'worker_step_seconds_sum':worker_seconds,'worker_busy_fraction_estimate':worker_seconds/(pool.n*elapsed),
           'scoring_seconds':score_seconds,'base_discounted_option_mean':base_sum/total,
           'shaping_mean_raw_per_option':shape_sum/total,'reward_terms_sum':terms,
           'decisions_per_second':total/elapsed,'partial_episodes_active_at_end':sum(not rr[-1].done for rr in per)}

    @torch.no_grad()
    def check_replay(self,records):
        maxgap=0.
        for si in range(0,len(records),PPO['actor_microbatch']):
            rr=records[si:si+PPO['actor_microbatch']];states=[]
            for r in rr:
                rebuilt=prepare(r.state.snapshot.encoded,training=True)
                if rebuilt.candidate_hash!=r.candidate_hash or rebuilt.keys!=r.state.keys or rebuilt.keys[r.selected]!=r.selected_key:
                    raise RuntimeError('Stored candidate/action identity changed')
                states.append(rebuilt)
            scores,_,_=batch_forward(self.policy,states)
            lp=torch.log_softmax(scores,1)[torch.arange(len(rr),device=self.device),torch.tensor([r.selected for r in rr],device=self.device)].cpu().numpy()
            old=np.asarray([r.old_logp for r in rr]);gap=np.abs(lp-old)
            maxgap=max(maxgap,float(gap.max()))
            if np.any(gap>1e-5+1e-5*np.abs(old)):raise RuntimeError(f'Old logp replay mismatch: {gap.max()}')
        return maxgap

    def update(self,records,actor_mb=None,critic_mb=None,epochs=None):
        start=time.perf_counter();T=len(records)
        if not T:raise ValueError('Empty rollout')
        replay=self.check_replay(records)
        targets=np.asarray([r.target for r in records],np.float32);oldvalues=np.asarray([r.value for r in records],np.float32)
        self.norm.update(targets)
        rn=torch.tensor(self.norm.normalize_np(targets),device=self.device);ov=torch.tensor(self.norm.normalize_np(oldvalues),device=self.device)
        gs=torch.tensor(np.stack([r.state.global_np for r in records]),device=self.device)
        den=max(1.,float(sum(r.mask.sum() for r in records)))
        am=actor_mb or (8 if hasattr(self.policy,'family_name') else PPO['actor_microbatch']);cm=critic_mb or PPO['critic_microbatch'];ne=epochs or PPO['epochs'];hist=[]
        for epoch in range(ne):
            self.actor_opt.zero_grad(set_to_none=True);tot=torch.zeros(5,device=self.device)
            for si in range(0,T,am):
                rr=records[si:si+am];scores,adm,valid=batch_forward(self.policy,[r.state for r in rr]);B=len(rr)
                lpall=torch.log_softmax(scores,1);prob=lpall.exp();ent=-(prob*lpall).sum(1)
                idx=torch.tensor([r.selected for r in rr],device=self.device);bi=torch.arange(B,device=self.device)
                lp=lpall[bi,idx];old=torch.tensor([r.old_logp for r in rr],device=self.device)
                av=torch.tensor([r.advantage for r in rr],device=self.device);logratio=lp-old;ratio=logratio.exp()
                surrogate=-torch.minimum(ratio*av,ratio.clamp(1-PPO['clip_ratio'],1+PPO['clip_ratio'])*av)
                lossvec=surrogate-PPO['entropy_coef']*ent
                width=adm.shape[-1]
                mask=torch.tensor(np.stack([np.pad(r.mask,(0,width-len(r.mask))) for r in rr]),device=self.device)
                labels=torch.tensor(np.stack([np.pad(r.y,(0,width-len(r.y))) for r in rr]),device=self.device)
                aux=(F.binary_cross_entropy_with_logits(adm[bi,idx],labels,reduction='none')*mask).sum()
                loss=lossvec.sum()/T+PPO['aux_coef']*aux/den
                if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite actor loss')
                loss.backward()
                tot+=torch.stack([lossvec.detach().sum(),ent.detach().sum(),aux.detach(),
                      ((ratio.detach()-1).abs()>PPO['clip_ratio']).sum(),((ratio-1)-logratio).detach().sum()])
                del scores,adm,lpall,prob,ent,lp,ratio,loss,lossvec,aux,surrogate,logratio
            ag=torch.nn.utils.clip_grad_norm_(self.policy.parameters(),PPO['max_grad_norm'],error_if_nonfinite=True)
            self.actor_opt.step()
            self.critic_opt.zero_grad(set_to_none=True);vt=torch.zeros(2,device=self.device)
            for si in range(0,T,cm):
                ei=min(T,si+cm);pred=self.critic(gs[si:ei]);clipped=ov[si:ei]+(pred-ov[si:ei]).clamp(-PPO['value_clip'],PPO['value_clip'])
                terms=torch.maximum(F.smooth_l1_loss(pred,rn[si:ei],reduction='none'),F.smooth_l1_loss(clipped,rn[si:ei],reduction='none'))
                vl=terms.sum()/T
                if not bool(torch.isfinite(vl)):raise RuntimeError('Nonfinite critic loss')
                vl.backward();vt+=torch.stack([terms.detach().sum(),((pred.detach()-ov[si:ei]).abs()>PPO['value_clip']).sum()])
            vg=torch.nn.utils.clip_grad_norm_(self.critic.parameters(),PPO['max_grad_norm'],error_if_nonfinite=True)
            self.critic_opt.step()
            # Coalesced metrics transfers once/epoch, rather than per record.
            t=tot.cpu().tolist();v=vt.cpu().tolist()
            h={'epoch':epoch+1,'actor_loss':t[0]/T,'entropy':t[1]/T,'aux_loss':t[2]/den,
               'clipfrac':t[3]/T,'approx_kl':t[4]/T,'critic_loss':v[0]/T,'value_clipfrac':v[1]/T,
               'actor_grad_norm':float(ag),'critic_grad_norm':float(vg)}
            hist.append(h);emit('PPO_EPOCH',**h)
        out={k:float(np.mean([h[k] for h in hist])) for k in hist[0] if k!='epoch'}
        out.update(seconds=time.perf_counter()-start,replay_max_logp_error=replay,actor_optimizer_steps=ne,
                   critic_optimizer_steps=ne,samples=T,mean_candidates=float(np.mean([len(r.state.specs) for r in records])),
                   mean_duration=float(np.mean([r.duration for r in records])),
                   actor_parameters_with_gradient=sum(p.numel() for p in self.policy.parameters() if p.grad is not None))
        return out

    def payload(self):
        return {'model_schema':MODEL_SCHEMA,'requires_explicit_snapshot':True,'pair_mode':('NATIVE_EXTERNAL' if hasattr(self.policy,'family_name') else 'SYMMETRIC'),'method':self.method,
                'policy':self.policy.state_dict(),'critic':self.critic.state_dict(),
                'actor_opt':self.actor_opt.state_dict(),'critic_opt':self.critic_opt.state_dict(),
                'value_norm':self.norm.state_dict(),'env_cfg':self.ec.to_dict(),'planner_cfg':self.pc.to_dict(),
                'decisions':self.decisions,'primitive_steps':self.primitive_steps,'high_steps':self.high_steps,
                'episodes':self.episodes,'successes':self.successes}
    def restore(self,p):
        if p.get('model_schema')!=MODEL_SCHEMA or p.get('method')!=self.method:raise RuntimeError('Wrong checkpoint schema; never load with legacy forward')
        self.policy.load_state_dict(p['policy'],strict=True);self.critic.load_state_dict(p['critic'],strict=True)
        self.actor_opt.load_state_dict(p['actor_opt']);self.critic_opt.load_state_dict(p['critic_opt']);self.norm.load_state_dict(p['value_norm'])
        for k in ('decisions','primitive_steps','high_steps','episodes','successes'):setattr(self,k,int(p[k]))
