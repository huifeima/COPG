"""One independent CPU BASC + simulator per worker, explicit task snapshots."""
from __future__ import annotations
import multiprocessing as mp
import os,time,traceback
from .io import emit

def _worker(conn,cfg):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    try:
        import numpy as np,torch
        torch.set_num_threads(1)
        try:torch.set_num_interop_threads(1)
        except RuntimeError:pass
        from cogp.environment import EnvConfig,PlannerConfig
        from cogp.environment import MultiDogHerdingEnv
        from cogp.environment import GraphObservationBuilder
        from cogp.control import build_frozen_execution
        from cogp.coordination import core
        from cogp.coordination.snapshot import TaskSnapshot,validate_spec
        from cogp.environment.scenarios import reset_scaling_case
        from .protocol import episode_seed,digest,TRAIN_SCENARIOS,training_dogs
        from cogp.coordination.interfaces import builder_for
        from .protocol import SeedAddress
        address=SeedAddress(**cfg['seed_address'])
        ec=EnvConfig(**cfg['ec']);pc=PlannerConfig(**cfg['pc'])
        controller,executor=build_frozen_execution(cfg['collect'],cfg['drive'],pc,device='cpu')
        env=MultiDogHerdingEnv(ec)
        ob=GraphObservationBuilder(ec.max_sheep,ec.max_dogs,graph_radius=pc.graph_radius,
                                  dog_sheep_edge_radius=pc.dog_sheep_edge_radius)
        tb=builder_for(cfg['method'],ec)
        co=core.HCCGCommittedCoordinator(executor,pc,tb,gamma=pc.gamma,max_option_high_steps=64)
        ctx=snapshot=mode=train=None;episode=0;meta={};epd=eph=epcoll=partchanges=switches=0
        epr=0.;lastkey=None;first_connected=None;peak_lcc=0.;sum_collect=sum_tasks=sum_components=0.
        def physical_lcc():
            comps=core._components_from_mask(env.social_graph_mask())
            return max((len(x) for x in comps),default=0)/max(1,env.num_sheep)
        def capture():
            packet=TaskSnapshot.capture(env,tb,ob,ctx.phase_id)
            if cfg.get('actor_variant') is not None:
                from cogp.controls.actor import enrich_snapshot
                packet=enrich_snapshot(packet,env,co,ctx)
            return packet
        def reset_case(case):
            nonlocal ctx,snapshot,meta,epd,eph,epcoll,partchanges,switches,epr,lastkey,first_connected,peak_lcc,sum_collect,sum_tasks,sum_components
            m=int(case.get('dogs',2));n=int(case.get('sheep',15));sd=int(case['seed'])
            env.cfg.num_dogs=m;env.cfg.num_sheep=n
            if case['kind']=='canonical':env.reset(seed=sd,scenario=case['scenario'],num_dogs=m,num_sheep=n)
            elif case['kind']=='scaling':
                env.num_dogs=m;env.num_sheep=n;reset_scaling_case(env,sd,case['scenario'])
            else:raise ValueError('Only original canonical/scaling generators accepted')
            initial_hash=digest({'state':{k:v.tolist() for k,v in env.state().items()},'rng':env.rng.bit_generator.state})
            co.reset(m);ctx=co.prepare(env);snapshot=capture();st=tb.build_from_env(env)
            meta={**case,'profile':case['scenario'],'initial_hash':initial_hash,
                  'initial_components':len(core._components_from_mask(env.social_graph_mask())),
                  'initial_interface_components':len(st.components),'initial_tasks':len(st.collect_tasks)}
            epd=eph=epcoll=partchanges=switches=0;epr=0.;lastkey=None
            sum_collect=sum_tasks=sum_components=0.;peak_lcc=physical_lcc();first_connected=0 if peak_lcc>=1. else None
            return snapshot.encoded
        def reset_train():
            nonlocal episode
            sd=episode_seed(train['seed'],train['block'],cfg['wid'],episode,address) 
            rng=np.random.default_rng(sd+address.scenario_offset)
            sc=str(rng.choice(TRAIN_SCENARIOS))
            case={'kind':'canonical','scenario':sc,'seed':sd,'dogs':training_dogs(cfg['method'],train['block'],cfg['wid']),'sheep':15,
                  'case_id':f'train_b{train["block"]}_w{cfg["wid"]}_e{episode}','family':'train'}
            episode+=1;return reset_case(case)
        conn.send({'type':'ready','worker':cfg['wid'],'method':cfg['method']})
        while True:
            msg=conn.recv();cmd=msg['cmd']
            if cmd=='close':conn.send({'type':'closed'});break
            if cmd=='reset_train':
                mode='train';train=msg;episode=0
                conn.send({'type':'state','encoded':reset_train(),'meta':meta});continue
            if cmd=='reset_case':
                mode='eval';conn.send({'type':'state','encoded':reset_case(msg['case']),'meta':meta});continue
            if cmd!='step' or snapshot is None:raise RuntimeError(f'Invalid command/state: {cmd}')
            spec=msg['spec'];validate_spec(snapshot,env,tb,spec,core)
            before=tb.build_from_env(env);start=int(env.primitive_step);tick=time.perf_counter()
            key=tuple(t.key() for t in spec.tasks_by_dog)
            switches+=int(lastkey is not None and key!=lastkey);lastkey=key
            sum_collect+=spec.collect_count;sum_tasks+=len(before.collect_tasks);sum_components+=len(before.components)
            reward,done,info,duration,raw,pending=co.step_committed(env,ctx,spec)
            end=int(env.primitive_step);worker_seconds=time.perf_counter()-tick
            if duration<=0 or end<=start:raise RuntimeError('No environment time advancement')
            if not np.isfinite([reward,raw]).all() or not all(np.isfinite(x).all() for x in (env.sheep_pos,env.sheep_vel,env.dog_pos,env.dog_vel)):
                raise RuntimeError('Nonfinite state or reward')
            y,mask=co.collect_admissibility_labels(before,spec,env,int(info.get('task_collision_any',0)>0))
            epd+=1;eph+=duration;epr+=raw;epcoll=max(epcoll,int(info.get('task_collision_any',0)>0))
            pcg=int(info.get('task_partition_changed',0)>0);partchanges+=pcg
            lcc=physical_lcc();peak_lcc=max(peak_lcc,lcc)
            if first_connected is None and lcc>=1.:first_connected=end
            done=bool(done or end>=env.cfg.max_steps or env.metrics()['success']>.5)
            result=None
            if done:
                result={**meta,'success':int(env.metrics()['success']>.5),'primitive_steps':end,'decision_steps':epd,
                  'high_steps':eph,'episode_raw_reward':float(epr),'collision_any':epcoll,'partition_changes':partchanges,
                  'assignment_switches':switches,'mean_collect_count':sum_collect/epd,'mean_task_count':sum_tasks/epd,
                  'mean_interface_components':sum_components/epd,'mean_duration':eph/epd,
                  'first_connected_observed_step':first_connected,'max_lcc_observed':peak_lcc,'final_lcc':lcc,
                  'final_interface_lcc':float(tb.build_from_env(env).lcc_fraction),'final_outliers':int(env.outlier_mask().sum()),
                  'acquire_started':int(co.acquire_started_total),'acquire_completed':int(co.acquire_completed_total),
                  'reward_shaping_applied':False}
                if mode=='train':reset_train()
                else:snapshot=None
            else:ctx=pending if pending is not None else co.prepare(env);snapshot=capture()
            conn.send({'type':'transition','reward':float(reward),'base_reward':float(reward),
              'shaping_applied':0.,'reward_terms':{},'worker_step_seconds':worker_seconds,'raw_reward':float(raw),
              'duration':int(duration),'primitive_duration':end-start,'done':done,'y':y,'mask':mask,'episode':result,
              'next_encoded':snapshot.encoded if snapshot else None,'next_meta':dict(meta),'partition_changed':pcg})
    except BaseException as exc:
        try:conn.send({'type':'error','error':repr(exc),'traceback':traceback.format_exc()})
        except (OSError,EOFError):pass
    finally:
        try:conn.close()
        except OSError:pass

class Pool:
    def __init__(self,ec,pc,collect,drive,n_envs=8,timeout=900,method='HCCG',seed_address=None,actor_variant=None):
        self.n=n_envs;self.timeout=timeout;self.parents=[];self.procs=[];self.states=[None]*n_envs;self.metas=[None]*n_envs
        ctx=mp.get_context('spawn')
        try:
            for wid in range(n_envs):
                from dataclasses import asdict
                from .protocol import DEFAULT_ADDRESS
                a,b=ctx.Pipe();cfg={'seed_address':asdict(seed_address or DEFAULT_ADDRESS),'ec':ec.to_dict(),'pc':pc.to_dict(),
                  'collect':str(collect),'drive':str(drive),'wid':wid,'method':method,'actor_variant':actor_variant}
                p=ctx.Process(target=_worker,args=(b,cfg),daemon=True,name=f'mainline-env-{wid}')
                p.start();b.close();self.parents.append(a);self.procs.append(p)
            for wid,p in enumerate(self.procs):
                msg=self.recv(wid)
                if msg.get('type')!='ready':raise RuntimeError(msg)
                emit('ENV_READY',worker=wid,pid=p.pid,method=method)
        except BaseException:self.close();raise
    def send(self,i,msg):self.parents[i].send(msg)
    def recv(self,i):
        c=self.parents[i];deadline=time.monotonic()+self.timeout
        while not c.poll(min(1.,max(.01,deadline-time.monotonic()))):
            if not self.procs[i].is_alive():raise RuntimeError(f'Worker{i} exit={self.procs[i].exitcode}')
            if time.monotonic()>=deadline:raise TimeoutError(f'Worker{i} timeout{self.timeout}s')
        msg=c.recv()
        if msg.get('type')=='error':raise RuntimeError(f'Worker{i}: {msg["error"]}\n{msg["traceback"]}')
        return msg
    def ready(self,indices):
        from multiprocessing.connection import wait
        ids=list(indices);deadline=time.monotonic()+self.timeout
        while ids:
            for i in ids:
                if not self.procs[i].is_alive():raise RuntimeError(f'Worker{i} died: {self.procs[i].exitcode}')
            ready=wait([self.parents[i] for i in ids],timeout=1.)
            if ready:return [i for i in ids if self.parents[i] in ready]
            if time.monotonic()>deadline:raise TimeoutError('No worker reply')
        return []
    def reset_training(self,seed,block,mixture=None):
        import numpy as np
        self.action_rng=[np.random.default_rng(np.random.SeedSequence([seed,block,i,140])) for i in range(self.n)]
        for i in range(self.n):self.send(i,{'cmd':'reset_train','seed':seed,'block':block})
        for i in range(self.n):
            msg=self.recv(i);self.states[i]=msg['encoded'];self.metas[i]=msg['meta']
    def close(self):
        for c in self.parents:
            try:c.send({'cmd':'close'})
            except (OSError,EOFError,BrokenPipeError):pass
        deadline=time.monotonic()+10
        for p in self.procs:
            p.join(timeout=max(0,deadline-time.monotonic()))
            if p.is_alive():p.terminate();p.join(timeout=3)
            if p.is_alive():p.kill();p.join(timeout=3)
        for c in self.parents:
            try:c.close()
            except OSError:pass
        self.parents=[];self.procs=[]
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
