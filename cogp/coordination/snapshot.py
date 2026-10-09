"""Immutable, hash-verified snapshots of authoritative task construction."""
from __future__ import annotations
import dataclasses, hashlib, json
import numpy as np
import torch
from cogp.learning.parent.io import plain, require
SCHEMA='authoritative-task-snapshot'
def canonical(obj):
    return json.dumps(plain(obj),sort_keys=True,separators=(',',':'),allow_nan=False)


def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def physical_stamp(env):
    h=hashlib.sha256()
    for k in ('sheep_pos','sheep_vel','dog_pos','dog_vel','goal'):
        a=np.asarray(getattr(env,k));h.update(k.encode());h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode());h.update(a.tobytes())
    h.update(canonical([int(env.num_sheep),int(env.num_dogs),int(env.primitive_step),
                        int(env.high_step),int(env.success_hold),env.cfg.to_dict()]).encode())
    return h.hexdigest()


class AuthoritativeTaskBuilder:
    """Cache the original physical-environment construction at an unchanged state.

    Global-state-only task reconstruction deliberately fails. A caller must
    replay the stored TaskSnapshot instead of silently rebuilding its targets.
    """
    def __init__(self, original):
        self.original=original; self._stamp=None; self._state=None; self.build_count=0

    def __getattr__(self,key):
        # deepcopy probes __setstate__ before the copied __dict__ exists.
        # Avoid recursive lookup of an as-yet uninitialized original wrapper.
        try: original=object.__getattribute__(self,'original')
        except AttributeError: raise AttributeError(key) from None
        return getattr(original,key)

    def build_from_env(self,env):
        s=physical_stamp(env)
        if s!=self._stamp:
            self._state=self.original.build_from_env(env)
            self._stamp=s;self.build_count+=1
        return self._state

    def build_from_global(self,*args,**kwargs):
        raise RuntimeError('Explicit task snapshot required: do not reconstruct task IDs from normalized observations.')


def structure_payload(st):
    return {'active_sheep':int(st.active_sheep),'active_dogs':int(st.active_dogs),
            'components':[list(c) for c in st.components],
            'main_component':list(st.main_component),
            'social_graph':np.asarray(st.social_graph,dtype=bool).tolist(),
            'collect_tasks':[plain(dataclasses.asdict(t)) for t in st.collect_tasks],
            'center':np.asarray(st.center,dtype=np.float32).tolist(),
            'axis':np.asarray(st.axis,dtype=np.float32).tolist(),
            'lateral':np.asarray(st.lateral,dtype=np.float32).tolist()}


def structure_from_payload(p,core):
    ts=[]
    for d in p['collect_tasks']:
        d=dict(d);d['component_members']=tuple(d['component_members'])
        ts.append(core.PressureTask(**d))
    return core.HCCGStructure(
        active_sheep=int(p['active_sheep']),active_dogs=int(p['active_dogs']),
        components=tuple(tuple(c) for c in p['components']),
        main_component=tuple(p['main_component']),
        social_graph=np.asarray(p['social_graph'],dtype=bool),collect_tasks=tuple(ts),
        center=np.asarray(p['center'],np.float32),axis=np.asarray(p['axis'],np.float32),
        lateral=np.asarray(p['lateral'],np.float32))


@dataclasses.dataclass(frozen=True)
class TaskSnapshot:
    """Immutable JSON packet: observation + phase + exact task/graph metadata.

    task_frame_units is normalized by arena width. Tasks' pressure_lateral
    and all discrete IDs are copied from the physical-environment builder.
    Sheep IDs are retained; this packet does NOT promise sheep-ID equivariance.
    """
    encoded: str

    @classmethod
    def capture(cls,env,builder,ob,phase_id):
        before=physical_stamp(env);st=builder.build_from_env(env)
        p=structure_payload(st)
        p['center']=(np.asarray(st.center,np.float32)/float(env.cfg.world_size)).tolist()
        g=np.asarray(ob.global_state_vector(env),np.float32)
        require(np.isfinite(g).all(),'Nonfinite snapshot observation')
        payload={'schema':SCHEMA,'physical_stamp':before,'phase_id':int(phase_id),
                 'primitive_step':int(env.primitive_step),'world_size':float(env.cfg.world_size),
                 'max_sheep':int(ob.max_sheep),'max_dogs':int(ob.max_dogs),
                 'global_state':g.tolist(),'structure':p,
                 'task_source':'environment_float32_physics_once',
                 'task_frame_units':'normalized_by_world_size'}
        require(before==physical_stamp(env),'Snapshot construction mutated environment')
        return cls(canonical(payload))

    @property
    def data(self): return json.loads(self.encoded)
    @property
    def sha256(self): return hashlib.sha256(self.encoded.encode()).hexdigest()

    def tensors(self,core,device='cpu'):
        p=self.data;require(p['schema']==SCHEMA,'Snapshot schema mismatch')
        st=structure_from_payload(p['structure'],core)
        g=torch.as_tensor(p['global_state'],dtype=torch.float32,device=device)
        require(g.numel()==5*p['max_sheep']+5*p['max_dogs']+6,'Wrong snapshot observation length')
        return g,st,int(p['phase_id'])

    def save_record(self): return {'sha256':self.sha256,'payload':self.data}

    @classmethod
    def load_record(cls,r):
        snap=cls(canonical(r['payload']))
        require(snap.sha256==r['sha256'],'Task snapshot digest mismatch')
        return snap


def task_key(spec): return tuple(t.key() for t in spec.tasks_by_dog)


def candidate_digest(specs): return digest([task_key(s) for s in specs])


def validate_spec(snapshot,env,builder,spec,core):
    """Bind a selected task to the same immutable snapshot before execution."""
    require(snapshot.data['physical_stamp']==physical_stamp(env),'Stale snapshot used for execution')
    _,st,phase=snapshot.tensors(core)
    raw=builder.build_from_env(env)
    require(tuple(raw.components)==tuple(st.components),'Partition differs from authoritative snapshot')
    require([dataclasses.asdict(t) for t in raw.collect_tasks]==
            [dataclasses.asdict(t) for t in st.collect_tasks], 'Task metadata differs from snapshot')
    require(core.HCCGActionSpace.is_valid(spec.tasks_by_dog,st,phase),'Invalid assignment')
    available={t.key():t for t in st.collect_tasks}
    for t in spec.tasks_by_dog:
        if t.task_type==core.TASK_COLLECT:
            require(t.key() in available and t==available[t.key()], 'Selected Collect not in snapshot')


def permute_global(g,S,D,p):
    require(sorted(p)==list(range(len(p))),'Invalid dog permutation')
    require(g.ndim==1 and g.numel()==5*S+5*D+6,'Observation layout mismatch')
    M=len(p);s=5*S;e=s+4*D;q=g.clone()
    require(bool((g[e:e+M]>.5).all()) and not bool((g[e+M:e+D]>.5).any()),'Active dog mask mismatch')
    q[s:e].view(D,4)[:M]=g[s:e].view(D,4)[list(p)]
    q[e:e+M]=g[e:e+M][list(p)]
    return q

