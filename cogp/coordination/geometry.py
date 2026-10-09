"""State-only geometric proposal, using ORIGINAL constraints and anchor semantics.

This is not Hungarian matching (coverage/formation constraints are retained),
not an optimal multi-step controller, and finite beam search is not exact.
The resulting training support is deterministic and independent of parameters.
"""
from __future__ import annotations
import numpy as np
import torch

def geometry_table(state):
    from . import core
    st=state.structure;M=st.active_dogs
    g=torch.from_numpy(state.global_np)
    sheep,sm,dogs,dm,goal,extras=core._parse_global_torch(g,int(state.snapshot.data['max_sheep']),int(state.snapshot.data['max_dogs']))
    center=sheep[:st.active_sheep,:2].mean(0)
    axis=goal-center;axis=axis/torch.linalg.vector_norm(axis).clamp_min(1e-6)
    lat=torch.stack([-axis[1],axis[0]])
    pool=(list(st.collect_tasks) if state.phase==core.PHASE_RECOVER else [])+core.HCCGActionSpace.support_pool(M)
    positions=torch.stack([core.HCCGPolicy._task_position(None,t,st,sheep,center,axis,lat) for t in pool])
    cost=torch.linalg.vector_norm(dogs[:M,None,:2]-positions[None,:,:],dim=-1).numpy().astype(np.float64)
    return pool,cost

def geometry_candidates(state,width=256):
    from . import core
    if width<1:raise ValueError('width must be positive')
    pool,cost=geometry_table(state);M=state.structure.active_dogs
    space=core.HCCGActionSpace(None)
    # Stable score then lexical task-index order; no random permutations.
    beam=[(0.,())];expanded=0;truncated=False
    for i in range(M):
        nxt=[]
        for value,ks in beam:
            for k in range(len(pool)):
                if k in ks:continue
                ix=ks+(k,);tasks=tuple(pool[q] for q in ix)
                if space.is_prefix_extendable(tasks,state.structure,state.phase):
                    nxt.append((value+float(cost[i,k]),ix))
        if not nxt:raise RuntimeError('No extendable geometric prefix; no fallback')
        expanded+=len(nxt);nxt.sort(key=lambda v:(v[0],v[1]))
        truncated|=len(nxt)>width;beam=nxt[:width]
    specs=[core.HCCGAssignmentSpec(tuple(pool[k] for k in ks)) for _,ks in beam
           if space.is_valid(tuple(pool[k] for k in ks),state.structure,state.phase)]
    if not specs:raise RuntimeError('No feasible geometric terminal assignment')
    return specs,dict(search=f'geometry_beam{width}',candidates=len(specs),expanded=expanded,
                      truncated=bool(truncated),parameter_independent=True)
