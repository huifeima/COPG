"""Real cross-state batching of learned layers; immutable CPU geometry cached.

Core parameter names and shapes are preserved. The batch implementation does
not promise bitwise arithmetic equality. Mandatory reference score/gradient
checks guard the FP32 batching change. Input gradients are not part of PPO.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn.functional as F
from .ranking import prepare,pair_scale_factor

@torch.no_grad()
def features(policy,state):
    if hasattr(state,'cpu_features'):return state.cpu_features
    from . import core
    st=state.structure;g=torch.from_numpy(state.global_np)
    sheep,sm,dogs,dm,goal,extras,n,m,sp,dp,center,axis,lat=policy._active_tensors(g)
    if n<1 or m<1:raise ValueError('Empty flock/team')
    C=len(st.components);pool=(list(st.collect_tasks) if state.phase==core.PHASE_RECOVER else [])+policy.space.support_pool(m)
    K=len(pool);index={t.key():i for i,t in enumerate(pool)}
    cand=torch.zeros(n)
    for t in st.collect_tasks:cand[list(t.component_members)]=1.
    membership=torch.zeros(C,n);compraw=[];cp=[];sizes=[]
    main=sp[list(st.main_component)].mean(0) if st.main_component else center
    for ci,comp in enumerate(st.components):
        ii=list(comp);membership[ci,ii]=1.
        pos=sp[ii].mean(0);cv=sheep[ii,2:4].mean(0);rr=torch.linalg.vector_norm(sp[ii]-pos,dim=-1)
        raw=torch.cat([torch.tensor([len(ii)/n,math.log1p(len(ii))/math.log1p(max(2,n)),float(ci==0)]),
              policy._edge_triplet_t(center,pos,axis,lat),policy._edge_triplet_t(main,pos,axis,lat),
              cv,torch.stack([rr.mean(),rr.max()])])[:core.COMP_RAW_DIM]
        compraw.append(raw);cp.append(pos);sizes.append(len(ii))
    cp=torch.stack(cp);size=torch.tensor(sizes,dtype=torch.float32)
    same=torch.zeros(C,C,1);same[:,0,0]=1.
    ce=torch.cat([policy._edge_triplet_t(cp[:,None,:],cp[None,:,:],axis,lat),
                 torch.log1p(size[None,:]/size[:,None]).unsqueeze(-1),same],-1)
    raw=torch.stack([policy._task_raw(t,st,sheep,center,axis,lat) for t in pool])
    pos=torch.stack([policy._task_position(t,st,sheep,center,axis,lat) for t in pool])
    taskci=torch.tensor([policy._component_index(st,t) for t in pool],dtype=torch.long)
    dogc=policy._edge_triplet_t(center,dp,axis,lat)
    taskc=policy._edge_triplet_t(center,pos,axis,lat)
    td=policy._edge_triplet_t(dp[:,None,:],pos[None,:,:],axis,lat)
    extra=torch.stack([torch.stack([raw[k,11],raw[k,13],raw[k,10]]) if t.task_type==core.TASK_COLLECT else torch.zeros(3) for k,t in enumerate(pool)])
    geom=torch.cat([dogc[:,None,:].expand(m,K,3),taskc[None,:,:].expand(m,K,3),td,extra[None,:,:].expand(m,K,3)],-1)
    te=policy._edge_triplet_t(pos[:,None,:],pos[None,:,:],axis,lat)
    flags=torch.tensor([[[float(a.task_type==b.task_type==core.TASK_COLLECT and a.component_members==b.component_members),
                        float(a.task_type==b.task_type==core.TASK_COLLECT),float(a.task_type==b.task_type)] for b in pool] for a in pool])
    pairgeom=[]
    for i in range(m):
        for j in range(i+1,m):
            dg=policy._edge_triplet_t(dp[i],dp[j],axis,lat);dr=policy._edge_triplet_t(dp[j],dp[i],axis,lat)
            pairgeom.append(torch.stack([torch.cat([dg.expand(K,K,3),te,flags,te[...,2:3]],-1),
                  torch.cat([dr.expand(K,K,3),te.transpose(0,1),flags.transpose(0,1),te[...,2:3]],-1)],2))
    result={'n':n,'m':m,'C':C,'K':K,'A':len(state.specs),
      'sheep':torch.cat([sheep[:n],cand[:,None]],-1),'dogs':dogs[:m],
      'ssadj':torch.tensor(st.social_graph,dtype=torch.float32),'ssedge':policy._edge_triplet_t(sp[:,None,:],sp[None,:,:],axis,lat),
      'membership':membership,'sizes':size,'compraw':torch.stack(compraw),'ce':ce,
      'd2c':policy._edge_triplet_t(dp[:,None,:],cp[None,:,:],axis,lat),
      'context_static':torch.cat([goal,extras,torch.tensor([C/n,math.log1p(m)/math.log(7.)])]),
      'taskraw':raw,'taskci':taskci,'geom':geom,'pairgeom':torch.stack(pairgeom) if pairgeom else torch.zeros((0,K,K,2,10)),
      'ix':torch.tensor([[index[t.key()] for t in a.tasks_by_dog] for a in state.specs],dtype=torch.long)}
    state.cpu_features=result;return result

def _message(layer,h,adj,edge):
    B,N,H=h.shape
    src=h[:,None,:,:].expand(B,N,N,H)
    msg=layer.msg(torch.cat([src,edge],-1));w=adj.unsqueeze(-1)
    deg=w.sum(2);agg=(msg*w).sum(2)/deg.clamp_min(1).sqrt()
    return layer.norm(h+layer.upd(torch.cat([h,agg,deg.log1p()],-1)))

def batch_forward(policy,states):
    if not states:raise ValueError('Empty state list')
    from cogp.learning.parent.protocol import TRAIN_BATCH_CANDIDATE_LIMIT
    groups={}
    for i,st in enumerate(states):
        key=(st.structure.active_sheep,st.structure.active_dogs)
        groups.setdefault(key,[]).append((i,st))
    outputs={};maxa=max(len(st.specs) for st in states);maxm=max(st.structure.active_dogs for st in states)
    forward=external_batch_forward if hasattr(policy,'family_name') else hccg_batch_forward
    for group in groups.values():
        # The candidate budget bounds activation memory, not the action set.
        pos=0
        while pos<len(group):
            end=pos+1;largest=len(group[pos][1].specs)
            while end<len(group) and max(largest,len(group[end][1].specs))*(end-pos+1)<=TRAIN_BATCH_CANDIDATE_LIMIT:
                largest=max(largest,len(group[end][1].specs));end+=1
            chunk=group[pos:end]
            scores,adm,valid=forward(policy,[st for _,st in chunk])
            for j,(index,st) in enumerate(chunk):
                a=len(st.specs);m=st.structure.active_dogs
                outputs[index]=(F.pad(scores[j,:a],(0,maxa-a),value=-1e9),
                    F.pad(adm[j,:a,:m],(0,maxm-m,0,maxa-a)),
                    F.pad(valid[j,:a],(0,maxa-a),value=False))
            pos=end
    return tuple(torch.stack([outputs[i][k] for i in range(len(states))]) for k in range(3))

def _hccg_batch_forward(policy,states):
    fs=[features(policy,s) for s in states]
    if not fs:raise ValueError('Empty batch')
    ns={(f['n'],f['m']) for f in fs}
    if len(ns)!=1:raise ValueError('A batched call must share active sheep/team size')
    n,m=next(iter(ns));B=len(fs);C=max(f['C'] for f in fs);K=max(f['K'] for f in fs);A=max(f['A'] for f in fs)
    if A==0:raise ValueError('Batched forward requires a nonempty prepared feasible support')
    device=next(policy.parameters()).device
    def stack(key):return torch.stack([f[key] for f in fs]).to(device)
    def padded(key,shape,fill=0,dtype=torch.float32):
        z=torch.full((B,*shape),fill,dtype=dtype)
        for b,f in enumerate(fs):
            x=f[key];z[(b,)+tuple(slice(0,t) for t in x.shape)]=x
        return z.to(device)
    count=torch.tensor([f['C'] for f in fs],device=device,dtype=torch.float32)
    cm=(torch.arange(C,device=device)[None,:]<count[:,None]).to(torch.float32)
    member=padded('membership',(C,n));sizes=padded('sizes',(C,),fill=1.)
    sh=policy.sheep_phi(stack('sheep'));dh=policy.dog_phi(stack('dogs'))
    if policy.use_component_messages:
        ssadj=stack('ssadj');ssedge=stack('ssedge')
        for layer in policy.ss_layers:sh=_message(layer,sh,ssadj,ssedge)
    sums=member@sh
    ch=policy.comp_init(torch.cat([sums/sizes[...,None],sums/sizes.sqrt()[...,None],padded('compraw',(C,12))],-1))*cm[...,None]
    if policy.use_component_messages:
        adj=cm[:,:,None]*cm[:,None,:]*(1-torch.eye(C,device=device)[None,:,:]);ce=padded('ce',(C,C,5))
        for layer in policy.comp_layers:ch=_message(layer,ch,adj,ce)*cm[...,None]
        d2c=padded('d2c',(m,C,3))
        msg=policy.dog_comp_msg(torch.cat([ch[:,None,:,:].expand(B,m,C,-1),d2c],-1))*cm[:,None,:,None]
        agg=msg.sum(2)/count.sqrt()[:,None,None]
        dh=policy.dog_norm(dh+policy.dog_update(torch.cat([dh,agg,count.log1p()[:,None,None].expand(B,m,1)],-1)))
    ctx=policy.context_net(torch.cat([sh.mean(1),sh.sum(1)/math.sqrt(n),dh.mean(1),
                       ch.sum(1)/count[:,None],ch.sum(1)/count.sqrt()[:,None],stack('context_static')],-1))
    tci=padded('taskci',(K,),fill=-1,dtype=torch.long)
    taskcomp=ch[torch.arange(B,device=device)[:,None],tci.clamp_min(0)]*(tci>=0)[...,None]
    th=policy.task_net(torch.cat([taskcomp,padded('taskraw',(K,16))],-1))
    x=torch.cat([ctx[:,None,None,:].expand(B,m,K,-1),dh[:,:,None,:].expand(B,m,K,-1),
                 th[:,None,:,:].expand(B,m,K,-1),padded('geom',(m,K,12))],-1)
    assignment_inputs=x
    adm=policy.admissibility_net(x).squeeze(-1)
    u=policy.unary_net(x).squeeze(-1)+policy.admissibility_score_weight*F.logsigmoid(adm)
    if hasattr(policy,'competition_head') and getattr(policy,'residual_scale',1.)!=0:
        taskmask=torch.arange(K,device=device)[None,:]<torch.tensor([f['K'] for f in fs],device=device)[:,None]
        u=u+float(getattr(policy,'residual_scale',1.))*policy.competition_head(x,taskmask)
    ix=padded('ix',(A,m),dtype=torch.long);ar=torch.arange(B,device=device)[:,None,None];dogs=torch.arange(m,device=device)[None,None,:]
    scores=u[ar,dogs,ix].mean(-1);selected_adm=adm[ar,dogs,ix]
    if m>1 and policy.pairwise_weight!=0:
        pg=padded('pairgeom',(m*(m-1)//2,K,K,2,10));pairs=[];pidx=0
        hi=th[:,:,None,:].expand(B,K,K,-1);hj=th[:,None,:,:].expand(B,K,K,-1)
        for i in range(m):
            for j in range(i+1,m):
                x=torch.cat([dh[:,i,None,None,:].expand(B,K,K,-1),dh[:,j,None,None,:].expand(B,K,K,-1),hi,hj,pg[:,pidx,:,:,0]],-1)
                xr=torch.cat([dh[:,j,None,None,:].expand(B,K,K,-1),dh[:,i,None,None,:].expand(B,K,K,-1),hj,hi,pg[:,pidx,:,:,1]],-1)
                pair=policy.pair_net(torch.stack([x,xr],-2)).squeeze(-1).mean(-1)
                pairs.append(pair[torch.arange(B,device=device)[:,None],ix[:,:,i],ix[:,:,j]]);pidx+=1
        scores=scores+policy.pairwise_weight*pair_scale_factor(policy,m)*torch.stack(pairs).mean(0)
    valid=torch.arange(A,device=device)[None,:]<torch.tensor([f['A'] for f in fs],device=device)[:,None]
    return scores.masked_fill(~valid,-1e9),selected_adm,valid


def external_batch_forward(policy,states):
    """Differentiable native external encoding + batched candidates per state.

    External variable line graphs are evaluated state-by-state (not advertised
    as cross-state batched). Loss gradients still accumulate over1024 records.
    """
    from .external_scores import ExternalScores
    if not states:raise ValueError('Empty batch')
    device=next(policy.parameters()).device;A=max(len(s.specs) for s in states)
    scores=[];adms=[];m=states[0].structure.active_dogs
    for s in states:
        if s.structure.active_dogs!=m or not s.specs:raise ValueError('Invalid external batch')
        tab=ExternalScores(policy,torch.as_tensor(s.global_np,device=device),s.structure,s.phase)
        ix=torch.tensor(tab.indices(s.specs),dtype=torch.long,device=device)
        y=tab.score_indices(ix);a=tab.state.edge_admiss[torch.arange(m,device=device)[None,:],ix]
        scores.append(F.pad(y,(0,A-len(y)),value=-1e9))
        adms.append(F.pad(a,(0,0,0,A-len(y))))
    valid=torch.arange(A,device=device)[None,:]<torch.tensor([len(s.specs) for s in states],device=device)[:,None]
    return torch.stack(scores),torch.stack(adms),valid


def hccg_batch_forward(policy,states):
    # The optional matched actor adds immutable raw execution features. This
    # context is per call and never patches global functions or stores learned
    # tensors across PPO updates.
    if getattr(policy,'side_variant',None) is None:
        return _hccg_batch_forward(policy,states)
    from cogp.controls.actor import actor_batch_context
    with actor_batch_context(policy,states):
        return _hccg_batch_forward(policy,states)
