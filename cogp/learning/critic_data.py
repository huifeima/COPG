"""True-simulator labels for the two matched information critics.

Sampling uses frozen greedy HCCG. A paused prefix is labelled by resuming its
original commitment, then frozen greedy HCCG continuation to episode end. Both
input arms share each physical endpoint and the one Monte Carlo target.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import time

import numpy as np

from cogp.controls.critic import SCHEMA, REWARD_SCALE, encode_endpoint, endpoint_tail_return, write_label_archive
from cogp.io import atomic_json, digest, sha
from cogp.planning.branching import state_digest


def label_protocol(panel=None, *, decisions=None, horizons=None):
    source=(panel or {}).get('fit',{}).get('critic',{})
    p=dict(decisions=list(source.get('decision_indices',[0,2])), horizons=list(source.get('cut_horizons',[12,64])),
           E=int(source.get('original_commitment',64)),candidate_limit=4)
    if decisions is not None: p['decisions']=list(decisions)
    if horizons is not None: p['horizons']=list(horizons)
    if not p['decisions'] or p['decisions']!=sorted(set(p['decisions'])) or any(type(d) is not int or d<0 for d in p['decisions']):
        raise ValueError('Label decision indices must be increasing nonnegative integers')
    if not p['horizons'] or p['horizons']!=sorted(set(p['horizons'])) or any(type(h) is not int or h<1 or h>p['E'] for h in p['horizons']):
        raise ValueError('Label cuts must be increasing positive checks within the original commitment')
    return p


def make_label_jobs(output_root, *, study_hash, panel_name='matched_critics', config_dir=None, seeds=None):
    """Read exact recorded training/validation manifests; never create worlds."""
    from cogp.protocols import load_panel
    panel=load_panel(panel_name,config_dir)
    if not study_hash: raise ValueError('An immutable study hash is required')
    requested_seeds=list(panel['training_seeds'] if seeds is None else seeds)
    if not requested_seeds or any(seed not in panel['training_seeds'] for seed in requested_seeds) or len(set(requested_seeds))!=len(requested_seeds):
        raise ValueError('Label seeds must be unique members of the panel')
    root=Path(output_root).expanduser().resolve()
    jobs=[]; seen={}
    for split,key,count in [('train','training_manifest',32),('validation','validation_manifest',8)]:
        manifest=Path(panel['config_dir'])/panel['fit'][key]
        worlds=json.loads(manifest.read_text())
        if len(worlds)!=count: raise ValueError(f'Matched labels require {count} {split} worlds')
        world_keys={(w['kind'],w['scenario'],w['seed']) for w in worlds}
        if len(world_keys)!=len(worlds): raise ValueError('Duplicate matched-critic worlds')
        seen[split]=world_keys
        for case in worlds:
            _check_case(case)
            for seed in requested_seeds:
                jobs.append(dict(seed=seed,split=split,case=copy.deepcopy(case),out=str(root),study_hash=study_hash,
                                 protocol=label_protocol(panel),manifest_sha256=sha(manifest),panel=panel_name))
    if seen['train'] & seen['validation']: raise ValueError('Train/validation worlds overlap')
    return jobs


def _check_case(case):
    if case.get('kind')!='canonical' or case.get('dogs')!=2 or case.get('sheep')!=15:
        raise ValueError('Critic labels use canonical N15 M2 training worlds only')
    ident=case.get('case_id','')
    if not isinstance(ident,str) or not ident or Path(ident).name!=ident or ident in ('.','..'):
        raise ValueError('Unsafe label case ID')
    if type(case.get('seed')) is not int or case['seed']<0: raise ValueError('Invalid label world seed')


def terminal(session, done=False):
    env=session.env
    return bool(done or int(env.primitive_step)>=int(env.cfg.max_steps) or float(env.metrics().get('success',0.))>.5)


def _fresh_native(session):
    """Greedy native HCCG on the actual local endpoint, with no prepare call."""
    if hasattr(session.policy,'side_variant') or hasattr(session.policy,'family_name'):
        raise ValueError('Matched-critic labels require the unaugmented frozen HCCG parent')
    session.capture()
    return session.native()


def _selected(native_selector, session):
    value=native_selector(session)
    # Native API may return (assignment, diagnostics).
    if isinstance(value,tuple) and len(value)==2 and isinstance(value[1],dict): return value[0]
    return value


def _candidate_specs(candidate_builder,session,k):
    if candidate_builder is None:
        from cogp.planning.proposals import build_candidates
        supplied=build_candidates(session,K=k)
    else:
        import inspect
        parameters=inspect.signature(candidate_builder).parameters
        supplied=candidate_builder(session,K=k) if 'K' in parameters else candidate_builder(session,k=k)
    if hasattr(supplied,'specs'): supplied=supplied.specs
    if isinstance(supplied,tuple) and supplied and isinstance(supplied[0],(list,tuple)):
        supplied=supplied[0]
    specs=list(supplied)
    if not specs: raise RuntimeError('No feasible label candidates')
    if len(specs)>k: raise RuntimeError('Label candidate builder exceeded its fixed budget')
    from cogp.coordination.snapshot import task_key
    keys=[task_key(spec) for spec in specs]
    if len(set(keys))!=len(keys): raise RuntimeError('Duplicate label candidate assignment')
    return specs


def frozen_continuation(session, *, native_selector=None, done=False):
    """Scaled MC return from a genuine completed-commitment endpoint."""
    native_selector=native_selector or _fresh_native
    if terminal(session,done): return 0.,0,0
    total,discount,checks,options=0.,1.,0,0
    gamma=float(session.pc.gamma)
    guard=max(1,int(session.env.cfg.max_steps)-int(session.env.primitive_step)+1)
    for _ in range(guard):
        before=int(session.env.primitive_step)
        spec=_selected(native_selector,session)
        row=session.step(spec)
        duration=int(row['duration'])
        if duration<=0 or int(session.env.primitive_step)<=before: raise RuntimeError('Frozen continuation made no progress')
        reward=float(row['reward'])
        if not np.isfinite(reward): raise RuntimeError('Nonfinite continuation reward')
        total+=discount*REWARD_SCALE*reward
        discount*=gamma**duration
        checks+=duration; options+=1
        if terminal(session,row['done']): return float(total),checks,options
    raise RuntimeError('Frozen continuation did not terminate')


def collect_world(session, job, *, candidate_builder=None, native_selector=None):
    """Collect one atomically committed world; evaluation worlds are rejected.

    Session.reset(case), Session.step(spec), immutable TaskSnapshot capture and
    the resumable commitment adapter are the only runtime contracts. Builders
    can be explicitly injected to use the exact deployed candidate ordering.
    """
    from cogp.planning.commitment import start,advance,finish
    native_selector=native_selector or _fresh_native
    split=job['split']; case=copy.deepcopy(job['case']); seed=int(job['seed'])
    if split not in ('train','validation'): raise ValueError('Evaluation/test labels are forbidden')
    _check_case(case)
    if not 201<=seed<=208: raise ValueError('Matched label training seed outside the fixed panel')
    p=label_protocol(decisions=job.get('protocol',{}).get('decisions'),horizons=job.get('protocol',{}).get('horizons'))
    p.update(job.get('protocol',{}))
    if p['E']!=64 or p['candidate_limit']!=4:
        raise ValueError('Matched labels preserve original E64 and first-four candidate budget')
    if any(h>p['E'] for h in p['horizons']): raise ValueError('A label cut exceeds the original commitment')
    identity=dict(schema=SCHEMA,study_hash=job['study_hash'],seed=seed,split=split,case=case,
                  max_steps=int(session.env.cfg.max_steps),decisions=p['decisions'],horizons=p['horizons'],E=p['E'],
                  candidate_limit=p['candidate_limit'],reward_scale=REWARD_SCALE,
                  label_target='resume-same-commitment-then-frozen-greedy-HCCG-MC-to-terminal',
                  asset_identity=job.get('asset_identity',{}),manifest_sha256=job.get('manifest_sha256'))
    if not identity['study_hash']: raise ValueError('Matched labels require an immutable study hash')
    root=Path(job['out']).expanduser().resolve()
    data=root/'critic_labels'/f's{seed}'/split/(case['case_id']+'.npz')
    receipt=data.with_suffix('.json')
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if old.get('identity')!=identity or old.get('dataset_sha256')!=sha(data):
            raise RuntimeError('Existing label identity or archive checksum changed')
        return {**old,'reused':True,'path':str(data)}
    if hasattr(session.policy,'side_variant') or hasattr(session.policy,'family_name'):
        raise ValueError('Label driver must be the frozen default HCCG policy')
    if any(parameter.requires_grad for parameter in session.policy.parameters()) or session.policy.training:
        raise ValueError('Freeze the HCCG driver in evaluation mode before collecting labels')
    session.reset(case)
    session.co.max_option_high_steps=p['E']
    if abs(float(session.pc.gamma)-float(session.co.gamma))>1e-12: raise RuntimeError('Label/executor discount mismatch')
    if '_compute_sheep_forces' in vars(session.env):
        raise RuntimeError('Matched labels require unmodified true-simulator dynamics')
    initial_hash=state_digest(session.env,session.co,session.ctx)
    rows,metadata,missing,costs=[],[],[],[]
    begun=time.perf_counter(); decision=0; exhausted=terminal(session)
    for requested in p['decisions']:
        while decision<requested and not exhausted:
            spec=_selected(native_selector,session)
            row=session.step(spec)
            exhausted=terminal(session,row['done']); decision+=1
        if exhausted:
            missing.append(requested); continue
        # Capture current native state before candidate construction; never a
        # prepare() at an internal prediction pause.
        _selected(native_selector,session)
        specs=_candidate_specs(candidate_builder,session,p['candidate_limit'])
        before=state_digest(session.env,session.co,session.ctx)
        for candidate_index,spec in enumerate(specs):
            handle=start(session,spec,E=p['E'],predictor=None)
            prefixes=[]
            for horizon in p['horizons']:
                snap=advance(handle,horizon)
                feature,schema=encode_endpoint(session,snap)
                if schema!='execution-state': raise RuntimeError('Changed label execution schema')
                prefixes.append((horizon,feature,float(snap.reward),int(snap.duration),bool(snap.done),bool(snap.active)))
            full=finish(handle)
            continuation_start=int(full.session.env.primitive_step)
            continuation,checks,options=frozen_continuation(full.session,native_selector=native_selector,done=full.done)
            costs.append(dict(decision=requested,candidate=candidate_index,first_commitment_checks=int(full.duration),
                              first_commitment_primitives=int(full.primitive_steps),continuation_checks=checks,
                              continuation_primitives=int(full.session.env.primitive_step)-continuation_start,continuation_options=options))
            for horizon,feature,reward,duration,done,active in prefixes:
                y=endpoint_tail_return(reward,duration,full.reward,full.duration,continuation,float(session.pc.gamma),done)
                rows.append({**feature,'target':np.float64(y),'horizon':np.int64(horizon),'decision':np.int64(requested),
                             'candidate':np.int64(candidate_index),'actual_checks':np.int64(duration),
                             'active_commitment':np.int8(active),'terminal':np.int8(done)})
                metadata.append(dict(horizon=horizon,decision=requested,candidate=candidate_index,requested_E=p['E'],
                                     prefix_checks=duration,full_checks=int(full.duration),active_commitment=active,terminal=done,
                                     prefix_unscaled_reward=reward,full_unscaled_reward=float(full.reward),continuation_scaled_return=continuation,
                                     continuation_checks=checks,continuation_options=options,target=y,gamma=float(session.pc.gamma)))
            if state_digest(session.env,session.co,session.ctx)!=before:
                raise RuntimeError('Label candidate branch mutated the live sampling trajectory')
    if not rows: raise RuntimeError('No critic labels were produced from the declared sampling states')
    result=write_label_archive(root,identity,rows,metadata=metadata)
    result.update(path=str(data),initial_hash=initial_hash,missing_terminal_decisions=missing,
                  collection_seconds=time.perf_counter()-begun,sampling_primitive_steps=int(session.env.primitive_step),
                  branch_costs=costs,additional_simulated_primitive_steps=int(session.env.primitive_step)+sum(
                      item['first_commitment_primitives']+item['continuation_primitives'] for item in costs))
    atomic_json(receipt,{key:value for key,value in result.items() if key!='reused'})
    return result


def collect_panel(registry, worlds, out, seed, device='cpu', *, decisions=(0,2), horizons=(12,64),
                  study_hash=None, split=None, session_factory=None, candidate_builder=None, native_selector=None):
    """CLI adapter for exact manifests or explicitly split custom training worlds.

    worlds may be None (the shipped 32/8 manifests), a JSON mapping containing
    `train` and `validation`, or an explicit list with `split` per case. A single
    split list must pass split='train' or split='validation'. Evaluation manifests
    without a declared training split are never silently converted into labels.
    """
    if session_factory is None:
        from cogp.runtime import Session
        session_factory=lambda:Session.from_registry(registry,int(seed),dict(generator='hccg',planner=False,critic='original',E=64,actor='default'),device)
    session=session_factory()
    used_assets=copy.deepcopy(getattr(session,'asset_identity',{}))
    if not used_assets:
        required=('basc.collect','basc.drive',f'parent.{int(seed)}.HCCG')
        used_assets={key:sha(registry.resolve(key)) for key in required}
    if study_hash is None:
        study_hash=digest(dict(protocol='matched-critic-supervision',seed=int(seed),assets=used_assets,
                               decisions=list(decisions),horizons=list(horizons),collector_sha256=sha(__file__)))
    if worlds is None:
        jobs=make_label_jobs(out,study_hash=study_hash,seeds=[int(seed)])
    else:
        manifest_path=Path(worlds).expanduser().resolve() if isinstance(worlds,(str,Path)) else None
        content=json.loads(manifest_path.read_text()) if manifest_path else copy.deepcopy(worlds)
        if isinstance(content,dict):
            cases=[(key,case) for key in ('train','validation') for case in content.get(key,[])]
        elif isinstance(content,list):
            cases=[(split or case.get('split'),case) for case in content]
        else: raise ValueError('Unsupported critic training-world manifest')
        if not cases or any(key not in ('train','validation') for key,case in cases):
            raise ValueError('Custom label worlds require explicit train/validation splits')
        train_keys={(case['kind'],case['scenario'],case['seed']) for key,case in cases if key=='train'}
        validation_keys={(case['kind'],case['scenario'],case['seed']) for key,case in cases if key=='validation'}
        if train_keys & validation_keys: raise ValueError('Custom label training and validation worlds overlap')
        jobs=[dict(seed=int(seed),split=key,case=case,out=str(Path(out).resolve()),study_hash=study_hash,
                   protocol=label_protocol(decisions=decisions,horizons=horizons),
                   manifest_sha256=sha(manifest_path) if manifest_path else digest(content)) for key,case in cases]
    for job in jobs:
        job['protocol']=label_protocol(decisions=decisions,horizons=horizons)
        job['asset_identity']=copy.deepcopy(registry.entries)
    for job in jobs: job['asset_identity']=copy.deepcopy(used_assets)
    results=[]
    for job in jobs:
        results.append(collect_world(session,job,candidate_builder=candidate_builder,native_selector=native_selector))
    report=dict(schema='matched-critic-label-panel',study_hash=study_hash,seed=int(seed),worlds=len(jobs),
                completed=sum(not item.get('reused',False) for item in results),reused=sum(bool(item.get('reused')) for item in results),
                train_worlds=sum(job['split']=='train' for job in jobs),validation_worlds=sum(job['split']=='validation' for job in jobs),
                samples=sum(item['samples'] for item in results),receipts=[item['path'] for item in results],
                label_target='original-commitment-completion-then-frozen-greedy-HCCG', planner_aware=False)
    atomic_json(Path(out)/'critic_labels'/f's{seed}'/'panel.json',report)
    return report
