"""True-simulator reference diagnostics; these are never deployed policy choices."""
from __future__ import annotations
import copy
import csv
import json
from pathlib import Path
import time
import numpy as np
from cogp.runtime import Session
from cogp.protocols import load_panel
from cogp.planning.branching import rollout_branch,clone_branch,Rollout,state_digest
from cogp.planning.proposals import build_candidates,CandidateSet
from cogp.planning.planner import Planner,score_components
from cogp.coordination.snapshot import task_key


def continuation(session,kind='hccg_direct'):
    """Scaled MC return from this endpoint under the explicitly named policy."""
    result=0.;discount=1.;decisions=0
    planner=Planner(session) if kind=='hccg_cogp' else None
    if kind not in ('hccg_direct','hccg_cogp'):raise ValueError('Unknown reference continuation')
    if session.method!='HCCG':raise ValueError('Reference continuation must use the frozen HCCG actor')
    while not session.done:
        session.capture()
        spec=planner.select().spec if planner else session.native()
        step=session.step(spec)
        result+=discount*.1*step['reward'];discount*=session.pc.gamma**step['duration'];decisions+=1
    return dict(return_value=result,decisions=decisions,success=int(session.env.metrics()['success']>.5))


def reference(session,spec,continuation_template,kind='hccg_direct',E=64):
    """True first commitment plus specified greedy continuation, not optimal Q."""
    first=rollout_branch(session.env,session.co,session.ctx,spec,None,E)
    local=continuation_template.clone_view(first)
    follow=continuation(local,kind) if not first.done else dict(return_value=0.,decisions=0,success=int(first.env.metrics()['success']>.5))
    total=.1*first.reward+session.pc.gamma**first.duration*follow['return_value']
    return dict(reference_return=total,true_reward=first.reward,true_duration=first.duration,true_terminal=first.done,
                continuation=kind,continuation_return=follow['return_value'],success=follow['success']),first


def _state_session(template,source):
    env,co,ctx=clone_branch(source.env,source.co,source.ctx)
    return template.clone_view(Rollout(env,co,ctx,0.,0.,0,0,source.done,{}))


def _driver_states(registry,seed,case,panel,device):
    base=dict(generator='hccg',planner=False,dynamics='graph',critic='original',H=64,E=64,K=4,actor='default')
    if panel['name']=='cross_generator':
        initial=Session.from_registry(registry,seed,base,device);initial.reset(case)
        yield 'initial',initial
        for gen in ('hccg','edge_gnn','agent_dag'):
            driver=Session.from_registry(registry,seed,{**base,'generator':gen},device);driver.reset(case)
            for decision in range(2):
                if driver.done:break
                driver.step(driver.native())
            if not driver.done:yield gen+'_decision2',driver
    else:
        uses_planner=panel.get('driver')=='hccg_cogp' or panel.get('common_state_driver')=='full'
        driver=Session.from_registry(registry,seed,{**base,'planner':uses_planner},device);driver.reset(case)
        planner=Planner(driver) if uses_planner else None
        indices=panel.get('decision_indices',panel.get('common_decision_indices',[0,2]))
        for decision in range(max(indices)+1):
            if driver.done:break
            if decision in indices:
                # The caller mutates isolated snapshots, never this driver.
                yield f'decision{decision}',driver
            driver.step(planner.select().spec if planner else driver.native())


def _ranking(reference_returns,scores,native_index=0):
    reference_returns=np.asarray(reference_returns,float);scores=np.asarray(scores,float)
    selected=int(scores.argmax());best=float(reference_returns.max())
    return dict(selected_index=selected,regret=best-float(reference_returns[selected]),
        best_agreement=int(np.isclose(reference_returns[selected],best,rtol=0,atol=1e-10)),
        native_gain=float(reference_returns[selected]-reference_returns[native_index]),best_of_K=best,
        selected_reference=float(reference_returns[selected]),native_reference=float(reference_returns[native_index]))


def run_job(job,registry,device='cpu'):
    if job['panel']=='cross_generator':return run_cross_job(job,registry,device)
    if job['panel']=='memory_states':return memory_common_states(job,registry,device)
    panel=load_panel(job['panel']);arm=copy.deepcopy(job['arm_config']);true_only=arm.get('dynamics')=='true_diagnostic_only'
    if true_only:arm['dynamics']='graph'
    template=Session.from_registry(registry,job['seed'],arm,device)
    continuation_template=Session.from_registry(registry,job['seed'],{**arm,'generator':'hccg','critic':'original'},device)
    states=[];cpu=time.process_time()
    for label,source in _driver_states(registry,job['seed'],job['case'],panel,device):
        state_id=f"s{job['seed']}/{job['case']['case_id']}/{label}"
        local=_state_session(template,source)
        sets={'native_hybrid':build_candidates(local,arm['K'],arm.get('proposal','native'))}
        if panel['name']=='cross_generator':
            allsets=[]
            for generator in ('hccg','edge_gnn','agent_dag','geometry'):
                other=Session.from_registry(registry,job['seed'],{**arm,'generator':generator},device)
                other=_state_session(other,source);allsets.append(build_candidates(other,arm['K']))
            matched=min(len(s.specs) for s in allsets)
            sets['native_hybrid'].specs=sets['native_hybrid'].specs[:matched];sets['native_hybrid'].scores=sets['native_hybrid'].scores[:matched]
            # Shared geometric support, ranked by each generator; no private representation anchors.
            from cogp.coordination.geometry import geometry_candidates
            from cogp.coordination.ranking import prepare
            pool=geometry_candidates(prepare(local.snapshot),width=256)[0]
            geometry=allsets[-1]
            if local.method=='Geometry':specs=pool[:matched];scores=geometry.scores[:matched]
            else:
                from cogp.coordination.ranking import prepare,rank_assignments
                specs,scores=rank_assignments(local.policy,prepare(local.snapshot),pool)
            sets['common_geometry_pool']=CandidateSet(specs[:matched],np.asarray(scores[:matched]),specs[0],{'matched_effective_K':matched})
        for regime,candidates in sets.items():
            planner=Planner(local,H=arm['H'],E=arm['E'],K=arm['K'],memory=arm.get('memory','inherit'))
            rows=planner.evaluate(candidates) if not true_only else []
            continuations=panel.get('reference_continuations',[panel.get('reference_continuation','hccg_direct')])
            for index,spec in enumerate(candidates.specs):
                refs={};true_endpoint=None
                for kind in continuations:
                    ref,true_endpoint=reference(local,spec,continuation_template,kind,arm['E']);refs[kind]=ref
                # True first-segment scoring uses the same requested forecast cut, same approximate critic.
                from cogp.dynamics.diagnostic_error import rollout_with_force_error
                true_forecast,force_error=rollout_with_force_error(local,spec,local.predictor,H=arm['H'])
                true_view=local.clone_view(true_forecast);true_score=score_components(true_forecast.reward,true_forecast.duration,true_forecast.done,true_view.value(),local.pc.gamma)
                if true_only:
                    row=dict(order=index,tasks=task_key(spec),predicted_reward=true_forecast.reward,predicted_duration=true_forecast.duration,
                        predicted_terminal=true_forecast.done,predicted_value=true_view.value(),**true_score)
                    rows.append(row)
                row=rows[index];row['references']=refs;row['true_first_score']=true_score['score']
                row['full_score_discrepancy']=row['score']-true_score['score']
                row['first_reward_discrepancy']=row['reward_term']-true_score['reward_term']
                row['duration_discrepancy']=row['predicted_duration']-true_forecast.duration
                row['force_error']=force_error
            metrics={kind:_ranking([r['references'][kind]['reference_return'] for r in rows],[r['score'] for r in rows]) for kind in continuations}
            states.append(dict(state_id=state_id,seed=job['seed'],driver=label,regime=regime,generator=arm['generator'],
                candidate_info=candidates.info,candidates=rows,metrics=metrics,native_index=0,native_verified=True,native_scope='retained native winner' if regime=='native_hybrid' else 'top generator score on shared geometric support',source_state_hash=state_digest(source.env,source.co,source.ctx)))
    return dict(seed=job['seed'],arm=job['arm'],world=job['case']['case_id'],states=states,cpu_seconds=time.process_time()-cpu,
        reference_scope='true first commitment + specified frozen greedy continuation; finite candidate union, no global optimality')


def memory_common_states(job,registry,device='cpu'):
    panel=load_panel('memory');arm=panel['arms']['full'];template=Session.from_registry(registry,job['seed'],arm,device)
    rows=[]
    from cogp.planning.memory import FIELDS,reinitialize_history
    for label,source in _driver_states(registry,job['seed'],job['case'],panel,device):
        local=_state_session(template,source);candidates=build_candidates(local,arm['K'])
        full=Planner(local,memory='inherit').select(candidates)
        reset=Planner(local,memory='reinitialize').select(candidates)
        clone=clone_branch(local.env,local.co,local.ctx)
        changed=reinitialize_history(clone[1].executor.low_level,clone[0].num_dogs)
        rows.append(dict(state_id=f"s{job['seed']}/{job['case']['case_id']}/{label}",seed=job['seed'],
            selected_changed=full.record['selected_index']!=reset.record['selected_index'],history_change=changed,
            score_changed=any(abs(a['score']-b['score'])>1e-10 for a,b in zip(full.record['candidates'],reset.record['candidates'])),
            candidate_hash=candidates.info['candidate_hash'],full=full.record,reinitialized=reset.record))
    return dict(seed=job['seed'],world=job['case']['case_id'],states=rows,scope='sparse shared states; not complete trajectory mediation')


def score_decomposition(records):
    """Post-hoc re-ranking without new simulator calls or fitted coefficients."""
    if not records:raise ValueError('No candidate state records')
    output=[]
    for state in records:
        rows=state['candidates']
        if not rows:raise ValueError('Empty candidate set')
        for kind in rows[0]['references']:
            ref=[r['references'][kind]['reference_return'] for r in rows]
            verified=bool(state.get('native_verified',False)) and type(state.get('native_index')) is int and 0<=state['native_index']<len(rows)
            native_index=state['native_index'] if verified else 0
            rules={'reward_only':[r['reward_term'] for r in rows],
                   'tail_only':[r['tail_term'] for r in rows],'full':[r['score'] for r in rows]}
            if verified:rules['native']=[float(i==native_index) for i in range(len(rows))]
            for name,scores in rules.items():
                metric=_ranking(ref,scores,native_index)
                if not verified:metric.pop('native_gain');metric.pop('native_reference')
                output.append(dict(state_id=state['state_id'],seed=state['seed'],rule=name,continuation=kind,native_verified=verified,**metric))
    groups={}
    for row in output:groups.setdefault((row['seed'],row['rule'],row['continuation']),[]).append(row)
    seed_rows=[]
    for (seed,rule,kind),group in sorted(groups.items()):
        seed_rows.append(dict(seed=seed,rule=rule,continuation=kind,states=len(group),
            **{key:float(np.mean([r[key] for r in group])) for key in ('regret','best_agreement','native_gain') if all(key in r for r in group)}))
    return dict(schema='posthoc_score_decomposition',new_rollouts=False,new_training=False,seed_results=seed_rows,state_results=output,
        scope='descriptive per seed; targets are different continuation policies and must not be compared as algorithm gains')


def read_states(path):
    p=Path(path);files=list(p.rglob('*.json')) if p.is_dir() else [p]
    states=[]
    for file in files:
        data=json.loads(file.read_text());result=data.get('result',data)
        states.extend(result.get('states',[]))
    if not states:raise ValueError('No compatible full candidate records found; summary means cannot reconstruct raw scores')
    return states


def run_cross_job(job,registry,device='cpu'):
    """Evaluate each finite union assignment once on four matched driver states."""
    from cogp.coordination.ranking import prepare,rank_assignments
    from cogp.coordination.geometry import geometry_candidates
    from cogp.dynamics.diagnostic_error import rollout_with_force_error
    panel=load_panel('cross_generator');arms=job.get('cohort_arms',panel['arms'])
    templates={name:Session.from_registry(registry,job['seed'],arm,device) for name,arm in arms.items()}
    reference_template=templates['hccg'];cpu=time.process_time();states=[];counts=[]
    for label,source in _driver_states(registry,job['seed'],job['case'],panel,device):
        locals={name:_state_session(template,source) for name,template in templates.items()}
        native={name:build_candidates(local,arms[name]['K']) for name,local in locals.items()}
        matched=min(len(c.specs) for c in native.values())
        pool=geometry_candidates(prepare(locals['hccg'].snapshot),width=panel['common_geometry_width'])[0]
        common={}
        for name,local in locals.items():
            if name=='geometry':
                selected=pool[:matched];scores=native[name].scores[:matched]
            else:selected,scores=rank_assignments(local.policy,prepare(local.snapshot),pool);selected=selected[:matched];scores=scores[:matched]
            common[name]=CandidateSet(selected,np.asarray(scores),selected[0],{'matched_effective_K':matched,'support':'shared_geometry_beam256','native_retention':False})
        for candidate_set in native.values():
            candidate_set.specs=candidate_set.specs[:matched];candidate_set.scores=candidate_set.scores[:matched]
            candidate_set.info['matched_effective_K']=matched
        state_id=f"s{job['seed']}/{job['case']['case_id']}/{label}"
        cache={}
        for regime,sets in (('native_hybrid',native),('common_geometry_pool',common)):
            for name,candidates in sets.items():
                local=locals[name];arm=arms[name];rows=[]
                for index,spec in enumerate(candidates.specs):
                    key=task_key(spec)
                    if key not in cache:
                        # Shared model and HCCG critic imply identical scores for a shared assignment.
                        singleton=CandidateSet([spec],np.asarray([0.]),spec,{'diagnostic_union_singleton':True})
                        predicted=Planner(locals['hccg'],H=arm['H'],E=arm['E'],K=1).evaluate(singleton)[0]
                        true_forecast,error=rollout_with_force_error(locals['hccg'],spec,locals['hccg'].predictor,H=arm['H'])
                        true_local=locals['hccg'].clone_view(true_forecast)
                        true_score=score_components(true_forecast.reward,true_forecast.duration,true_forecast.done,true_local.value(),local.pc.gamma)
                        first=true_forecast if arm['H']==arm['E'] else rollout_branch(local.env,local.co,local.ctx,spec,None,arm['E'])
                        follow=reference_template.clone_view(first)
                        cont=continuation(follow,'hccg_direct') if not first.done else {'return_value':0.,'success':int(first.env.metrics()['success']>.5),'decisions':0}
                        ref=dict(reference_return=.1*first.reward+local.pc.gamma**first.duration*cont['return_value'],
                            true_reward=first.reward,true_duration=first.duration,true_terminal=first.done,
                            continuation='hccg_direct',continuation_return=cont['return_value'],success=cont['success'])
                        cache[key]={**predicted,'references':{'hccg_direct':ref},'true_first_score':true_score['score'],
                            'full_score_discrepancy':predicted['score']-true_score['score'],
                            'first_reward_discrepancy':predicted['reward_term']-true_score['reward_term'],
                            'duration_discrepancy':predicted['predicted_duration']-true_forecast.duration,'force_error':error}
                    row=copy.deepcopy(cache[key]);row['order']=index;rows.append(row)
                metrics={'hccg_direct':_ranking([r['references']['hccg_direct']['reference_return'] for r in rows],[r['score'] for r in rows])}
                states.append(dict(state_id=state_id,seed=job['seed'],driver=label,regime=regime,generator=name,
                    candidate_info=candidates.info,candidates=rows,metrics=metrics,native_index=0,native_verified=True,
                    native_scope='retained native winner' if regime=='native_hybrid' else 'top generator score on shared geometric support',
                    source_state_hash=state_digest(source.env,source.co,source.ctx)))
        counts.append(dict(state_id=state_id,unique_reference_assignments=len(cache),returned_candidate_records=sum(len(c.specs) for group in (native,common) for c in group.values()),matched_effective_K=matched))
    return dict(seed=job['seed'],arm=job['arm'],world=job['case']['case_id'],states=states,union_counts=counts,
        cpu_seconds=time.process_time()-cpu,reference_scope='each finite union assignment once; true first commitment + frozen greedy HCCG; no global optimality')
