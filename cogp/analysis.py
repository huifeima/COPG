"""Seed-paired experiment summaries, independent of simulator/model runtime.

All inferential intervals use complete paired training-seed panel means. Nested
episodes are never treated as independent training replicates. Missing panels
and missing contrast-family arms do not silently receive zero-valued outcomes.
"""
from __future__ import annotations
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

from .protocols import load_panel

class AnalysisError(ValueError):
    pass

class IncompletePanelError(AnalysisError):
    pass

def _boolean_marker(value: Any, field: str) -> bool:
    """Read JSON/CSV provenance flags without treating ``'False'`` as true."""
    if value is None or value=='':return False
    if isinstance(value,str):
        normalized=value.strip().lower()
        if normalized in ('true','1','yes'):return True
        if normalized in ('false','0','no',''):return False
    elif isinstance(value,(bool,int,float)) and value in (0,1):return bool(value)
    raise AnalysisError(f'Invalid {field} provenance marker: {value!r}')

def _engineering_marked(*sources: dict[str,Any]) -> bool:
    """Any positive engineering flag wins over an inner/outer false value."""
    marked=False
    for source in sources:
        if not isinstance(source,dict):continue
        for field in ('test_only','engineering_test'):
            if field in source:marked=_boolean_marker(source[field],field) or marked
        if source.get('scientific_policy') not in (None,''):
            marked=not _boolean_marker(source['scientific_policy'],'scientific_policy') or marked
    return marked

def _beta_fraction(a: float, b: float, x: float) -> float:
    tiny=1e-300; qab=a+b; qap=a+1.; qam=a-1.
    c=1.;d=1.-qab*x/qap
    if abs(d)<tiny:d=tiny
    d=1./d;h=d
    for m in range(1,301):
        m2=2*m;aa=m*(b-m)*x/((qam+m2)*(a+m2))
        d=1.+aa*d
        if abs(d)<tiny:d=tiny
        c=1.+aa/c
        if abs(c)<tiny:c=tiny
        d=1./d;h*=d*c
        aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2))
        d=1.+aa*d
        if abs(d)<tiny:d=tiny
        c=1.+aa/c
        if abs(c)<tiny:c=tiny
        d=1./d;delta=d*c;h*=delta
        if abs(delta-1.)<3e-14:return h
    raise AnalysisError('Incomplete-beta continued fraction did not converge')

def _regularized_beta(x: float, a: float, b: float) -> float:
    if x<=0.:return 0.
    if x>=1.:return 1.
    front=math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log1p(-x))
    if x<(a+1.)/(a+b+2.):return front*_beta_fraction(a,b,x)/a
    return 1.-front*_beta_fraction(b,a,1.-x)/b

def student_t_cdf(value: float, df: int) -> float:
    if df<=0:raise AnalysisError('Student t degrees of freedom must be positive')
    if math.isnan(value):raise AnalysisError('NaN t statistic')
    if value==0.:return .5
    if math.isinf(value):return 1. if value>0 else 0.
    tail=.5*_regularized_beta(df/(df+value*value),df/2.,.5)
    return 1.-tail if value>0 else tail

def student_t_quantile(probability: float, df: int) -> float:
    """Numerically invert the Student t CDF using only the standard library."""
    if not 0.<probability<1.:raise AnalysisError('t quantile probability must be strictly between zero and one')
    if probability==.5:return 0.
    if probability<.5:return -student_t_quantile(1.-probability,df)
    lo=0.;hi=1.
    while student_t_cdf(hi,df)<probability:
        hi*=2.
        if hi>1e12:raise AnalysisError('Unbounded t quantile')
    for _ in range(90):
        mid=(lo+hi)/2.
        if student_t_cdf(mid,df)<probability:lo=mid
        else:hi=mid
    return (lo+hi)/2.

def paired_interval(differences: Iterable[float], family_size: int=1, confidence: float=.95) -> dict[str, Any]:
    values=[float(x) for x in differences]
    if len(values)<2 or any(not math.isfinite(x) for x in values):raise AnalysisError('At least two finite paired seed differences are required')
    if type(family_size) is not int or family_size<1 or not 0.<confidence<1.:raise AnalysisError('Invalid contrast family or confidence')
    n=len(values);mean=statistics.mean(values);sd=statistics.stdev(values);se=sd/math.sqrt(n);df=n-1
    critical=student_t_quantile(1.-(1.-confidence)/(2.*family_size),df)
    if sd==0.:
        p=1. if mean==0. else 0.;statistic=None
    else:
        statistic=mean/se;p=2.*student_t_cdf(-abs(statistic),df)
    return dict(mean_difference_pp=mean,seed_difference_sd_pp=sd,standard_error_pp=se,
        lower_pp=mean-critical*se,upper_pp=mean+critical*se,n_training_seeds=n,df=df,
        family_size=family_size,confidence=confidence,critical_t=critical,p_value=p,
        family_adjusted_p=min(1.,p*family_size),t_statistic=statistic,degenerate_variance=sd==0.,
        statistical_unit='paired_training_seed_means',paired_seed_differences_pp=values)

def _finite_nonnegative(value: Any, field: str) -> float:
    try:out=float(value)
    except (TypeError,ValueError) as error:raise AnalysisError(f'Invalid {field}: {value}') from error
    if not math.isfinite(out) or out<0.:raise AnalysisError(f'Invalid {field}: {value}')
    return out

def _latencies(row: dict[str,Any]) -> tuple[list[float],str]:
    trace=row.get('decision_trace',row.get('trace'))
    if isinstance(trace,list):
        values=[step['selection_seconds'] for step in trace if isinstance(step,dict) and 'selection_seconds' in step]
        if values:return [_finite_nonnegative(x,'selection_seconds') for x in values],('decision' if len(values)==len(trace) else 'partial_decision')
    value=row.get('latencies',row.get('selection_seconds',row.get('latency')))
    if value is None or value=='':return [],'unavailable'
    if isinstance(value,str) and value.strip().startswith('['):
        try:value=json.loads(value)
        except ValueError as error:raise AnalysisError('Invalid latency array JSON') from error
    if isinstance(value,list):return [_finite_nonnegative(x,'latency') for x in value],'decision'
    return [_finite_nonnegative(value,'latency')],'supplied_scalar'

def normalize_record(record: dict[str,Any]) -> dict[str,Any]:
    if not isinstance(record,dict):raise AnalysisError('Episode record is not a mapping')
    status=record.get('status')
    if status not in (None,'') and str(status).upper()!='COMPLETE':
        raise AnalysisError('A failed or partial receipt cannot be analyzed as a completed episode')
    job=record.get('job',{})
    identity=record.get('identity',{})
    if not job and isinstance(identity,dict):job=identity.get('job',{})
    if not isinstance(job,dict):job={}
    body=record.get('result',record.get('body',{}))
    if not isinstance(body,dict):body={}
    identity_job=identity.get('job',{}) if isinstance(identity,dict) else {}
    provenance_sources=[record,job,identity,identity_job,body,record.get('body',{})]
    provenance_sources.extend(source.get('arm_config',{}) for source in list(provenance_sources) if isinstance(source,dict))
    test_only=_engineering_marked(*provenance_sources)
    row={**job,**record,**body}
    case=row.get('case',job.get('case',{}))
    if not isinstance(case,dict):case={}
    seed=row.get('training_seed',row.get('seed'))
    arm=row.get('arm');world=row.get('world',row.get('case_id',case.get('case_id')))
    condition=row.get('condition')
    if all(k in case for k in ('sheep','dogs','kind','scenario')):
        condition='|'.join(str(x) for x in (case['sheep'],case['dogs'],case['kind'],case['scenario'],case.get('physics_multiplier',1.0)))
    elif all(k in row for k in ('N','M','kind','scenario')):
        condition='|'.join(str(x) for x in (row['N'],row['M'],row['kind'],row['scenario'],row.get('physics_multiplier',1.0)))
    if seed is None or arm is None or world is None:raise AnalysisError('Episode must contain training seed, arm and world identity')
    if isinstance(seed,bool) or (isinstance(seed,float) and (not math.isfinite(seed) or seed!=math.floor(seed))):
        raise AnalysisError('Training seed must be an integer identity')
    try:seed=int(seed)
    except (ValueError,TypeError) as error:raise AnalysisError('Invalid training seed') from error
    success=row.get('success')
    if isinstance(success,str):
        try:success=float(success)
        except ValueError as error:raise AnalysisError('Invalid success outcome') from error
    if success not in (0,1,False,True):raise AnalysisError('Success must be exactly zero or one')
    cpu=row.get('cpu_seconds',row.get('episode_cpu_seconds'))
    cpu=None if cpu is None or cpu=='' else _finite_nonnegative(cpu,'cpu_seconds')
    latency,latency_scope=_latencies(row)
    return dict(seed=seed,arm=str(arm),world=str(world),condition=str(condition if condition is not None else 'unspecified'),
                success=int(success),test_only=test_only,scientific_policy=not test_only,
                cpu_seconds=cpu,latencies=latency,latency_scope=latency_scope,
                sheep=case.get('sheep',row.get('N',row.get('sheep'))),dogs=case.get('dogs',row.get('M',row.get('dogs'))),
                kind=case.get('kind',row.get('kind')),scenario=case.get('scenario',row.get('scenario')),
                physics_multiplier=case.get('physics_multiplier',row.get('physics_multiplier',1.0)),
                exclusive_worker=bool(row.get('exclusive_worker',job.get('exclusive_worker',False))))

def summarize_timing(records: Iterable[dict[str,Any]]) -> list[dict[str,Any]]:
    """Descriptive decision-weighted timings grouped by method, flock and team.

    Include successful and unsuccessful completed episodes. Raw decision arrays
    are required to claim full decision-latency coverage; scalar summaries are
    labelled separately. This function does not establish panel completeness.
    """
    normalized=[normalize_record(record) for record in records]
    groups=defaultdict(list)
    for row in normalized:groups[(row['arm'],row['sheep'],row['dogs'])].append(row)
    result=[]
    for (arm,sheep,dogs),rows in sorted(groups.items(),key=lambda item:tuple(str(x) for x in item[0])):
        values=[value for row in rows for value in row['latencies']]
        full=bool(rows) and all(row['latency_scope']=='decision' for row in rows)
        cpu=[row['cpu_seconds'] for row in rows if row['cpu_seconds'] is not None]
        result.append(dict(arm=arm,N=sheep,M=dogs,observed_episodes=len(rows),successful_episodes=sum(row['success'] for row in rows),
            test_only=any(row['test_only'] for row in rows),engineering_episodes=sum(row['test_only'] for row in rows),
            training_seeds=sorted({row['seed'] for row in rows}),latency_observations=len(values),
            decision_weighted_mean_seconds=statistics.mean(values) if values and full else None,
            supplied_observation_mean_seconds=statistics.mean(values) if values else None,
            observed_max_seconds=max(values) if values else None,complete_decision_latency_coverage=full,
            exclusive_worker=all(row['exclusive_worker'] for row in rows),
            mean_episode_cpu_seconds=statistics.mean(cpu) if cpu else None,cpu_episode_observations=len(cpu),
            kinds=sorted({str(row['kind']) for row in rows if row['kind'] is not None}),
            workloads=sorted({str(row['scenario']) for row in rows if row['scenario'] is not None}),
            physics_multipliers=sorted({float(row['physics_multiplier']) for row in rows}),
            descriptive_only=True,scope='observed completed trajectories; no isolated causal size effect or latency quantile claim'))
    return result

def read_episode_records(run_dir: str|Path) -> list[dict[str,Any]]:
    root=Path(run_dir)
    for name in ('episodes.csv','episode_records.csv'):
        path=root/name
        if path.is_file():
            with path.open(newline='',encoding='utf-8') as stream:return list(csv.DictReader(stream))
    for name in ('episodes.jsonl','episode_records.jsonl'):
        path=root/name
        if path.is_file():return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    records=[]
    for path in sorted((root/'receipts').rglob('*.json')):
        value=json.loads(path.read_text(encoding='utf-8'))
        if str(value.get('status','COMPLETE')).upper()!='COMPLETE':continue
        body=value.get('result',value.get('body',value))
        if isinstance(body,dict) and 'success' in body:records.append(value)
    return records

def summarize_records(records: Iterable[dict[str,Any]], panel: dict[str,Any], *, allow_incomplete: bool=False,
                      expected_worlds: Iterable[str]|None=None, allow_engineering: bool=False) -> dict[str,Any]:
    """Analyze a scientific panel, or explicitly describe engineering outputs.

    Engineering/test-only records (including CSV markers and wrapped receipts)
    are rejected by default. ``allow_engineering=True`` permits descriptive
    tables only; it can never enable inferential contrasts for such a panel.
    ``allow_incomplete`` does not override this provenance restriction.
    """
    raw_records=list(records)
    rows=[normalize_record(record) for record in raw_records]
    arm_configs=panel.get('arms',{})
    panel_test_only=_engineering_marked(panel,*(arm_configs.values() if isinstance(arm_configs,dict) else []))
    engineering_episodes=sum(row['test_only'] for row in rows)
    engineering_data=panel_test_only or bool(engineering_episodes)
    if engineering_data and not allow_engineering:
        raise AnalysisError('Engineering/test_only data cannot be used for scientific analysis; '
                            'allow_engineering=True permits descriptive output without inference only')
    seeds=[int(x) for x in panel['training_seeds']];arms=list(panel['arms']);world_count=panel.get('worlds')
    if not arms:raise AnalysisError('This panel has no closed-loop arms; use its dedicated offline diagnostic analyzer')
    if type(world_count) is not int or world_count<=0:raise AnalysisError('Closed-loop panel requires its finite world count')
    by=defaultdict(dict)
    for row in rows:
        if row['seed'] not in seeds or row['arm'] not in arms:raise AnalysisError(f'Unexpected arm or seed: {row["arm"]}/{row["seed"]}')
        key=(row['arm'],row['seed']);world=row['world']
        if world in by[key]:raise AnalysisError(f'Duplicate episode {row["arm"]}/{row["seed"]}/{world}')
        by[key][world]=row
    expected=set(str(x) for x in expected_worlds) if expected_worlds is not None else None
    if expected is not None and len(expected)!=world_count:raise AnalysisError('Expected-world manifest count differs from panel world count')
    if expected is None and rows:
        expected=set(by[(arms[0],seeds[0])])
    problems=[]
    for arm in arms:
        for seed in seeds:
            world_ids=set(by[(arm,seed)])
            if len(world_ids)!=world_count:problems.append(f'{arm}/seed{seed}: {len(world_ids)}/{world_count} worlds')
            if expected is not None and world_ids!=expected:problems.append(f'{arm}/seed{seed}: world identities differ from common fixed panel')
    complete=not problems
    if not complete and not allow_incomplete:raise IncompletePanelError('Incomplete or unmatched panel: '+'; '.join(problems[:8]))
    per_seed=[];per_condition=[];summary=[];arm_seed_values={}
    for arm in arms:
        summaries=[];latencies=[];cpu=[];conditions=defaultdict(lambda:defaultdict(list))
        for seed in seeds:
            episodes=list(by[(arm,seed)].values());successes=sum(x['success'] for x in episodes)
            rate=100.*successes/len(episodes) if episodes else None
            sr=dict(arm=arm,seed=seed,expected_episodes=world_count,observed_episodes=len(episodes),successes=successes,
                    success_percent=rate,test_only=panel_test_only or any(x['test_only'] for x in episodes),
                    complete_seed_panel=len(episodes)==world_count and (expected is None or set(by[(arm,seed)])==expected))
            per_seed.append(sr)
            if rate is not None:summaries.append(rate)
            arm_seed_values[(arm,seed)]=rate
            for row in episodes:
                latencies.extend(row['latencies'])
                if row['cpu_seconds'] is not None:cpu.append(row['cpu_seconds'])
                conditions[row['condition']][seed].append(row['success'])
        for condition,data in sorted(conditions.items()):
            seed_means=[100.*statistics.mean(data[s]) for s in seeds if data[s]]
            per_condition.append(dict(arm=arm,condition=condition,observed_training_seeds=len(seed_means),
                observed_episodes=sum(len(v) for v in data.values()),success_percent=statistics.mean(seed_means),
                seed_sd_pp=statistics.stdev(seed_means) if len(seed_means)>1 else None,descriptive_only=True))
        arm_rows=[row for row in rows if row['arm']==arm]
        scope='decision' if arm_rows and all(x['latency_scope']=='decision' for x in arm_rows) else 'supplied_observations_not_full_decision_distribution'
        summary.append(dict(arm=arm,complete_panel=complete,expected_episodes=world_count*len(seeds),observed_episodes=len(arm_rows),
            test_only=panel_test_only or any(row['test_only'] for row in arm_rows),
            successes=sum(row['success'] for row in arm_rows),mean_seed_success_percent=statistics.mean(summaries) if summaries else None,
            seed_sd_pp=statistics.stdev(summaries) if len(summaries)>1 else None,n_training_seeds=len(summaries),
            mean_episode_cpu_seconds=statistics.mean(cpu) if cpu else None,cpu_observations=len(cpu),
            selection_latency_mean_seconds=statistics.mean(latencies) if latencies else None,
            selection_latency_observed_max_seconds=max(latencies) if latencies else None,
            latency_observations=len(latencies),latency_scope=scope))
    contrasts=[];families=[]
    for family in panel.get('statistics_families',[]):
        named=family.get('contrasts',[]);size=int(family.get('size',len(named)))
        if size!=len(named):raise AnalysisError('Declared family size differs from defined contrast count')
        required={arm for contrast in named for arm in contrast['coefficients']}
        absent=sorted(required-set(arms));family_complete=complete and not absent
        if absent and not allow_incomplete:raise IncompletePanelError(f'Family {family["name"]} requires missing arms: {absent}')
        families.append(dict(name=family['name'],size=size,complete=family_complete,missing_arms=absent,
                             inference_enabled=family_complete and not engineering_data))
        if not family_complete or engineering_data:continue
        if len(seeds)<2:raise AnalysisError('Paired inference requires at least two trained seeds')
        for contrast in named:
            values=[sum(float(coefficient)*arm_seed_values[(arm,seed)] for arm,coefficient in contrast['coefficients'].items()) for seed in seeds]
            estimate=paired_interval(values,size,float(family.get('confidence',.95)))
            contrasts.append(dict(family=family['name'],contrast=contrast['name'],coefficients=contrast['coefficients'],
                                  complete_family=True,seeds=seeds,**estimate))
    return dict(schema='seed_paired_analysis',panel=panel['name'],complete_panel=complete,
        test_only=engineering_data,scientific_policy=not engineering_data,engineering_episodes=engineering_episodes,
        analysis_mode='engineering_descriptive' if engineering_data else 'scientific',
        inference_enabled=not engineering_data and complete and bool(families) and all(f['complete'] for f in families),
        training_seeds=seeds,worlds_per_seed=world_count,world_manifest_matched=expected_worlds is not None,
        statistical_scope=panel.get('statistical_scope','conditional on fixed worlds and executor'),
        incomplete_problems=problems,summary=summary,per_seed=per_seed,per_condition=per_condition,
        contrast_families=families,contrasts=contrasts,episode_records=len(rows),timing_by_N_M=summarize_timing(raw_records),
        restrictions=['episodes are nested outcomes, not training replicates','condition summaries are descriptive','observed timing maxima are not uncertainty intervals']
        +(['engineering/test_only outputs are descriptive only; scientific inference is disabled'] if engineering_data else []))

def _write_csv(path:Path,rows:list[dict[str,Any]]) -> None:
    if not rows:
        path.write_text('',encoding='utf-8');return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fields);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in row.items()})

def analyze(run_dir: str|Path, panel: dict[str,Any]|str, *, records: Iterable[dict[str,Any]]|None=None,
            allow_incomplete:bool=False, output_dir:str|Path|None=None, expected_worlds:Iterable[str]|None=None,
            allow_engineering:bool=False) -> dict[str,Any]:
    if isinstance(panel,str):panel=load_panel(panel)
    root=Path(run_dir)
    result=summarize_records(read_episode_records(root) if records is None else records,panel,
                             allow_incomplete=allow_incomplete,expected_worlds=expected_worlds,allow_engineering=allow_engineering)
    dest=Path(output_dir) if output_dir is not None else root/'analysis';dest.mkdir(parents=True,exist_ok=True)
    (dest/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for name in ('summary','per_seed','per_condition','contrasts','timing_by_N_M'):_write_csv(dest/(name+'.csv'),result[name])
    return result

analyse=analyze

def plot(summary: dict[str,Any]|str|Path, output: str|Path) -> list[str]:
    """Draw exact scientific summaries. Matplotlib is optional until called."""
    if isinstance(summary,(str,Path)):summary=json.loads(Path(summary).read_text(encoding='utf-8'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination=Path(output)
    if destination.suffix.lower() in ('.pdf','.png','.svg'):
        path=destination;path.parent.mkdir(parents=True,exist_ok=True)
    else:
        destination.mkdir(parents=True,exist_ok=True);path=destination/'success_and_paired_effects.pdf'
    rows=summary['summary'];contrasts=summary.get('contrasts',[])
    ncols=2 if contrasts else 1
    fig,axes=plt.subplots(1,ncols,figsize=(11 if contrasts else 6,max(3.2,.38*max(len(rows),len(contrasts))+1.6)),squeeze=False)
    ax=axes[0,0];ys=list(range(len(rows)))
    means=[row['mean_seed_success_percent'] or 0. for row in rows]
    errors=[row['seed_sd_pp'] or 0. for row in rows]
    ax.barh(ys,means,xerr=errors,color='#2c6d8a',capsize=3)
    ax.set_yticks(ys,[row['arm'].replace('_',' ') for row in rows]);ax.invert_yaxis();ax.set_xlim(0,100)
    ax.set_xlabel('Success (%)');ax.set_title('Seed means; bars show seed SD')
    if summary.get('test_only',False):ax.text(.98,.02,'ENGINEERING: descriptive only',transform=ax.transAxes,ha='right',color='#a32b20',fontsize=9)
    elif not summary.get('complete_panel',False):ax.text(.98,.02,'INCOMPLETE: descriptive only',transform=ax.transAxes,ha='right',color='#a32b20',fontsize=9)
    if contrasts:
        ax=axes[0,1]
        for i,row in enumerate(contrasts):
            m=row['mean_difference_pp'];lo=row['lower_pp'];hi=row['upper_pp']
            ax.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',color='#b46a20',capsize=3)
            ax.scatter(row['paired_seed_differences_pp'],[i+.13]*len(row['paired_seed_differences_pp']),s=12,color='#2c6d8a',alpha=.6)
        ax.axvline(0,color='0.6',linewidth=.8);ax.set_yticks(range(len(contrasts)),[row['family']+': '+row['contrast'] for row in contrasts]);ax.invert_yaxis()
        ax.set_xlabel('Paired difference (percentage points)');ax.set_title('Family-adjusted 95% intervals; seed points')
    fig.suptitle(summary['panel'].replace('_',' '));fig.tight_layout();fig.savefig(path,bbox_inches='tight');plt.close(fig)
    return [str(path)]

def analyze_panel(name: str, output_root: str|Path, config_dir: str|Path|None=None, *,
                  allow_incomplete: bool=False, output_dir: str|Path|None=None, allow_engineering: bool=False) -> dict[str,Any]:
    """CLI convenience wrapper for the deterministic protocol receipt layout."""
    panel=load_panel(name,config_dir);base=Path(output_root)
    root=base/name if (base/name).is_dir() else base
    expected=None
    candidates=[root/'worlds.json',root/'manifest.json']
    if panel.get('world_manifest'):
        configured=Path(panel['world_manifest'])
        candidates.append(configured if configured.is_absolute() else Path(panel['config_dir'])/configured)
    for manifest in candidates:
        if not manifest.is_file():continue
        value=json.loads(manifest.read_text(encoding='utf-8'))
        if isinstance(value,list) and all(isinstance(w,dict) and 'case_id' in w for w in value):
            expected=[w['case_id'] for w in value];break
    return analyze(root,panel,allow_incomplete=allow_incomplete,output_dir=output_dir,expected_worlds=expected,allow_engineering=allow_engineering)
