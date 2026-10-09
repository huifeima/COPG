"""Spawned finite-panel evaluation with verified, atomic per-job receipts.

A completed receipt is reusable only when its contents, scientific inputs,
assets, code and execution device match. Timing panels always run one worker.
"""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import importlib.metadata
import json
import multiprocessing as mp
import os
import time
import traceback

from cogp.assets import AssetRegistry
from cogp.io import atomic_json, digest, emit, lock, sha
from cogp.protocols import load_panel, load_worlds, make_jobs


def required_assets(panel):
    """Assets read by the fixed arms, not every optional registered file."""
    if panel.get('kind') == 'offline_analysis':
        records=panel.get('candidate_records')
        return [] if panel.get('candidate_records_path') else ([records] if isinstance(records,str) else [])
    result={'basc.collect','basc.drive'}
    names={'hccg':'HCCG','edge_gnn':'EdgeGNN','agent_dag':'AgentDAG','geometry':'HCCG'}
    for seed in panel['training_seeds']:
        for arm in panel['arms'].values():
            parent=names[arm['generator']]
            result.add(f'parent.{seed}.{parent}')
            owner=parent if arm.get('critic')=='own' else 'HCCG'
            result.add(f'parent.{seed}.{owner}')
            if arm.get('planner'):
                kind='graph' if arm.get('dynamics')=='true_diagnostic_only' else arm.get('dynamics','graph')
                result.add(f'dynamics.{seed}.{kind}')
            if arm.get('actor','default')!='default':result.add(f"actor.{seed}.{arm['actor']}")
            if arm.get('critic') in ('physical','execution'):result.add(f"critic.{seed}.{arm['critic']}")
            if arm.get('proposal') in ('return_rank','winner','winner_imitation'):
                result.add(f"proposal.{seed}.{parent}.{arm['proposal']}")
    if panel.get('kind')=='diagnostic':
        for seed in panel['training_seeds']:
            result.add(f'parent.{seed}.HCCG')
            if panel.get('name')=='cross_generator':
                for method in ('HCCG','EdgeGNN','AgentDAG'):result.add(f'parent.{seed}.{method}')
            if panel.get('driver')=='hccg_cogp' or 'hccg_cogp' in panel.get('reference_continuations',[]):
                result.add(f'dynamics.{seed}.graph')
    return sorted(result)


def source_identity():
    root=Path(__file__).resolve().parent
    return digest({str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*.py'))})


def runtime_versions():
    result={}
    for package in ('numpy','torch','gymnasium','stable-baselines3','scipy'):
        try:result[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:result[package]=None
    return result


def choose_workers(requested='auto', *, exclusive=False, memory_per_worker_gib=1.5):
    if exclusive:return 1
    available=max(1,(os.cpu_count() or 1)-2)
    if str(requested)!='auto':
        count=int(requested)
        if count<1:raise ValueError('jobs must be positive or auto')
        return count
    try:
        import psutil
        available=min(available,max(1,int(psutil.virtual_memory().available/(memory_per_worker_gib*(1<<30)))))
    except ImportError:pass
    return available


def _receipt_matches(path,identity):
    path=Path(path); sidecar=path.with_suffix(path.suffix+'.sha256.json')
    if not path.exists() and not sidecar.exists():return False
    if not path.is_file() or not sidecar.is_file():
        raise RuntimeError(f'Incomplete receipt pair at {path}; inspect and remove the incomplete pair explicitly before rerunning')
    stamp=json.loads(sidecar.read_text())
    if stamp.get('sha256')!=sha(path):raise RuntimeError(f'Receipt checksum mismatch: {path}')
    record=json.loads(path.read_text())
    if record.get('status')!='COMPLETE':raise RuntimeError(f'Receipt is not COMPLETE: {path}')
    if record.get('identity')!=identity:
        raise RuntimeError(f'Changed inputs/code/device at {path}; use a new output directory rather than combining incompatible results')
    return True


def _evaluate(job,assets_path,device,identity):
    """Top-level worker entry is spawn/pickle safe and never inherits a CUDA model."""
    path=Path(job['path']); begun=time.perf_counter()
    try:
        with lock(path.with_suffix(path.suffix+'.lock')):
            if _receipt_matches(path,identity):return {'status':'RESUMED','path':str(path),'job':job}
            from cogp.io import configure_torch
            configure_torch(job.get('seed',0),device)
            registry=AssetRegistry(assets_path)
            if job['kind'] in ('episode','timing'):
                from cogp.runtime import run_episode
                result=run_episode(job,registry,device)
            elif job['kind']=='offline_analysis':
                from cogp.diagnostics import score_decomposition,read_states
                records=Path(job['candidate_records'])
                if not records.exists():records=registry.resolve(job['candidate_records'])
                result=score_decomposition(read_states(records))
            elif job['kind']=='diagnostic':
                from cogp.diagnostics import run_job
                result=run_job(job,registry,device)
            else:raise ValueError('Unsupported job kind: '+job['kind'])
            if job.get('test_only'):result['test_only']=True
            record={'status':'COMPLETE','identity':identity,'job':job,'result':result,
                    'worker_wall_seconds':time.perf_counter()-begun}
            atomic_json(path,record)
            atomic_json(path.with_suffix(path.suffix+'.sha256.json'),{'schema':'experiment_receipt_sha256','sha256':sha(path),'identity':identity})
            error=path.with_suffix(path.suffix+'.error.json')
            if error.exists():error.unlink()
            return {'status':'COMPLETE','path':str(path),'job':job,'success':result.get('success')}
    except Exception as error:
        detail={'status':'ERROR','identity':identity,'job':job,'type':type(error).__name__,
                'message':str(error),'traceback':traceback.format_exc(),'worker_wall_seconds':time.perf_counter()-begun}
        atomic_json(path.with_suffix(path.suffix+'.error.json'),detail)
        return {'status':'ERROR','path':str(path),'job':job,'error':str(error)}


def run_panel(name,assets_path,output_root,*,jobs='auto',devices=('cpu',),world_manifest=None,
              config_dir=None,fail_fast=False,candidate_records=None):
    panel=load_panel(name,config_dir);registry=AssetRegistry(assets_path)
    if candidate_records is not None:
        if panel.get('kind')!='offline_analysis':raise ValueError('--records applies only to an offline-analysis panel')
        records=Path(candidate_records).expanduser().resolve()
        if not records.exists():raise FileNotFoundError(records)
        panel['candidate_records']=str(records);panel['candidate_records_path']=True
    prerequisites=registry.check(required_assets(panel))
    if not prerequisites['ready']:
        raise RuntimeError('Required experiment assets are unavailable: '+json.dumps(prerequisites,ensure_ascii=False))
    work=make_jobs(panel,output_root,world_manifest)
    if panel.get('test_only'):
        for job in work:
            job['test_only']=True
            if 'arm_config' in job:job['arm_config']['engineering_test']=True
    devices=tuple(str(x) for x in devices)
    if not devices:raise ValueError('At least one execution device is required')
    workers=choose_workers(jobs,exclusive=panel.get('exclusive_worker',False))
    workers=min(workers,max(1,len(work)))
    panel_path=Path(output_root).expanduser().resolve()/name
    scientific={k:v for k,v in panel.items() if k!='config_dir'}
    identity_common={'schema':'fixed_panel_run','panel':scientific,'assets':{k:sha(registry.resolve(k)) for k in required_assets(panel)},
                     'code_sha256':source_identity(),'runtime':runtime_versions()}
    if panel.get('kind')!='offline_analysis':
        worlds=load_worlds(panel,world_manifest);identity_common['worlds_sha256']=digest(worlds)
    else:
        worlds=[]
        records=Path(panel['candidate_records']) if panel.get('candidate_records_path') else registry.resolve(panel['candidate_records'])
        identity_common['records_sha256']=sha(records) if records.is_file() else digest({str(p.relative_to(records)):sha(p) for p in sorted(records.rglob('*.json'))})
    manifest={'schema':'finite_panel_run','panel':name,'scientific_identity':identity_common,'worlds':len(worlds),
              'job_count':len(work),'workers':workers,'devices':devices,'explicit_manifest_override':world_manifest is not None}
    manifest_path=panel_path/'run.json'
    if manifest_path.exists():
        prior=json.loads(manifest_path.read_text())
        if prior['scientific_identity']!=identity_common:raise RuntimeError('Output directory belongs to different scientific inputs; choose a new output directory')
    else:atomic_json(manifest_path,manifest)
    if worlds:atomic_json(panel_path/'worlds.json',worlds)
    pending=[];resumed=0
    for index,job in enumerate(work):
        device=devices[index%len(devices)]
        identity=digest({'run':identity_common,'job':{k:v for k,v in job.items() if k!='path'},'device':device})
        if _receipt_matches(job['path'],identity):resumed+=1
        else:pending.append((job,device,identity))
    emit('RESOURCE_PLAN',panel=name,workers=workers,devices=devices,total=len(work),pending=len(pending),resumed=resumed,
         exclusive_timing=bool(panel.get('exclusive_worker')))
    failures=[];completed=0
    if pending:
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
            futures={pool.submit(_evaluate,job,str(registry.path),device,identity):job for job,device,identity in pending}
            for future in as_completed(futures):
                try:receipt=future.result()
                except Exception as error:
                    job=futures[future]
                    receipt={'status':'ERROR','job':job,'path':job['path'],'error':repr(error)}
                    atomic_json(Path(job['path']).with_suffix('.json.error.json'),receipt)
                if receipt['status']=='ERROR':failures.append(receipt)
                else:completed+=1
                emit('JOB',status=receipt['status'],panel=name,seed=receipt['job'].get('seed'),arm=receipt['job'].get('arm'),
                     case_id=receipt['job'].get('case',{}).get('case_id'),completed=completed,resumed=resumed,
                     total=len(work),success=receipt.get('success'),error=receipt.get('error'))
                if failures and fail_fast:
                    for item in futures:item.cancel()
                    break
    summary={'status':'COMPLETE' if completed+resumed==len(work) and not failures else 'INCOMPLETE',
             'panel':name,'expected':len(work),'completed_now':completed,'resumed':resumed,'errors':failures,
             'run_identity':digest(identity_common)}
    atomic_json(panel_path/'run_status.json',summary)
    if summary['status']!='COMPLETE':raise RuntimeError(f"Panel {name} incomplete: {len(failures)} error jobs; see {panel_path/'run_status.json'}")
    emit('PANEL_COMPLETE',panel=name,total=len(work),resumed=resumed)
    return summary
