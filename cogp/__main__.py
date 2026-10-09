"""Unified, standalone command-line interface for the COGP support code."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from cogp.io import atomic_json,emit
from cogp.assets import AssetRegistry
from cogp.protocols import list_panels,load_panel,load_worlds,generate_worlds

PROJECT=Path(__file__).resolve().parents[1]
DEFAULT_ASSETS=PROJECT/'assets'/'manifest.json'


def _json(path):return json.loads(Path(path).expanduser().read_text(encoding='utf-8'))


def _registry(arguments):return AssetRegistry(arguments.assets)


def _assets(parser):parser.add_argument('--assets',type=Path,default=DEFAULT_ASSETS,help='Explicit asset manifest; checkpoint SHA-256 is verified')


def _panel(parser):
    parser.add_argument('--panel',required=True)
    parser.add_argument('--config-dir',type=Path)




def build_parser():
    parser=argparse.ArgumentParser(prog='cogp',description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    panels=sub.add_parser('panels',help='List fixed experiment panels');panels.add_argument('--config-dir',type=Path)
    init=sub.add_parser('init-assets',help='Create an empty manifest without generating model weights');_assets(init)
    add=sub.add_parser('import-asset',help='Copy one real checkpoint/data asset and record its hash');_assets(add)
    add.add_argument('--key',required=True);add.add_argument('--source',type=Path,required=True);add.add_argument('--destination',type=Path)
    worlds=sub.add_parser('make-worlds',help='Explicitly generate a new independent evaluation panel');_panel(worlds)
    worlds.add_argument('--base-seed',type=int,required=True);worlds.add_argument('--worlds-per-condition',type=int,required=True);worlds.add_argument('--output',type=Path,required=True)
    run=sub.add_parser('run',help='Evaluate a fixed finite panel with spawned workers and hash-verified resume');_assets(run);_panel(run)
    run.add_argument('--output',type=Path,default=PROJECT/'runs');run.add_argument('--jobs',default='auto')
    run.add_argument('--devices',default='cpu',help='Comma-separated devices; each job has one device')
    run.add_argument('--worlds',type=Path,help='Explicit replacement manifest, recorded as a new panel')
    run.add_argument('--fail-fast',action='store_true')
    run.add_argument('--records',type=Path,help='Explicit complete candidate JSON records for the offline score-components panel')
    train=sub.add_parser('train-parent',help='Train a real relational actor/default critic using eight environment streams');_assets(train)
    train.add_argument('--method',choices=('HCCG','EdgeGNN','AgentDAG'),required=True);train.add_argument('--seed',type=int,required=True)
    train.add_argument('--output',type=Path,required=True);train.add_argument('--device',default='cpu')
    train.add_argument('--seed-address',type=Path,help='Explicit SeedAddress JSON for archived reset-address replay')
    train.add_argument('--actor-variant',choices=('physical','execution'));train.add_argument('--study-hash')
    collect=sub.add_parser('collect-dynamics',help='Collect true sheep acceleration labels using frozen HCCG');_assets(collect)
    collect.add_argument('--seed',type=int,required=True);collect.add_argument('--output',type=Path,required=True);collect.add_argument('--device',default='cpu')
    collect.add_argument('--world-seed-origin',type=int,required=True,help='Explicit unused training-world address; never inferred from unavailable historical files')
    collect.add_argument('--decisions',type=int,default=2048);collect.add_argument('--sampling-seed',type=int)
    collect.add_argument('--scenario-seed',type=int);collect.add_argument('--reservoir-seed',type=int)
    collect.add_argument('--capacity',type=int,default=60000);collect.add_argument('--stride',type=int,default=12)
    fit=sub.add_parser('fit-dynamics',help='Fit Graph, DeepSets, Boids identification, or the local-pooled control')
    fit.add_argument('--data',type=Path,required=True);fit.add_argument('--output',type=Path,required=True)
    fit.add_argument('--kind',choices=('graph','deepsets','boids_id','local_pooled'),required=True);fit.add_argument('--seed',type=int,required=True)
    fit.add_argument('--device',default='cpu');fit.add_argument('--updates',type=int,default=5000);fit.add_argument('--batch-size',type=int,default=64)
    fit.add_argument('--geometry',type=Path);fit.add_argument('--split-manifest',type=Path)
    convert=sub.add_parser('normalize-control',help='Convert compatible archived BoidsID/local-pooled tensors without source-tree imports')
    convert.add_argument('--source',type=Path,required=True);convert.add_argument('--output',type=Path,required=True)
    convert.add_argument('--kind',choices=('boids_id','local_pooled'),required=True);convert.add_argument('--geometry',type=Path);convert.add_argument('--seed',type=int,required=True)
    labels=sub.add_parser('collect-critic-labels',help='Collect exact paused-endpoint continuation labels on train/validation worlds');_assets(labels)
    labels.add_argument('--seed',type=int,required=True);labels.add_argument('--output',type=Path,required=True);labels.add_argument('--device',default='cpu')
    labels.add_argument('--worlds',type=Path,help='Train/validation JSON mapping, or a list with explicit split fields')
    labels.add_argument('--split',choices=('train','validation'));labels.add_argument('--study-hash')
    critics=sub.add_parser('fit-critics',help='Fit matched physical/execution-information critics on identical labels')
    critics.add_argument('--data',type=Path,required=True);critics.add_argument('--seed',type=int,required=True);critics.add_argument('--study-hash')
    critics.add_argument('--device',default='cpu');critics.add_argument('--options',type=Path)
    proposals=sub.add_parser('collect-proposal-labels',help='Collect fixed true-simulator reference labels on 16/8 explicit train/validation worlds');_assets(proposals)
    proposals.add_argument('--seed',type=int,required=True);proposals.add_argument('--train-worlds',type=Path,required=True)
    proposals.add_argument('--validation-worlds',type=Path,required=True);proposals.add_argument('--output',type=Path,required=True);proposals.add_argument('--device',default='cpu')
    head=sub.add_parser('fit-proposal-head',help='Fit the fixed return-ranking or winner-imitation residual head')
    head.add_argument('--records',type=Path,required=True);head.add_argument('--output',type=Path,required=True)
    head.add_argument('--mode',choices=('return_rank','winner'),required=True)
    head.add_argument('--generator',choices=('HCCG','EdgeGNN','AgentDAG'),help='Required when the collector contains multiple generators')
    diag=sub.add_parser('analyze-diagnostics',help='Summarize candidate supply, ranking and scoring records')
    diag.add_argument('--records',type=Path,required=True);diag.add_argument('--output',type=Path,required=True);diag.add_argument('--plot','--figures',dest='plot',action='store_true')
    meman=sub.add_parser('analyze-memory',help='Summarize sparse shared-state history interventions separately from episode success')
    meman.add_argument('--records',type=Path,required=True);meman.add_argument('--output',type=Path,required=True)
    score=sub.add_parser('score-components',help='Offline Native/Reward-only/Tail-only/Full reranking on actual candidate records')
    score.add_argument('--records',type=Path,required=True);score.add_argument('--output',type=Path,required=True)
    memory=sub.add_parser('memory-diagnostic',help='Compare inherited/reinitialized history at shared early trajectory states');_assets(memory)
    memory.add_argument('--seed',type=int,required=True);memory.add_argument('--case-id',required=True);memory.add_argument('--output',type=Path,required=True)
    memory.add_argument('--worlds',type=Path);memory.add_argument('--config-dir',type=Path);memory.add_argument('--device',default='cpu')
    analyze=sub.add_parser('analyze',help='Paired-seed summaries and prespecified simultaneous confidence intervals');_panel(analyze)
    analyze.add_argument('--output',type=Path,default=PROJECT/'runs');analyze.add_argument('--allow-incomplete',action='store_true')
    analyze.add_argument('--allow-engineering',action='store_true',help='Describe engineering/test_only records; never compute their inferential intervals')
    analyze.add_argument('--figures',type=Path)
    plot=sub.add_parser('plot',help='Export publication-style figures from an analysis JSON')
    plot.add_argument('--summary',type=Path,required=True);plot.add_argument('--output',type=Path,required=True)
    return parser


def main(argv=None):
    args=build_parser().parse_args(argv)
    command=args.command
    if command=='panels':
        for name in list_panels(args.config_dir):print(name)
    elif command=='init-assets':
        if args.assets.exists():raise FileExistsError(f'Asset manifest already exists: {args.assets}')
        atomic_json(args.assets,{'schema':'checkpoint_assets','assets':{}});emit('ASSET_MANIFEST',path=args.assets)
    elif command=='import-asset':emit('ASSET_IMPORTED',key=args.key,**_registry(args).add(args.key,args.source,args.destination))
    elif command=='make-worlds':emit('NEW_WORLDS',**generate_worlds(args.panel,args.base_seed,args.worlds_per_condition,args.output,args.config_dir))
    elif command=='run':
        from cogp.runner import run_panel
        run_panel(args.panel,args.assets,args.output,jobs=args.jobs,devices=tuple(x.strip() for x in args.devices.split(',') if x.strip()),
                  world_manifest=args.worlds,config_dir=args.config_dir,fail_fast=args.fail_fast,candidate_records=args.records)
    elif command=='train-parent':
        from cogp.learning.parent.train import train
        from cogp.learning.parent.protocol import SeedAddress
        registry=_registry(args)
        address=SeedAddress(**_json(args.seed_address)) if args.seed_address else None
        emit('PARENT_TRAINED',**train(args.output,args.method,args.seed,args.device,registry.resolve('basc.collect'),registry.resolve('basc.drive'),
                                    seed_address=address,actor_variant=args.actor_variant,study_hash=args.study_hash))
    elif command=='collect-dynamics':
        from cogp.runtime import Session
        from cogp.learning.dynamics_data import collect
        from cogp.protocols import known_world_seeds
        registry=_registry(args);parent=registry.resolve(f'parent.{args.seed}.HCCG')
        session=Session.from_registry(registry,args.seed,dict(generator='hccg',planner=False,critic='original',E=64),args.device)
        emit('DYNAMICS_COLLECTED',**collect(session,args.output,args.seed,world_seed_origin=args.world_seed_origin,decisions=args.decisions,
              sampling_seed=args.sampling_seed,scenario_seed=args.scenario_seed,reservoir_seed=args.reservoir_seed,
              capacity=args.capacity,stride=args.stride,parent_checkpoint=parent,occupied_world_seeds=known_world_seeds()))
    elif command=='fit-dynamics':
        from cogp.dynamics.fit import fit
        result=fit(args.data,args.output,args.kind,_json(args.geometry) if args.geometry else {},args.seed,
                   updates=args.updates,batch_size=args.batch_size,device=args.device,split_manifest=args.split_manifest)
        emit('DYNAMICS_FITTED',**result)
    elif command=='normalize-control':
        from cogp.dynamics.convert import normalize_controls
        result=normalize_controls(args.source,args.output,args.kind,geometry=_json(args.geometry) if args.geometry else None,seed=args.seed)
        emit('CONTROL_NORMALIZED',**result)
    elif command=='collect-critic-labels':
        from cogp.learning.critic_data import collect_panel
        emit('CRITIC_LABELS',**collect_panel(_registry(args),args.worlds,args.output,args.seed,args.device,study_hash=args.study_hash,split=args.split))
    elif command=='fit-critics':
        from cogp.controls.critic import fit_matched_critics
        study_hash=args.study_hash
        if study_hash is None:
            metadata=args.data/'critic_labels'/f's{args.seed}'/'panel.json'
            if not metadata.is_file():raise FileNotFoundError('Provide --study-hash or the collector panel.json: '+str(metadata))
            study_hash=_json(metadata)['study_hash']
        emit('CRITICS_FITTED',**fit_matched_critics(args.data,seed=args.seed,study_hash=study_hash,device=args.device,options=_json(args.options) if args.options else None))
    elif command=='collect-proposal-labels':
        from cogp.learning.proposal_data import collect
        states=collect(_registry(args),args.seed,args.train_worlds,args.validation_worlds,args.output,args.device)
        emit('PROPOSAL_LABELS',states=len(states),seed=args.seed,output=args.output)
    elif command=='fit-proposal-head':
        from cogp.refinements import fit_head,_records
        records=_records(args.records)
        if args.generator:
            normalize=lambda value:str(value).lower().replace('_','').replace('-','')
            records=[row for row in records if normalize(row['generator'])==normalize(args.generator)]
            if not records:raise ValueError('No label records for the requested generator')
        emit('PROPOSAL_HEAD_FITTED',**fit_head(records,args.output,args.mode))
    elif command=='analyze-diagnostics':
        from cogp.diagnostic_analysis import analyze
        result=analyze(args.records,args.output,plot=args.plot)
        emit('DIAGNOSTICS_ANALYZED',output=args.output,schema=result.get('schema'))
    elif command=='analyze-memory':
        from cogp.diagnostic_analysis import read_states,summarize_memory
        states,inputs=read_states(args.records);result=summarize_memory(states);result['inputs']=inputs
        output=args.output if args.output.suffix=='.json' else args.output/'memory_summary.json'
        atomic_json(output,result);emit('MEMORY_ANALYZED',output=output,**result['summary'])
    elif command=='score-components':
        from cogp.diagnostics import read_states,score_decomposition
        result=score_decomposition(read_states(args.records));atomic_json(args.output,result);emit('SCORE_COMPONENTS',path=args.output,**result)
    elif command=='memory-diagnostic':
        from cogp.diagnostics import memory_common_states
        panel=load_panel('memory',args.config_dir);worlds=load_worlds(panel,args.worlds)
        case=next((case for case in worlds if case['case_id']==args.case_id),None)
        if case is None:raise ValueError('Unknown case-id in the complete memory panel')
        if args.seed not in panel['training_seeds']:raise ValueError('Seed outside the prespecified memory protocol')
        job={'kind':'diagnostic','panel':'memory','seed':args.seed,'arm':'full','arm_config':panel['arms']['full'],'case':case}
        result=memory_common_states(job,_registry(args),args.device);atomic_json(args.output,result);emit('MEMORY_DIAGNOSTIC',path=args.output)
    elif command=='analyze':
        from cogp.analysis import analyze_panel,plot
        result=analyze_panel(args.panel,args.output,args.config_dir,allow_incomplete=args.allow_incomplete,
                             allow_engineering=args.allow_engineering)
        if args.figures:plot(result,args.figures)
        emit('ANALYZED',panel=args.panel,status='COMPLETE' if result.get('inference_enabled') else 'DESCRIPTIVE',
             analysis_mode=result['analysis_mode'],inference_enabled=result['inference_enabled'],output=args.output)
    elif command=='plot':
        from cogp.analysis import plot
        emit('FIGURES',paths=plot(args.summary,args.output))
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (RuntimeError,ValueError,FileNotFoundError,FileExistsError,KeyError) as error:
        print(f'ERROR: {type(error).__name__}: {error}',file=sys.stderr)
        raise SystemExit(2)
