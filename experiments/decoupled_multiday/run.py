"""Five-seed ablations, raw-metric weight selection, freeze, then diagnostic test."""
import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from experiments.similarity_graph.prepare import config_file,environment
from experiments.similarity_graph.data import Window,read_window,contacts,signature,check_schedule
from experiments.similarity_graph.reference import build,resolve
from experiments.similarity_graph.evaluate import geometry,with_identity
from .model import DecoupledModel,frame
from .collaborative import unit
from .search import RawWindow,cached_block_grams,from_grams,raw_utility


def weight_key(weights):return '/'.join(f'{w:g}' for w in weights)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['config','trained','output']:parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();config=config_file(args.config);torch.set_num_threads(2)
    train=[Window(**w) for w in config['train']];dev=[Window(**w) for w in config['development']]
    tests=[Window(**w) for w in config.get('test',[])];check_schedule(train,dev,tests)
    frozen=json.loads((args.trained/'frozen_training.json').read_text())
    if frozen['train']!=[asdict(w) for w in sorted(train,key=lambda w:w.start)]:raise ValueError('training windows changed')
    for w in train:
        s=Path(w.flows).stat();stamp=frozen['training_files'][w.name]
        if (s.st_size,s.st_mtime_ns)!=(stamp['bytes'],stamp['mtime_ns']):raise ValueError('training input changed')
    if frozen['training_end']>min(w.start for w in dev):raise ValueError('training overlaps development')
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    hosts=frozen['hosts'];read=lambda w:read_window(w,hosts,config['cache'])
    source_obs=read(Window(**config['source_window']));development=[read(w) for w in dev]
    def reference(obs):
        output=args.output/'references'/obs.window.name
        if obs.window.name in config.get('references',{}):
            return resolve(config['references'][obs.window.name],source_obs,obs,hosts,output,config)
        build(source_obs,obs,hosts,output,config['cache']);return output
    refs={o.window.name:reference(o) for o in development}
    raw={o.window.name:RawWindow(contacts(source_obs),o,hosts,refs[o.window.name]) for o in development}
    weights=config.get('weights',{'identity':[0.,.15,.4],'partner':[0.,.3,.8],'graph':[0.,.1]})
    grid=[(1.,i,p,g) for i,p,g in itertools.product(weights['identity'],weights['partner'],weights['graph'])]
    seeds=[row['seed'] for row in frozen['seeds']];raw_records=[]
    for stamp in frozen['seeds']:
        seed=stamp['seed'];path=args.trained/f'seed{seed}'/'blocks.joblib'
        if signature(path)!=stamp['checkpoint_sha256']:raise ValueError('block checkpoint changed')
        payload=joblib.load(path);fp=payload['fingerprint'];base=payload['base']
        for partner_kind in ['partner','cf_initial']:
            blocks=[base.codes,fp.bank,unit(payload[partner_kind]),unit(payload['graph'])]
            if sum(b.shape[1] for b in blocks)>1024:raise ValueError('requested dimensions exceed 1024')
            results={weight_key(w):{} for w in grid}
            for obs in development:
                grams,norms=cached_block_grams(blocks,raw[obs.window.name].population)
                for w in grid:
                    results[weight_key(w)][obs.window.name]=raw[obs.window.name].evaluate(from_grams(grams,norms,w))
                del grams,norms
            for w in grid:
                rr=results[weight_key(w)]
                raw_records.append({'seed':seed,'weights':w,'partner_kind':partner_kind,'results':rr,
                                    'raw_utility':raw_utility(list(rr.values()))})
            print('raw grid complete',seed,partner_kind,flush=True)
        del payload,blocks
        (args.output/'raw_development.json').write_text(json.dumps(raw_records,indent=2)+'\n')
    aggregated=[]
    for kind,w in itertools.product(['partner','cf_initial'],grid):
        rows=[r for r in raw_records if r['partner_kind']==kind and tuple(r['weights'])==w]
        aggregated.append({'weights':w,'partner_kind':kind,'utility':float(np.mean([r['raw_utility'] for r in rows])),
            'passes_guard':all(v['consistency@10_mean']>=config.get('min_raw_neighbors',.222) for r in rows for v in r['results'].values())})
    eligible=[r for r in aggregated if r['passes_guard']]
    if not eligible and not config.get('allow_infeasible_diagnostic',False):
        (args.output/'infeasible.json').write_text(json.dumps(aggregated,indent=2)+'\n')
        raise ValueError('no candidate meets strict raw Neighbors threshold; see infeasible.json')
    pool=eligible or aggregated
    shortlist=sorted(pool,key=lambda r:-r['utility'])[:config.get('shortlist_size',4)]
    # Fixed block/dimension controls, independent of development outcomes.
    for w in [(1.,0.,0.,0.),(1.,.15,0.,0.),(1.,0.,.3,0.),(1.,0.,0.,.1),(1.,.15,.3,.1)]:
        if w in grid and not any(tuple(r['weights'])==w and r['partner_kind']=='partner' for r in shortlist):
            shortlist.append(next(r for r in aggregated if tuple(r['weights'])==w and r['partner_kind']=='partner'))
    records=[];controls=[]
    for seed in seeds:
        payload=joblib.load(args.trained/f'seed{seed}'/'blocks.joblib');base=payload['base'];fp=payload['fingerprint']
        # True unchanged codebook+Ridge; plus control uses EXACTLY the same history.
        if seed==seeds[0]:
            for label in ['codebook_ridge','plus_as_is']:
                if label=='plus_as_is' and base.traffic_model is None:continue
                results={}
                source=base.codes if label=='codebook_ridge' else base.traffic_model.source
                for obs in development:
                    late=base.late(obs) if label=='codebook_ridge' else base.traffic_model.late(obs,.5)
                    core=geometry(frame(source,hosts),refs[obs.window.name],config['probe_reference'],config['cache'])
                    results[obs.window.name]=with_identity(frame(source,hosts),frame(late,hosts),refs[obs.window.name],core)
                controls.append({'name':label,'results':results})
                print('unchanged control',label,[r['score'] for r in results.values()],flush=True)
        for candidate in shortlist:
            w=tuple(candidate['weights']);kind=candidate['partner_kind']
            model=DecoupledModel(base,fp,payload[kind],payload['graph'],w)
            source=frame(model.source,hosts)
            cores={o.window.name:geometry(source,refs[o.window.name],config['probe_reference'],config['cache']) for o in development}
            # Hard retrieval and the proposed soft20/0.02 receive the SAME evaluation.
            for retrieval in config.get('retrieval',[{'k':1,'temperature':.02},{'k':20,'temperature':.02}]):
                model.k=retrieval['k'];model.temperature=retrieval['temperature']
                results={}
                for obs in development:
                    results[obs.window.name]=with_identity(source,frame(model.predict(obs),hosts),refs[obs.window.name],cores[obs.window.name])
                row={'seed':seed,'weights':w,'partner_kind':kind,**retrieval,'results':results,
                     'passes_guard':all(r['metrics']['consistency@10_mean']>=config.get('min_raw_neighbors',.222) for r in results.values()),
                     'raw_utility':raw_utility([r['metrics'] for r in results.values()])}
                records.append(row)
                print('full development',seed,w,kind,retrieval,'score',np.mean([r['score'] for r in results.values()]),flush=True)
                (args.output/'development.json').write_text(json.dumps({'candidates':records,'controls':controls},indent=2)+'\n')
    recipes=[]
    for c in shortlist:
        for retrieval in config.get('retrieval',[{'k':1,'temperature':.02},{'k':20,'temperature':.02}]):
            rows=[r for r in records if tuple(r['weights'])==tuple(c['weights']) and r['partner_kind']==c['partner_kind']
                  and r['k']==retrieval['k'] and r['temperature']==retrieval['temperature']]
            recipes.append({'weights':c['weights'],'partner_kind':c['partner_kind'],**retrieval,
                            'raw_utility':float(np.mean([r['raw_utility'] for r in rows])),
                            'passes_guard':all(r['passes_guard'] for r in rows)})
    allowed=[r for r in recipes if r['passes_guard']]
    best=max(allowed or recipes,key=lambda r:r['raw_utility'])
    selected_models={};source_shas={}
    for seed in seeds:
        p=joblib.load(args.trained/f'seed{seed}'/'blocks.joblib')
        model=DecoupledModel(p['base'],p['fingerprint'],p[best['partner_kind']],p['graph'],best['weights'],best['k'],best['temperature'])
        folder=args.output/f'seed{seed}';folder.mkdir()
        joblib.dump(model,folder/'model.joblib',compress=3);frame(model.source,hosts).to_csv(folder/'source.csv')
        selected_models[str(seed)]=signature(folder/'model.joblib');source_shas[str(seed)]=signature(folder/'source.csv')
    selection={'recipe':best,'seeds':seeds,'delivery_seed':seeds[0],
               'checkpoint_sha256':selected_models,'source_sha256':source_shas,
               'selection':'mean five-seed geometric mean of five uncapped RAW metrics, no test access',
               'strict_constraint_passed':bool(best['passes_guard']),'min_raw_neighbors':config.get('min_raw_neighbors',.222),
               'development_test_independent':False,'new_closed_holdout_available':False,
               'training_end':frozen['training_end'],'test_not_read_before_freeze':True,
               'config':config,'versions':environment(),
               'code_sha256':{p.name:signature(p) for p in Path(__file__).parent.glob('*.py') if not p.name.startswith('test_')}}
    (args.output/'frozen_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    report={'selection':selection,'development':records,'controls':controls,'test':{},'official_leaderboard_result':False}
    # NO test flow/statistics/labels above this line.
    for w in tests:
        obs=read(w);ref=reference(obs);report['test'][w.name]={}
        for seed in seeds:
            folder=args.output/f'seed{seed}';model=joblib.load(folder/'model.joblib')
            source=frame(model.source,hosts);late=frame(model.predict(obs),hosts)
            core=geometry(source,ref,config['probe_reference'],config['cache'],100)
            report['test'][w.name][str(seed)]=with_identity(source,late,ref,core,100)
            late.to_csv(folder/f'{w.name}.csv')
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('finished',args.output,'strict guard:',selection['strict_constraint_passed'],flush=True)


if __name__=='__main__':main()
