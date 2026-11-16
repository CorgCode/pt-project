"""Frozen plus → Identity; final-source adaptation; full local five-metric selection."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from evaluation import scoreboard, submit
from evaluation.evaluate import run as evaluate
from .data import Window, read_window, contacts, check_schedule, signature
from .identity import IdentityAdapter
from .neighbors import (contact_matrix, ContactForecaster, swap_teacher, fit_residual,
                        apply_residual, assess_forecast)
from .plus_reference import build as build_reference
from .run import load_config, assess
from .run_neighbors import raw_behavior, transfer_late
from .protocol import record
from .codebook import PlusDelivery, predict_identity

REQUIRED=['consistency@10_mean','link_learned_roc_auc','partner_new_mrr','pair_roc_auc','identity_top1']


def score(result, ceiling):
    if not scoreboard.complete(result,REQUIRED): raise ValueError('five mandatory metrics required')
    old=scoreboard.NEIGHBORS_IDEAL.copy()
    try:
        scoreboard.NEIGHBORS_IDEAL['local_plus']=ceiling
        return scoreboard._one_score(result,REQUIRED,window='local_plus')
    finally:
        scoreboard.NEIGHBORS_IDEAL.clear();scoreboard.NEIGHBORS_IDEAL.update(old)


def frame(values, hosts):
    return pd.DataFrame(values,index=pd.Index(hosts,name='host_id')).add_prefix('emb_')


def select_identity(source, adapter, baseline_late, development):
    """Baseline can win. Select only on development, never on the diagnostic test."""
    candidates=[('baseline',.02),('ridge',.02),('rare',.02),('rare',.05),('rare',.1),
                ('gate',.05),('baseline_mix:.25',.02),('baseline_mix:.5',.02),('baseline_mix:.75',.02)]
    rows=[]
    for variant,temperature in candidates:
        predictions=[predict_identity(adapter,b,obs,variant,temperature) for obs,b in zip(development,baseline_late)]
        metrics=[assess(source,p,obs) for obs,p in zip(development,predictions)]
        row={'variant':variant,'temperature':temperature,
             'official_top1':float(np.mean([m['official']['top1'] for m in metrics])),
             'direct_late_to_source_top1':float(np.mean([m['late_to_source_full_gallery']['top1'] for m in metrics]))}
        rows.append(row);print('identity development',row,flush=True)
    # Prefer unchanged baseline on exact ties.
    best=min(rows,key=lambda r:(-r['official_top1'],r['variant']!='baseline',r['variant'],r['temperature']))
    return {k:best[k] for k in ('variant','temperature')},rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--trained',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    config=load_config(args.config)
    windows=sorted([Window(**w) for w in config['train']],key=lambda w:w.start)
    dev_windows=[Window(**w) for w in config['development']]
    test_windows=[Window(**w) for w in config.get('test',[])]
    check_schedule(windows,dev_windows,test_windows)
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    record(args.output,config)
    torch.set_num_threads(2)
    training=json.loads((args.trained/'frozen_training.json').read_text())
    if signature(args.trained/'plus.joblib')!=training['checkpoint_sha256']: raise ValueError('changed checkpoint')
    if training['train']!=[asdict(w) for w in windows]: raise ValueError('training configuration mismatch')
    recorded=json.loads((args.trained/'protocol.json').read_text())['files']
    for w in windows:
        stat=Path(w.flows).stat()
        if (stat.st_size,stat.st_mtime_ns)!=(recorded[w.name]['bytes'],recorded[w.name]['mtime_ns']):
            raise ValueError('training flows changed after fit; train a new checkpoint')
    base=joblib.load(args.trained/'plus.joblib')
    if base.training_end>min(w.start for w in dev_windows): raise ValueError('base trained after development')
    hosts=base.hosts
    source=pd.read_csv(args.trained/'source.csv',index_col=0)
    source.index=source.index.astype(str)
    np.testing.assert_allclose(base.source,source.to_numpy(),rtol=1e-6,atol=1e-8)
    if signature(args.trained/'source.csv')!=training['source_sha256']: raise ValueError('changed source')
    read=lambda w:read_window(w,hosts,config['cache'])
    observations=[read(w) for w in windows]
    development=[read(w) for w in dev_windows]
    print('development observations ready',flush=True)
    source_obs=read(Window(**config.get('source_window',config['train'][0])))
    trained_source=training.get('source_window',training['train'][0])
    if ((source_obs.window.start,source_obs.window.end)!=(trained_source['start'],trained_source['end']) or
        Path(source_obs.window.flows).resolve()!=Path(trained_source['flows']).resolve()):
        raise ValueError('source_window differs from the trained canonical source')
    refs={}
    for obs in development:
        path=args.output/'references'/obs.window.name
        refs[obs.window.name]=(path,build_reference(source_obs,obs,hosts,path,config['cache']))
        print('reference ready',obs.window.name,refs[obs.window.name][1]['neighbor_ceiling'],flush=True)

    core_cache={}
    def full(values,later,obs,name,bootstrap=0):
        out=args.output/'evaluation'/obs.window.name/name
        out.mkdir(parents=True,exist_ok=True)
        a,b=out/'source.csv',out/'late.csv'
        frame(values,hosts).to_csv(a);frame(later,hosts).to_csv(b)
        ref,meta=refs[obs.window.name]
        key=(joblib.hash(values),obs.window.name)
        if key not in core_cache:
            core_cache[key]=evaluate(a,ref,name,n_boot=bootstrap,fit_reference_dir=Path(config['probe_reference']))
        result=copy.deepcopy(core_cache[key]);result['method']=name
        submit.add_identity(result,a,b,ref,shared=True,n_boot=bootstrap)
        row={'score':score(result,meta['neighbor_ceiling']),'metrics':result['metrics'],
             'chance':{s['metric']:s['chance'] for s in result['summary']}}
        # task_scores need the same explicit calibration as score, not default validation.
        old=scoreboard.NEIGHBORS_IDEAL.copy()
        try:
            scoreboard.NEIGHBORS_IDEAL['local_plus']=meta['neighbor_ceiling']
            row['task_scores']=scoreboard.task_scores(result,REQUIRED,window='local_plus')
        finally:
            scoreboard.NEIGHBORS_IDEAL.clear();scoreboard.NEIGHBORS_IDEAL.update(old)
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print('full score',obs.window.name,name,row['score'],flush=True)
        return row

    blends=[]
    for blend in (0.,.5,1.):
        results=[full(source.to_numpy(),base.late(obs,blend),obs,f'baseline_blend_{blend:g}') for obs in development]
        blends.append({'blend':blend,'mean_score':float(np.mean([r['score'] for r in results]))})
    blend=min(blends,key=lambda r:(-r['mean_score'],r['blend']))['blend']
    baseline_dev=[base.late(obs,blend) for obs in development]
    adapter=IdentityAdapter(source.to_numpy(),**config.get('model',{})).fit(observations).fit_gate(observations)
    selection,identity_rows=select_identity(source,adapter,baseline_dev,development)
    baseline_scores=[full(source.to_numpy(),b,obs,'baseline') for obs,b in zip(development,baseline_dev)]
    identity_scores=[full(source.to_numpy(),predict_identity(adapter,b,obs,**selection),obs,'identity_only')
                     for obs,b in zip(development,baseline_dev)]
    # Fit the neighbor teacher only to historical targets. Refit Identity to EACH final geometry.
    source_rows=contacts(source_obs)
    vocabulary=sorted(set().union(*(set(r) for r in source_rows)))
    matrix=contact_matrix(source_rows,vocabulary)
    forecaster=ContactForecaster(source_rows,vocabulary,max_queries=config.get('teacher_queries',128)).fit(observations)
    triples,teacher=swap_teacher(source.to_numpy(),matrix,vocabulary,source_rows,hosts,forecaster,
        max_queries=config.get('teacher_queries',128),proposal_k=8)
    state=fit_residual(source.to_numpy(),triples,steps=150) if len(triples) else None
    candidates=[{'strength':0.,'selection':selection,'scores':identity_scores,'passes_guard':True,
                 'mean_score':float(np.mean([r['score'] for r in identity_scores]))}]
    fitted={0.:(adapter,selection)}
    for strength in config.get('residual_strengths',[0.,.1,.2]):
        if not strength or state is None: continue
        values=apply_residual(source.to_numpy(),state,strength)
        updated=frame(values,hosts)
        # Deliberately fit against the final vectors, not a transferred old Identity head.
        new_adapter=IdentityAdapter(values,**config.get('model',{})).fit(observations).fit_gate(observations)
        late=[transfer_late(b,source.to_numpy(),values) for b in baseline_dev]
        selected,rows=select_identity(updated,new_adapter,late,development)
        checks=[raw_behavior(updated,str(refs[o.window.name][0]/'features.csv')) for o in development]
        passes=all(c['neighbors_at_10']>=config.get('min_behavior_retention',.95)*r['metrics']['consistency@10_mean']
                   for c,r in zip(checks,baseline_scores))
        scores=[full(values,predict_identity(new_adapter,b,obs,**selected),obs,f'combined_strength_{strength:g}')
                for obs,b in zip(development,late)]
        candidates.append({'strength':strength,'selection':selected,'identity_candidates':rows,
                           'scores':scores,'passes_guard':passes,'behavior':checks,
                           'mean_score':float(np.mean([r['score'] for r in scores]))})
        fitted[strength]=(new_adapter,selected)
    best=min((r for r in candidates if r['passes_guard']),key=lambda r:(-r['mean_score'],r['strength']))
    strength=best['strength']
    values=apply_residual(source.to_numpy(),state,strength) if strength else source.to_numpy()
    chosen_adapter,chosen_selection=fitted[strength]
    delivery=PlusDelivery(base,chosen_adapter,chosen_selection,blend,values,state,strength)
    identity_delivery=PlusDelivery(base,adapter,selection,blend,source.to_numpy())
    joblib.dump(delivery,args.output/'selected.joblib',compress=3)
    joblib.dump(identity_delivery,args.output/'identity_only.joblib',compress=3)
    frame(values,hosts).to_csv(args.output/'source.csv')
    frozen={'blend':blend,'identity_selection':selection,'selected_strength':strength,
            'selected_identity':chosen_selection,'test_not_read_before_freeze':True,
            'source_sha256':signature(args.output/'source.csv'),
            'checkpoint_sha256':signature(args.output/'selected.joblib'),
            'train':[asdict(w) for w in windows],'development':[asdict(w) for w in dev_windows],
            'selection_objective':'full five-metric development score with raw Behavior guard',
            'test_policy':config['test_policy']}
    (args.output/'frozen_selection.json').write_text(json.dumps(frozen,indent=2)+'\n')
    report={'training':training,'baseline_blends':blends,'identity_candidates':identity_rows,
            'baseline_development':baseline_scores,'identity_development':identity_scores,
            'neighbor_teacher_train':teacher,'candidates':candidates,'selection':frozen,
            'test':{},'is_official_scoreboard_result':False}
    for w in test_windows:
        obs=read(w)  # No test statistics, features, profiles or labels above this boundary.
        ref=args.output/'references'/w.name
        refs[w.name]=(ref,build_reference(source_obs,obs,hosts,ref,config['cache']))
        original=base.late(obs,blend)
        identity=identity_delivery.predict(obs)
        transferred=transfer_late(original,source.to_numpy(),values) if strength else original
        configurations={
            'baseline':full(source.to_numpy(),original,obs,'baseline',100),
            'identity_only':full(source.to_numpy(),identity,obs,'identity_only',100),
            'neighbors_only':full(values,transferred,obs,'neighbors_only',100),
            'combined_refitted_identity':full(values,delivery.predict(obs),obs,'combined_refitted_identity',100)}
        # Separate diagnostic of a nonzero candidate, even when the zero-weight control wins.
        direct=assess_forecast(source.to_numpy(),matrix,vocabulary,source_rows,hosts,forecaster,obs)
        diagnostic=copy.copy(forecaster);diagnostic.folds=[(obs,forecaster.history,forecaster.model)]
        _,oracle=swap_teacher(source.to_numpy(),matrix,vocabulary,source_rows,hosts,diagnostic,
                             max_queries=128,proposal_k=8,seed=31)
        np.testing.assert_allclose(oracle[0]['baseline_mrr'],direct['source_embedding_mrr_same_queries'],atol=1e-12)
        np.testing.assert_allclose(oracle[0]['direct_forecast_mrr'],direct['direct_forecast_mrr'],atol=1e-12)
        direct['forced_swap_oracle_same_queries']=oracle[0]
        direct['selected_embedding_same_queries']=assess_forecast(values,matrix,vocabulary,source_rows,hosts,forecaster,obs)['source_embedding_mrr_same_queries']
        report['test'][w.name]={'four_configurations':configurations,'forecast_levels':direct,
                              'neighbor_ceiling':refs[w.name][1]['neighbor_ceiling']}
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    if signature(args.trained/'source.csv')!=training['source_sha256']: raise RuntimeError('base source changed')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('completed',args.output,flush=True)


if __name__=='__main__':
    main()
