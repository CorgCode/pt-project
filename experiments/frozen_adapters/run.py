"""Пример: python -m experiments.frozen_adapters.run --config config.json --output runs/identity01."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from evaluation.identity import identity_retrieval
from .data import Window, read_window, check_schedule, signature
from .identity import IdentityAdapter
from .protocol import record


def load_config(path):
    config = json.loads(Path(path).read_text())
    def expand(value):
        if isinstance(value, str): return os.path.expandvars(value)
        if isinstance(value, list): return [expand(x) for x in value]
        if isinstance(value, dict): return {k: expand(v) for k,v in value.items()}
        return value
    return expand(config)


def assess(source, later, observation, min_events=300, bootstrap=0, activity_threshold=None):
    active = np.flatnonzero(observation.events >= min_events)
    hosts = source.index[active]
    if len(active) < 20:
        raise ValueError('слишком мало активных хостов для Identity')
    a = source.loc[hosts]
    b = pd.DataFrame(later[active], index=hosts, columns=source.columns)
    official = identity_retrieval(a,b,alignment='procrustes',n_boot=bootstrap,seed=0)
    direct = identity_retrieval(a,b,alignment='none',eval_fraction=1.0,n_boot=bootstrap,seed=0)
    reverse = identity_retrieval(b,a,alignment='none',eval_fraction=1.0,n_boot=bootstrap,seed=0)
    # Подгруппы используют одну и ту же полную галерею, а не облегченную галерею подгруппы.
    sim = normalize(b.to_numpy()) @ normalize(a.to_numpy()).T
    truth = np.diag(sim)[:,None]
    greater = (sim > truth).sum(1)
    ties = np.isclose(sim,truth,rtol=0,atol=1e-12).sum(1)
    hits = np.clip(1-greater,0,ties)/ties
    groups = {}
    if activity_threshold is not None:
        for name, mask in [('low_activity',observation.events[active] <= activity_threshold),
                           ('high_activity',observation.events[active] > activity_threshold)]:
            groups[name] = {'n_queries':int(mask.sum()), 'gallery_size':len(hosts),
                            'late_to_source_top1':float(hits[mask].mean()) if mask.any() else None}
    return {'official':official,'direct_full_gallery':direct,'late_to_source_full_gallery':reverse,
            'activity':groups,'n_active':len(active),
            'directions':{'official':'source query -> late gallery, split-half Procrustes',
                          'direct_full_gallery':'source query -> late gallery, all active hosts',
                          'late_to_source_full_gallery':'late query -> source gallery, all active hosts'}}


def baseline(observation, source, checkpoint=None, processed_root=None, window_name=None):
    if observation.window.baseline:
        frame=pd.read_csv(observation.window.baseline,index_col=0)
        frame.index=frame.index.astype(str)
        required=source.index[observation.events >= 300]
        if not set(required) <= set(frame.index): raise ValueError('неполный контрольный later.csv')
        if frame.shape[1] != source.shape[1]: raise ValueError('размерность контроля не совпадает с source')
        return frame.reindex(source.index).fillna(0).to_numpy()
    if checkpoint:
        # Локальный контроль работоспособности; это НЕ Ridge-адаптер настоящего plus.
        from pipeline.data import ProcessedData
        from scoreboard.temporal_stable_hybrid import TemporalStableHybrid
        model=TemporalStableHybrid.load(checkpoint)
        if model.hosts != list(source.index): raise ValueError('контрольный checkpoint другого source')
        return model.embed(ProcessedData(processed_root),window_name,list(source.index)).to_numpy()
    return None


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args()
    config=load_config(args.config)
    train=[Window(**x) for x in config['train']]
    dev=[Window(**x) for x in config['development']]
    tests=[Window(**x) for x in config.get('test',[])]
    check_schedule(train,dev,tests)
    if config.get('source_training_end',-np.inf)>min(w.start for w in dev):
        raise ValueError('замороженный source обучался после начала development')
    if args.output.exists(): raise FileExistsError('нужна новая папка результата')
    args.output.mkdir(parents=True)
    record(args.output,config)
    source_path=Path(config['source'])
    source=pd.read_csv(source_path,index_col=0)
    source.index=source.index.astype(str)
    if not source.index.is_unique or not np.isfinite(source.to_numpy()).all(): raise ValueError('некорректный source')
    before=signature(source_path)
    shutil.copyfile(source_path,args.output/'source.csv')
    cache=Path(config.get('cache',args.output/'cache'))
    def read(w):
        print('Чтение',w.name,flush=True)
        return read_window(w,list(source.index),cache)
    observations=[read(w) for w in train]
    options=config.get('model',{})
    model=IdentityAdapter(source.to_numpy(),**options).fit(observations).fit_gate(observations)
    events=np.concatenate([o.events[o.events >= model.min_events] for o in observations])
    threshold=float(np.median(events))
    candidates=[('ridge',0.05),('rare',0.02),('rare',0.05),('rare',0.1),
                ('mix:0.25',0.05),('mix:0.5',0.05),('mix:0.75',0.05),('gate',0.05)]
    development=[read(w) for w in dev]
    rows=[]
    for variant,temperature in candidates:
        metrics=[assess(source,model.predict(obs,variant,temperature),obs,model.min_events,
                        activity_threshold=threshold) for obs in development]
        row={'variant':variant,'temperature':temperature,
             'official_top1':float(np.mean([m['official']['top1'] for m in metrics])),
             'direct_top1':float(np.mean([m['direct_full_gallery']['top1'] for m in metrics])),
             'windows':metrics}
        rows.append(row)
        print('development',variant,temperature,row['official_top1'],row['direct_top1'],flush=True)
    best=sorted(rows,key=lambda r:(-r['official_top1'],-r['direct_top1'],r['variant'],r['temperature']))[0]
    frozen={'selection':{k:best[k] for k in ('variant','temperature')},'source_sha256':before,
            'train':[asdict(w) for w in train],'development':[asdict(w) for w in dev],
            'test_policy':config.get('test_policy','diagnostic, not an independent holdout'),
            'gate_oof':model.gate_history,'activity_threshold_from_train':threshold,
            'test_not_read_before_freeze':True,'model':options}
    (args.output/'frozen_selection.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2)+'\n')
    model.hosts=list(source.index)
    model.source_sha256=before
    model.selected=frozen['selection']
    joblib.dump(model,args.output/'adapter.joblib',compress=3)
    report={'source_name':config.get('source_name','unspecified'), 'development':rows,
            'selection':frozen['selection'],'test_policy':frozen['test_policy'],'test':{},'controls':{}}
    for obs in development:
        controls=config.get('baseline_checkpoint',{})
        original=baseline(obs,source,**controls.get(obs.window.name,{}))
        if original is not None:
            report['controls'][obs.window.name]=assess(source,original,obs,model.min_events,activity_threshold=threshold)
    for window in tests:
        obs=read(window)  # первая обработка test — строго после freeze
        evaluated={}
        for name in ('ridge','rare','gate',best['variant']):
            temp=best['temperature'] if name==best['variant'] else 0.05
            prediction=model.predict(obs,name,temp)
            evaluated[name]=assess(source,prediction,obs,model.min_events,bootstrap=100,activity_threshold=threshold)
            if name==best['variant']:
                pd.DataFrame(prediction,index=source.index,columns=source.columns).to_csv(args.output/f'{window.name}.csv')
        controls=config.get('baseline_checkpoint',{})
        original=baseline(obs,source,**controls.get(window.name,{}))
        if original is not None:
            evaluated['frozen_source_encoder']=assess(source,original,obs,model.min_events,bootstrap=100,activity_threshold=threshold)
        report['test'][window.name]=evaluated
        print('test',window.name,{k:round(v['official']['top1'],4) for k,v in evaluated.items()},flush=True)
    after=signature(source_path)
    if after != before or signature(args.output/'source.csv') != before:
        raise RuntimeError('source-векторы изменились')
    report['source_unchanged']=True
    report['first_three_tasks_invariant_by_construction']=True
    report['new_score_claimed']=False
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if config.get('refit_after_diagnostic'):
        # Этот checkpoint включает просмотренный локальный test. Ему НЕ приписывается прежняя test-оценка.
        final=[read(Window(**w)) for w in config['refit_after_diagnostic']]
        if any(o.window.end > config['delivery_training_end'] for o in final): raise ValueError('нарушен cutoff финального обучения')
        refit=IdentityAdapter(source.to_numpy(),**options).fit(final).fit_gate(final)
        refit.hosts=list(source.index)
        refit.source_sha256=before
        refit.selected=frozen['selection']
        joblib.dump(refit,args.output/'delivery_adapter.joblib',compress=3)
        (args.output/'delivery_protocol.json').write_text(json.dumps({'train':[asdict(o.window) for o in final],
            'selection':frozen['selection'],'source_sha256':before,'trained_after_local_diagnostic':True,
            'has_independent_test_result':False},ensure_ascii=False,indent=2)+'\n')
    print('Результат',args.output,flush=True)


if __name__=='__main__':
    main()
