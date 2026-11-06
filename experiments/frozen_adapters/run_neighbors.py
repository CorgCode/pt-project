"""Полезные замены доноров → поправка source. Полный score — только с согласованными эталонами."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from evaluation import scoreboard, submit
from evaluation.evaluate import run as evaluate
from evaluation.partners import partner_ranking
from evaluation.knn_consistency import evaluate_embedding
from evaluation.frozen_neighbors import evaluate_frozen_neighbors
from .data import Window, read_window, contacts, check_schedule, signature
from .identity import unit
from .neighbors import ContactForecaster, contact_matrix, swap_teacher, fit_residual, apply_residual, assess_forecast
from .run import load_config, baseline
from .protocol import record


def partner_frame(source_rows,target_rows,hosts,population):
    records=[]
    for window,rows in [('source',source_rows),('target',target_rows)]:
        for index in population:
            records.extend((hosts[index],p,window,c) for p,c in rows[index].items())
    return pd.DataFrame(records,columns=['host_id','partner','window','count'])


def transfer_late(original,source,updated,k=20,temperature=0.02):
    """Перенос только поправки к source, без label позднего хоста."""
    out=np.asarray(original).copy()
    base=normalize(source)
    for start in range(0,len(out),128):
        stop=min(start+128,len(out))
        sim=normalize(original[start:stop])@base.T
        top=np.argsort(-sim,axis=1,kind='stable')[:,:min(k,len(source))]
        values=np.take_along_axis(sim,top,axis=1)
        weights=np.exp((values-values[:,:1])/temperature)
        weights/=weights.sum(1,keepdims=True)
        out[start:stop]+=np.einsum('bk,bkd->bd',weights,(updated-source)[top])
    return out


def raw_behavior(frame,path):
    if not path: return None
    reference=pd.read_csv(path,index_col=0)
    reference.index=reference.index.astype(str)
    hosts=frame.index.intersection(reference.index).sort_values()
    directory=Path(path).parent
    exact=(len(hosts)==len(reference) and (directory/'neighbors.csv').exists() and (directory/'manifest.json').exists())
    if exact:
        split=json.loads((directory/'manifest.json').read_text()).get('split','test')
        result=evaluate_frozen_neighbors(frame.loc[hosts],directory/'neighbors.csv',ks=(10,),split=split)
    else:
        result=evaluate_embedding(frame.loc[hosts],reference.loc[hosts],ks=(10,))
    return {'neighbors_at_10':float(result.loc[(10,'consistency'),'mean']),
            'n_hosts':len(hosts),'reference':str(path),
            'population':'exact frozen reference' if exact else 'intersection, local diagnostic'}


def evaluate_candidate(frame,original,updated,source_rows,obs,config,out,adapter,selection,bootstrap=0):
    hosts=list(frame.index)
    population=np.flatnonzero(obs.events>=300)
    partners=partner_frame(source_rows,contacts(obs),hosts,population)
    source_frame=pd.DataFrame(updated,index=frame.index,columns=frame.columns)
    raw=partner_ranking(source_frame.iloc[population],partners,n_boot=bootstrap)
    ref=config.get('references',{}).get(obs.window.name)
    behavior_path=config.get('behavior_features',{}).get(obs.window.name)
    if not behavior_path and ref: behavior_path=str(Path(ref)/'features.csv')
    result={'new_partners_mrr':raw['new']['embedding']['mrr'],'partners':raw,
            'behavior':raw_behavior(source_frame,behavior_path),
            'full_score_available':False}
    if ref:
        reference_hosts=set(pd.read_csv(Path(ref)/'eval_hosts.csv',dtype=str)['host_id'])
        if reference_hosts!=set(frame.index[population]):
            raise ValueError('эвалуационная когорта reference не совпадает с активной source-когортой окна')
        reference_partners=pd.read_csv(Path(ref)/'partners.csv',dtype={'host_id':str,'partner':str,'window':str})
        def source_counts(p):
            return p[p.window=='source'].groupby(['host_id','partner'])['count'].sum().sort_index()
        expected,actual=source_counts(partners),source_counts(reference_partners)
        if not expected.index.equals(actual.index) or not np.array_equal(expected.to_numpy(),actual.to_numpy()):
            raise ValueError('source-контакты reference отличаются от source_window эксперимента')
        out.mkdir(parents=True,exist_ok=True)
        if not config.get('probe_reference'):
            raise ValueError('для Links нужен probe_reference более раннего окна с тем же source')
        fit_ref=config['probe_reference']
        combined={}
        for name,use_updated,use_adapter in [('baseline',False,False),('identity_only',False,True),
                                            ('neighbors_only',True,False),('combined',True,True)]:
            if original is None: raise ValueError('для четырёх сравнений нужны исходные later-векторы')
            if adapter is None: raise ValueError('для четырёх сравнений нужен обученный IdentityAdapter')
            train=updated if use_updated else frame.to_numpy()
            late=adapter.predict(obs,selection['variant'],selection['temperature']) if use_adapter else original
            if use_updated: late=transfer_late(late,frame.to_numpy(),updated)
            directory=out/name;directory.mkdir()
            a,b=directory/'embeddings.csv',directory/'later.csv'
            pd.DataFrame(train,index=frame.index,columns=frame.columns).to_csv(a,index_label='host_id')
            pd.DataFrame(late,index=frame.index,columns=frame.columns).to_csv(b,index_label='host_id')
            r=evaluate(a,Path(ref),name,n_boot=bootstrap,fit_reference_dir=Path(fit_ref))
            submit.add_identity(r,a,b,Path(ref),shared=True,n_boot=bootstrap)
            cols=['consistency@10_mean','link_learned_roc_auc','partner_new_mrr','pair_roc_auc','identity_top1']
            if not scoreboard.complete(r,cols): raise ValueError('неполная оценка: нужны все пять обязательных метрик')
            ceiling=config.get('neighbor_ceilings',{}).get(obs.window.name)
            window=config.get('score_windows',{}).get(obs.window.name)
            if ceiling is not None:
                if not 0<float(ceiling)<=1: raise ValueError('некорректный потолок Neighbors')
                window='custom_'+obs.window.name
                previous=scoreboard.NEIGHBORS_IDEAL.copy()
                try:
                    scoreboard.NEIGHBORS_IDEAL[window]=float(ceiling)
                    score=scoreboard._one_score(r,cols,window=window)
                finally:
                    scoreboard.NEIGHBORS_IDEAL.clear();scoreboard.NEIGHBORS_IDEAL.update(previous)
            else:
                if window not in scoreboard.NEIGHBORS_IDEAL or (window=='validation' and obs.window.day!=2):
                    raise ValueError('укажите правильный потолок neighbor_ceilings: validation дня 13 не равен validation дня 02')
                score=scoreboard._one_score(r,cols,window=window)
            combined[name]={'score':score,'result':r,'neighbor_ceiling':ceiling or scoreboard.NEIGHBORS_IDEAL[window]}
        result['four_configurations']=combined
        result['full_score_available']=True
        result['selection_score']=combined['combined']['score']
    else:
        result['selection_score']=result['new_partners_mrr']
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',required=True,type=Path)
    ap.add_argument('--identity-run',type=Path)
    ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args()
    config=load_config(args.config)
    windows=[Window(**w) for w in config['train']]
    dev=[Window(**w) for w in config['development']]
    tests=[Window(**w) for w in config.get('test',[])]
    check_schedule(windows,dev,tests)
    if config.get('source_training_end',-np.inf)>min(w.start for w in dev):
        raise ValueError('замороженный source обучался после начала development')
    if args.output.exists(): raise FileExistsError('нужна новая папка')
    args.output.mkdir(parents=True)
    record(args.output,config)
    source=pd.read_csv(config['source'],index_col=0)
    source.index=source.index.astype(str)
    hosts=list(source.index)
    original_source=source.to_numpy().copy()
    frozen_hash=signature(config['source'])
    cache=Path(config.get('cache',args.output/'cache'))
    def read(w):
        print('Чтение',w.name,flush=True)
        return read_window(w,hosts,cache)
    history=[read(w) for w in windows]
    source_observation=read(Window(**config['source_window'])) if config.get('source_window') else history[0]
    if source_observation.window.end>max(o.window.end for o in history):
        raise ValueError('окно контактов source выходит за обучающий cutoff')
    source_rows=contacts(source_observation)
    vocabulary=sorted({p for row in source_rows for p in row})
    matrix=contact_matrix(source_rows,vocabulary)
    forecaster=ContactForecaster(source_rows,vocabulary,max_queries=config.get('teacher_queries',128)).fit(history)
    triples,teacher=swap_teacher(original_source,matrix,vocabulary,source_rows,hosts,forecaster,
                                max_queries=config.get('teacher_queries',128),
                                proposal_k=config.get('teacher_proposals',16))
    np.save(args.output/'teacher_triples.npy',triples)
    (args.output/'teacher.json').write_text(json.dumps(teacher,ensure_ascii=False,indent=2)+'\n')
    print('Полезных замен',len(triples),flush=True)
    adapter=selection=None
    if args.identity_run:
        frozen=json.loads((args.identity_run/'frozen_selection.json').read_text())
        if frozen['source_sha256']!=frozen_hash: raise ValueError('IdentityAdapter обучен для другого source')
        adapter=joblib.load(args.identity_run/'adapter.joblib')
        selection=frozen['selection']
    state=fit_residual(original_source,triples,steps=config.get('neighbor_steps',150)) if len(triples) else None
    strengths=config.get('neighbor_strengths',[0.0,0.1,0.2,0.4]) if state else [0.0]
    if 0.0 not in strengths:
        raise ValueError('сетка должна включать нулевой контроль')
    candidates=[]
    development=[read(w) for w in dev]
    baseline_behavior={}
    for obs in development:
        path=config.get('behavior_features',{}).get(obs.window.name)
        if not path and config.get('references',{}).get(obs.window.name):
            path=str(Path(config['references'][obs.window.name])/'features.csv')
        baseline_behavior[obs.window.name]=raw_behavior(source,path)
    for strength in strengths:
        updated=apply_residual(original_source,state,strength) if state else original_source.copy()
        rows=[]
        for obs in development:
            original=(baseline(obs,source,**config.get('baseline_checkpoint',{}).get(obs.window.name,{}))
                      if config.get('references',{}).get(obs.window.name) else None)
            assessment=evaluate_candidate(source,original,updated,source_rows,obs,config,
                args.output/f'dev_{strength}_{obs.window.name}',adapter,selection)
            reference=baseline_behavior[obs.window.name]
            assessment['passes_behavior_guard']=(reference is None or
                assessment['behavior']['neighbors_at_10']>=reference['neighbors_at_10']*config.get('min_behavior_retention',0.95))
            rows.append(assessment)
        row={'strength':strength,'selection_score':float(np.mean([r['selection_score'] for r in rows])),
             'passes_guard':all(r['passes_behavior_guard'] for r in rows),'development':rows}
        candidates.append(row)
        print('development',strength,row['selection_score'],row['passes_guard'],flush=True)
    accepted=[r for r in candidates if r['passes_guard']]
    if not accepted: raise ValueError('даже нулевой контроль не прошёл behavior guard')
    best=sorted(accepted,key=lambda r:(-r['selection_score'],r['strength']))[0]
    freeze={'strength':best['strength'],'source_sha256':frozen_hash,'test_not_read_before_freeze':True,
            'selection_objective':'full combined score if references supplied, otherwise diagnostic New partners MRR',
            'train':[asdict(w) for w in windows],'development':[asdict(w) for w in dev]}
    joblib.dump({'state':state,'source_sha256':frozen_hash,'hosts':hosts,
                 'selected_strength':best['strength']},args.output/'neighbor_head.joblib',compress=3)
    freeze['head_sha256']=signature(args.output/'neighbor_head.joblib')
    (args.output/'frozen_selection.json').write_text(json.dumps(freeze,ensure_ascii=False,indent=2)+'\n')
    report={'teacher':teacher,'validation':candidates,'selected':freeze,'test':{},'forecast':{},
            'test_policy':config.get('test_policy'),'is_official_submission':False}
    updated=apply_residual(original_source,state,best['strength']) if state else original_source.copy()
    pd.DataFrame(updated,index=source.index,columns=source.columns).to_csv(args.output/'updated_source.csv',index_label='host_id')
    for window in tests:
        obs=read(window)
        report['forecast'][window.name]=assess_forecast(original_source,matrix,vocabulary,source_rows,hosts,forecaster,obs)
        original=(baseline(obs,source,**config.get('baseline_checkpoint',{}).get(obs.window.name,{}))
                  if config.get('references',{}).get(obs.window.name) else None)
        report['test'][window.name]={
            'baseline':evaluate_candidate(source,original,original_source,source_rows,obs,config,
                args.output/f'test_baseline_{window.name}',adapter,selection,100),
            'selected':evaluate_candidate(source,original,updated,source_rows,obs,config,
                args.output/f'test_selected_{window.name}',adapter,selection,100)}
        print('test',window.name,{k:v['new_partners_mrr'] for k,v in report['test'][window.name].items()},flush=True)
    if signature(config['source'])!=frozen_hash: raise RuntimeError('исходный файл source изменён')
    report['original_source_file_unchanged']=True
    report['candidate_source_changes_other_tasks']=bool(best['strength'])
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if config.get('refit_after_diagnostic'):
        final=[read(Window(**w)) for w in config['refit_after_diagnostic']]
        if any(o.window.end>config['delivery_training_end'] for o in final): raise ValueError('нарушен cutoff поставляемой головы')
        final_forecaster=ContactForecaster(source_rows,vocabulary,max_queries=config.get('teacher_queries',128)).fit(final)
        final_triples,final_teacher=swap_teacher(original_source,matrix,vocabulary,source_rows,hosts,final_forecaster,
            max_queries=config.get('teacher_queries',128),proposal_k=config.get('teacher_proposals',16))
        final_state=fit_residual(original_source,final_triples,steps=config.get('neighbor_steps',150)) if len(final_triples) else None
        joblib.dump({'state':final_state,'source_sha256':frozen_hash,'hosts':hosts,
                    'selected_strength':best['strength']},args.output/'delivery_neighbor_head.joblib',compress=3)
        (args.output/'delivery_protocol.json').write_text(json.dumps({'train':[asdict(o.window) for o in final],
            'strength':best['strength'],'teacher':final_teacher,'trained_after_local_diagnostic':True,
            'has_independent_test_result':False},ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
