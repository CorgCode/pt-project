"""All five mandatory metrics, project functions, explicit local calibration.

The Links calculation alone uses an equivalent streamed objective to bound RAM.
No metric is silently omitted and no published-window ceiling is reused.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from evaluation import scoreboard
from evaluation.frozen_neighbors import evaluate_frozen_neighbors
from evaluation.partners import partner_ranking,load_partners
from evaluation.retrieval import load_pairs
from evaluation.io import cosine_pairs,select_split
from evaluation.uncertainty import auc_with_ci,host_index
from evaluation.future_links import load_candidate_edges
from evaluation.identity import identity_retrieval
from .streaming_links import score_learned
from .data import signature

REQUIRED=['consistency@10_mean','link_learned_roc_auc','partner_new_mrr','pair_roc_auc','identity_top1']


def cache_file(source,reference,probe,cache,bootstrap=0):
    metadata=json.loads(Path(reference,'manifest.json').read_text())
    key=joblib.hash((source.index.tolist(),source.to_numpy(),metadata['files_sha256'],
                     metadata.get('split','test'),metadata['neighbor_ceiling'],
                     signature(Path(probe,'edges.csv')),json.loads(Path(probe,'manifest.json').read_text()).get('split','validation'),
                     bootstrap,'sg_geometry_v3'))
    return Path(cache)/f'sg-geometry-{key}.joblib'


def pair_metric(frame,pairs,split,bootstrap):
    p=select_split(pairs,split,'pairs.csv')
    scores=np.concatenate([cosine_pairs(frame,p.host_a.iloc[i:i+512].tolist(),p.host_b.iloc[i:i+512].tolist())
                           for i in range(0,len(p),512)])
    return auc_with_ci(p.label.to_numpy(int),scores,n_boot=bootstrap,seed=0,hosts=host_index(p.host_a,p.host_b))


def geometry(source,reference,probe,cache,bootstrap=0):
    reference=Path(reference);probe=Path(probe)
    saved=cache_file(source,reference,probe,cache,bootstrap)
    if saved.exists():
        print('reusing identical source/reference geometry',source.shape,flush=True)
        return joblib.load(saved)
    manifest=json.loads((reference/'manifest.json').read_text());split=manifest.get('split','test')
    hosts=pd.read_csv(reference/'eval_hosts.csv',dtype=str).host_id.tolist()
    a=source.loc[hosts]
    neighbors=evaluate_frozen_neighbors(a,reference/'neighbors.csv',ks=(10,),split=split)
    print('geometry neighbors ready',source.shape,flush=True)
    partners=partner_ranking(a,load_partners(reference/'partners.csv'),n_boot=bootstrap)
    pairs=pair_metric(a,load_pairs(reference/'pairs.csv'),split,bootstrap)
    print('geometry voting and pairs ready',flush=True)
    probe_split=json.loads((probe/'manifest.json').read_text()).get('split','validation')
    links=score_learned(source,load_candidate_edges(probe/'edges.csv'),load_candidate_edges(reference/'edges.csv'),
                        fit_split=probe_split,eval_split=split,cache=cache,n_boot=bootstrap)
    print('geometry links ready',links['roc_auc'],links['random_roc_auc'],flush=True)
    metrics={'consistency@10_mean':float(neighbors.loc[(10,'consistency'),'mean']),
             'link_learned_roc_auc':links['roc_auc'],'partner_new_mrr':partners['new']['embedding']['mrr'],
             'pair_roc_auc':pairs['roc_auc']}
    chance={'consistency@10_mean':10/(len(a)-1),'link_learned_roc_auc':links['random_roc_auc'],
            'partner_new_mrr':max(partners['new'][name]['mrr'] or 0 for name in ['history','popularity']),
            'pair_roc_auc':.5}
    summary=[{'metric':key,'value':value,'chance':chance[key],
              'group':{'consistency@10_mean':'Поведение','link_learned_roc_auc':'Будущие связи',
                       'partner_new_mrr':'Будущие связи','pair_roc_auc':'Общие собеседники'}[key]}
             for key,value in metrics.items()]
    result={'metrics':metrics,'summary':summary,'n_hosts':len(a),'dim':source.shape[1],
            'neighbor_ceiling':manifest['neighbor_ceiling'],
            'reference_sha256':manifest['files_sha256'],
            'details':{'partners':partners,'pairs':pairs,'links':links},
            'links_evaluation':'batched float64 C=1 logistic loss; verified against project implementation'}
    saved.parent.mkdir(parents=True,exist_ok=True);joblib.dump(result,saved)
    return result


def with_identity(source,later,reference,core,bootstrap=0):
    import copy
    reference=Path(reference);hosts=pd.read_csv(reference/'eval_hosts.csv',dtype=str).host_id.tolist()
    a,b=source.loc[hosts],later.loc[hosts]
    official=identity_retrieval(a,b,alignment='procrustes',n_boot=bootstrap,seed=0)
    direct=identity_retrieval(a,b,alignment='none',eval_fraction=1.,n_boot=bootstrap,seed=0)
    reverse=identity_retrieval(b,a,alignment='none',eval_fraction=1.,n_boot=bootstrap,seed=0)
    result=copy.deepcopy(core)
    result['metrics']['identity_top1']=official['top1']
    result['summary'].append({'metric':'identity_top1','value':official['top1'],
                              'chance':official['chance_top1'],'group':'Узнаваемость'})
    result['details']['identity']={'official':official,'direct_source_to_late':direct,'direct_late_to_source':reverse}
    if not scoreboard.complete(result,REQUIRED):raise ValueError('missing one of five mandatory metrics')
    previous=scoreboard.NEIGHBORS_IDEAL.copy()
    try:
        scoreboard.NEIGHBORS_IDEAL['similarity_graph_local']=result['neighbor_ceiling']
        result['score']=scoreboard._one_score(result,REQUIRED,window='similarity_graph_local')
        result['task_scores']=scoreboard.task_scores(result,REQUIRED,window='similarity_graph_local')
    finally:
        scoreboard.NEIGHBORS_IDEAL.clear();scoreboard.NEIGHBORS_IDEAL.update(previous)
    return result
