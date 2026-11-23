"""Пересчёт всех пяти метрик из сохранённых CSV с новым пустым кэшем."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from evaluation.validate import validate_submission
from experiments.temporal_graph_mixer.common.evaluate import geometry, with_identity
from experiments.temporal_graph_mixer.data import sha


def jobs(runs):
    dev = runs/'similarity_graph_recent/select01/references/day13_development'
    canonical = runs/'temporal_graph_mixer_01/v2/benchmark/references'
    for window,filename in [('test','test14.csv'),('test_day15','test15.csv'),('test_day16','test16.csv')]:
        p=runs/'similarity_graph_recent'
        yield 'similarity_graph_recent_control',window,p/'refine01/source.csv',p/filename,canonical/window
    for name, folder in [('directed_bytes_gnn','directed_bytes_gnn_01/benchmark_submission')]:
        for window in ['test','test_day15','test_day16']:
            p=runs/folder
            yield name,window,p/'embeddings.csv',p/(window+'.csv'),canonical/window
    for version,folder in [('v2','v2/benchmark')]:
        for window in ['test','test_day15','test_day16']:
            p=runs/'temporal_graph_mixer_01'/folder/window
            yield 'temporal_graph_mixer_'+version,window,p/'source.csv',p/'late.csv',canonical/window
    for variant in ['dev_step13','dev_step30','dev_no_graph','dev_no_bytes','dev_no_vote']:
        p=runs/'temporal_gatv2_pilot'/variant
        yield 'temporal_gatv2_'+variant,'development13',p/'embeddings.csv',p/'validation.csv',dev
    base=runs/'codebook_directed_gnn_01'
    for variant in ['zero','dev_identity','dev_no_graph','dev_undirected_bytes','dev_donors']:
        source=base/('train_donors' if variant=='dev_donors' else 'zero')/'source.csv'
        late=base/('dev_identity' if variant=='dev_donors' else variant)/'late.csv'
        yield 'codebook_directed_gnn_'+variant,'development13',source,late,dev
    p=runs/'directed_bytes_gnn_01/development'
    for variant in ['baseline','gnn_w050','no_graph_w050','no_bytes_w050']:
        yield 'directed_bytes_gnn_'+variant,'development13',p/(variant+'_source.csv'),p/(variant+'_day13.csv'),dev
    p=runs/'temporal_graph_mixer_01/v2/development/no_graph_global_w2'
    yield 'temporal_graph_mixer_v2','development13',p/'source.csv',p/'late.csv',dev


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    a=parser.parse_args()
    if a.output.exists() and not a.resume:raise FileExistsError('Для пересчёта нужен новый пустой каталог')
    a.output.mkdir(parents=True,exist_ok=a.resume)
    previous=json.loads((a.output/'results.json').read_text())['records'] if a.resume else []
    records=[]
    probe=a.data/'late_test/reference/validation'
    for name,window,source,late,reference in jobs(a.runs):
        validate_submission(str(source));validate_submission(str(late))
        x=pd.read_csv(source,index_col=0);y=pd.read_csv(late,index_col=0)
        if not set(y.index)<=set(x.index) or x.shape[1]!=y.shape[1]:raise ValueError('Несовместимые source/late')
        manifest=json.loads((reference/'manifest.json').read_text())
        for filename,digest in manifest['files_sha256'].items():
            if not filename.startswith('._') and sha(reference/filename)!=digest:
                raise ValueError('Эталон изменился: '+filename)
        cached=next((r for r in previous if r['method']==name and r['window']==window),None)
        if cached is not None:
            if (cached['source_sha256']!=sha(source) or cached['late_sha256']!=sha(late)
                    or cached['reference_sha256']!=manifest['files_sha256']
                    or cached['probe_edges_sha256']!=sha(probe/'edges.csv')):
                raise ValueError('Входы изменились после прерванного пересчёта')
            records.append(cached);continue
        core=geometry(x,reference,probe,a.output/'fresh_cache')
        result=with_identity(x,y,reference,core)
        p=a.output/(name+'__'+window+'.json')
        p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        records.append({'method':name,'window':window,'dim':x.shape[1],
                        'score':result['score'],'metrics':result['metrics'],
                        'source_sha256':sha(source),'late_sha256':sha(late),
                        'reference_sha256':manifest['files_sha256'],
                        'probe_edges_sha256':sha(probe/'edges.csv'),
                        'n_hosts':result['n_hosts']})
        output={'scoring_snapshot':'70347a3','fresh_cache':True,
                'benchmark_previously_studied':True,'independent_holdout':False,
                'records':records}
        (a.output/'results.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
        print(name,window,result['score'],flush=True)
    (a.output/'results.json').write_text(json.dumps({'scoring_snapshot':'70347a3','fresh_cache':True,
        'benchmark_previously_studied':True,'independent_holdout':False,'records':records},ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
