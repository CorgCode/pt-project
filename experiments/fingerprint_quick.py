"""Быстрый подбор PCA/частичного whitening без переобучения сетей."""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from experiments.fingerprint_expanded_search import slim, worker_env
from experiments.fingerprint_search import (ArrayEmbeddings, BASE, ROOT, SOURCE, dump, load, state, write_json)
from experiments.multiday_finish import freeze_selection
from experiments.multiday_gain import evaluate
from pipeline.data import ProcessedData


PREVIOUS = SOURCE/'fingerprint_expanded'
VALIDATION = SOURCE/'fingerprint_memory'
INCUMBENT = PREVIOUS/'final/model.pkl'


class Projection:
    def __init__(self, pca, dim, power):
        self.mean = pca.mean_.copy()
        self.components = pca.components_[:dim].copy()
        variance = pca.explained_variance_[:dim]
        self.scale = np.maximum(variance, pca.explained_variance_[0]*1e-4)**(-power/2)

    def transform(self, values):
        x = normalize(np.asarray(values,dtype=np.float32))
        return normalize(((x-self.mean)@self.components.T)*self.scale).astype(np.float32)


class ProjectedEncoder:
    def __init__(self, base, projection):
        self.base, self.projection = base, projection

    def embed(self, data, window, hosts):
        values = self.base.embed(data,window,hosts)
        return pd.DataFrame(self.projection.transform(values.to_numpy()),index=values.index).add_prefix('emb_')


def run(work,stage,module,*args):
    state(work,stage)
    with (work/f'{stage}.log').open('a') as log:
        subprocess.run([sys.executable,'-m',module,*map(str,args)],cwd=ROOT,env=worker_env(),
                       stdout=log,stderr=subprocess.STDOUT,check=True)


def select(work):
    if (work/'final/frozen_selection.json').exists(): return
    old = json.loads((PREVIOUS/'final/frozen_selection.json').read_text())['winner']
    source=Path(old['directory'])
    train=pd.read_csv(source/'embeddings.csv',index_col=0)
    later=pd.read_csv(source/'later.csv',index_col=0)
    train.index=train.index.astype(str); later.index=later.index.astype(str)
    data=ProcessedData(VALIDATION/'paired_validation')
    if list(train.index)!=data.cohort or set(later.index)!=set(data.evaluable('validation')):
        raise ValueError('Кэш эмбеддингов не соответствует фиксированной когорте')
    state(work,'pca')
    # Обучение преобразования использует ТОЛЬКО train-векторы.
    pca=PCA(n_components=384,svd_solver='randomized',iterated_power=4,random_state=73)
    pca.fit(normalize(train.to_numpy(dtype=np.float32)))
    dump(pca,work/'pca.pkl')
    ref=VALIDATION/'reference_validation/validation'
    initial=work/'candidates/incumbent'
    initial.mkdir(parents=True,exist_ok=True)
    write_json(initial/'best.json',{'name':'incumbent','score':old['score'],'identity':old['identity'],
                                  'cached_from':str(source),'reference':str(ref)})
    if not (initial/'best.pkl').exists(): (initial/'best.pkl').symlink_to(INCUMBENT)
    records=[json.loads((initial/'best.json').read_text())]
    for dim in [64,128,256,384]:
        for power in [0.0,0.5]:
            name=f'pca{dim}_power{power}'
            out=work/'candidates'/name
            saved=out/'best.json'
            if saved.exists():
                records.append(json.loads(saved.read_text())); continue
            if psutil.virtual_memory().available < 2*1024**3:
                raise MemoryError('Перед оценкой осталось меньше2 ГБ RAM')
            projection=Projection(pca,dim,power)
            a=pd.DataFrame(projection.transform(train.to_numpy()),index=train.index).add_prefix('emb_')
            b=pd.DataFrame(projection.transform(later.to_numpy()),index=later.index).add_prefix('emb_')
            state(work,'validation',candidate=name,completed=len(records),total=9)
            score,result=evaluate(ArrayEmbeddings(a,b),data,'validation',ref,out,name)
            row={'name':name,'score':score,'identity':result['metrics']['identity_top1'],'dim':dim,'power':power}
            write_json(saved,row); dump(projection,out/'projection.pkl')
            records.append(row); write_json(work/'validation_results.json',records)
            print(json.dumps(row),flush=True)
            gc.collect()
    winner=min(records,key=lambda r:(-r['score'],r['name']))
    write_json(work/'validation_results.json',records)
    bestdir=work/'candidates'/winner['name']
    if winner['name']!='incumbent':
        state(work,'export_model',winner=winner)
        model=ProjectedEncoder(slim(load(INCUMBENT)),load(bestdir/'projection.pkl'))
        dump(model,bestdir/'best.pkl')
    freeze_selection(sorted((work/'candidates').iterdir()),work/'final')
    write_json(work/'selection_summary.json',{'validation_gain':winner['score']-old['score'],
        'incumbent_score':old['score'],'winner':winner,'test_gate':1.0,
        'test_will_run':winner['score']-old['score']>=1.0})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work',required=True,type=Path)
    ap.add_argument('--select-only',action='store_true')
    args=ap.parse_args(); work=args.work; work.mkdir(parents=True,exist_ok=True)
    if args.select_only: select(work); return
    if (work/'final/REPORT.md').exists():
        state(work,'complete',**json.loads((work/'final/summary.json').read_text())); return
    protocol={'train_days':[],'dimensions':[],'candidates':9,'validation_day':10,'test_day':13,
        'target_score':80,'method':'frozen 32h encoder + PCA on train vectors only',
        'dims':[64,128,256,384],'whitening_power':[0.0,0.5],
        'selection':'validation only; test13 once after freeze if gain>=1 point',
        'test_gate_validation_gain':1.0,'original_scoreboard_comparable':False}
    if not (work/'protocol.json').exists(): write_json(work/'protocol.json',protocol)
    run(work,'search','experiments.fingerprint_quick','--work',work,'--select-only')
    selection=json.loads((work/'selection_summary.json').read_text())
    if not selection['test_will_run']:
        state(work,'complete',result='no_material_validation_gain',**selection)
        return
    # Сеть и новый день нужны только при существенном приросте на validation.
    state(work,'download13')
    env=dict(worker_env(),LANL_TOKEN=(SOURCE/'.lanl_token').read_text().strip(),VERIFY_BZ2='0')
    with (work/'download13.log').open('a') as log:
        subprocess.run(['/bin/bash','experiments/lanl/fetch.sh','13',str(SOURCE/'multiday/raw')],
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    if not (work/'day13/manifest.json').exists():
        run(work,'prepare_day13','experiments.multiday_prepare','--raw',SOURCE/'multiday/raw/netflow_day-13.bz2',
            '--base',BASE,'--out',work/'day13','--day',13)
    run(work,'reference_test','experiments.multiday_reference','--base',BASE,'--target',work/'day13',
        '--out',work/'paired_test','--reference',work/'reference_test','--window','test')
    run(work,'final_evaluation','experiments.multiday_finish','--candidates',*sorted((work/'candidates').iterdir()),
        '--data',work/'paired_test','--reference',work/'reference_test/test','--out',work/'final',
        '--incumbent',INCUMBENT,'--test-label','day13, 12:30–13:30','--protocol-text',
        'Замороженная модель32ч + PCA по train-векторам. Выбор на validation10, test13 один раз после фиксации. Когорта дня02.')
    state(work,'complete',report=str(work/'final/REPORT.md'),**json.loads((work/'final/summary.json').read_text()))


if __name__=='__main__':
    try: main()
    except Exception as error:
        if '--work' in sys.argv:
            state(Path(sys.argv[sys.argv.index('--work')+1]),'failed',error=f'{type(error).__name__}: {error}')
        raise
