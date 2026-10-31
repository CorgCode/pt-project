"""Небольшая голова под transfer новых собеседников; validation10 → test13."""
from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pandas as pd
import psutil

from experiments.multiday.donors import DonorHybrid
from experiments.multiday.expanded_search import pool, slim, worker_env
from experiments.multiday.quick import candidate_directories
from experiments.multiday.search import ArrayEmbeddings, BASE, ROOT, SOURCE, dump, load, state, write_json
from experiments.multiday.finish import freeze_selection
from experiments.multiday.gain import evaluate
from experiments.overnight.model import T
from pipeline.data import ProcessedData


PREVIOUS=SOURCE/'fingerprint_expanded'
INCUMBENT=PREVIOUS/'final/model.pkl'
VALIDATION=SOURCE/'fingerprint_memory'


def run(work,label,module,*args):
    state(work,label)
    with (work/f'{label}.log').open('a') as log:
        subprocess.run([sys.executable,'-m',module,*map(str,args)],cwd=ROOT,env=worker_env(),
                       stdout=log,stderr=subprocess.STDOUT,check=True)


def download(work):
    env=dict(worker_env(),LANL_TOKEN=(SOURCE/'.lanl_token').read_text().strip(),VERIFY_BZ2='0')
    with (work/'download13.log').open('a') as log:
        subprocess.run(['/bin/bash','experiments/lanl/fetch.sh','13',str(SOURCE/'multiday/raw')],
                       cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)


def search(work):
    if (work/'selection_summary.json').exists(): return
    old=json.loads((PREVIOUS/'final/frozen_selection.json').read_text())['winner']
    source=Path(old['directory'])
    a=pd.read_csv(source/'embeddings.csv',index_col=0)
    b=pd.read_csv(source/'later.csv',index_col=0)
    a.index=a.index.astype(str); b.index=b.index.astype(str)
    data=ProcessedData(VALIDATION/'paired_validation')
    ref=VALIDATION/'reference_validation/validation'
    if list(a.index)!=data.cohort or set(b.index)!=set(data.evaluable('validation')):
        raise ValueError('Кэш исходных эмбеддингов не соответствует когорте')
    oldresult=json.loads((source/'result.json').read_text())
    original_partner=oldresult['metrics']['partner_new_mrr']
    initial=work/'candidates/incumbent'; initial.mkdir(parents=True,exist_ok=True)
    row={'name':'incumbent','score':old['score'],'identity':old['identity'],'partner_new_mrr':original_partner}
    write_json(initial/'best.json',row)
    if not (initial/'best.pkl').exists(): (initial/'best.pkl').symlink_to(INCUMBENT)
    records=[row]
    for label in ['donor0','donor025']:
        for epoch in [100,400]:
            head=load(work/label/f'epoch{epoch}.pkl')
            train_part=T._block(head.encode(a.iloc[:,-256:].to_numpy()))
            later_part=T._block(head.encode(b.iloc[:,-256:].to_numpy()))
            for weight in [0.25,0.5,1.0]:
                name=f'{label}_e{epoch}_w{weight}'
                out=work/'candidates'/name
                if (out/'best.json').exists():
                    records.append(json.loads((out/'best.json').read_text())); continue
                if psutil.virtual_memory().available<2*1024**3:
                    raise MemoryError('Осталось меньше2 ГБ RAM перед метриками')
                import numpy as np
                train=pd.DataFrame(np.hstack([T._block(a.to_numpy()),train_part*weight]),index=a.index).add_prefix('emb_')
                later=pd.DataFrame(np.hstack([T._block(b.to_numpy()),later_part*weight]),index=b.index).add_prefix('emb_')
                state(work,'validation',candidate=name,completed=len(records),total=13)
                score,result=evaluate(ArrayEmbeddings(train,later),data,'validation',ref,out,name)
                row={'name':name,'score':score,'identity':result['metrics']['identity_top1'],
                     'partner_new_mrr':result['metrics']['partner_new_mrr'],'head':label,'epoch':epoch,'weight':weight}
                write_json(out/'best.json',row); records.append(row)
                write_json(work/'validation_results.json',records)
                print(json.dumps(row),flush=True); gc.collect()
    winner=min(records,key=lambda r:(-r['score'],r['name']))
    if winner['name']!='incumbent':
        state(work,'export_model',winner=winner)
        head=load(work/winner['head']/f"epoch{winner['epoch']}.pkl")
        model=DonorHybrid(slim(load(INCUMBENT)),head,winner['weight'])
        dump(model,work/'candidates'/winner['name']/'best.pkl')
    summary={'winner':winner,'validation_gain':winner['score']-old['score'],
             'partner_mrr_gain':winner['partner_new_mrr']-original_partner,
             'test_will_run':winner['score']>=old['score']+1 and winner['partner_new_mrr']>=original_partner+0.01,
             'test_gate':{'score_gain':1.0,'partner_mrr_gain':0.01}}
    # Модель и решение фиксируются перед любым чтением test13.
    freeze_selection(candidate_directories(work),work/'final')
    write_json(work/'selection_summary.json',summary)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work',required=True,type=Path)
    ap.add_argument('--stage',choices=['all','search','download'],default='all')
    args=ap.parse_args(); work=args.work; work.mkdir(parents=True,exist_ok=True)
    if args.stage=='search': search(work); return
    if args.stage=='download': download(work); return
    if (work/'final/REPORT.md').exists():
        state(work,'complete',**json.loads((work/'final/summary.json').read_text())); return
    protocol={'train_days':list(range(2,10)),'train_hours':[11],'dimensions':[],
        'head_labels':['donor0','donor025'],'checkpoints':[100,400],'candidates':13,
        'validation_day':10,'test_day':13,'target_score':80,
        'objective':'neighbors that donate new next-day partners; soft target distribution over other hosts',
        'identity_weights':[0.0,0.25],'fusion_weights':[0.25,0.5,1.0],
        'feature_encoder':'frozen 32h encoder, historical inputs02–09',
        'historical_pairs_role':'training diagnostics only, not independent validation',
        'test_gate':{'validation_score_gain':1.0,'partner_new_mrr_gain':0.01},
        'test_policy':'new day13 exactly once after freeze; no re-evaluation on11/12',
        'original_scoreboard_comparable':False}
    if not (work/'protocol.json').exists(): write_json(work/'protocol.json',protocol)
    network_log=(work/'network.log').open('a')
    network=subprocess.Popen([sys.executable,'-m','experiments.multiday.donor_search','--work',str(work),
        '--stage','download'],cwd=ROOT,env=worker_env(1),stdout=network_log,stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,start_new_session=True)
    try:
        if not (work/'inputs_ready.json').exists():
            run(work,'donor_inputs','experiments.multiday.donors','--work',work,'--stage','inputs')
        jobs=[(label,'experiments.multiday.donors',['--work',work,'--stage','train','--identity-weight',weight])
              for label,weight in [('donor0',0),('donor025',0.25)] if not (work/label/'complete.json').exists()]
        # Небольшие dense-головы, два процесса без больших embedding-таблиц.
        pool(work,'donor_training',jobs)
        run(work,'search','experiments.multiday.donor_search','--work',work,'--stage','search')
        summary=json.loads((work/'selection_summary.json').read_text())
        if not summary['test_will_run']:
            state(work,'complete',result='no_material_validation_gain',**summary); return
        state(work,'waiting_for_download13')
        if network.wait(): raise RuntimeError('Ошибка скачивания дня13; см.download13.log')
        if not (work/'day13/manifest.json').exists():
            run(work,'prepare_day13','experiments.multiday.prepare','--raw',SOURCE/'multiday/raw/netflow_day-13.bz2',
                '--base',BASE,'--out',work/'day13','--day',13)
        run(work,'reference_test','experiments.multiday.reference','--base',BASE,'--target',work/'day13',
            '--out',work/'paired_test','--reference',work/'reference_test','--window','test')
        run(work,'final_evaluation','experiments.multiday.finish','--candidates',*candidate_directories(work),
            '--data',work/'paired_test','--reference',work/'reference_test/test','--out',work/'final',
            '--incumbent',INCUMBENT,'--test-label','day13, 12:30–13:30','--protocol-text',
            'Замороженный энкодер32ч + голова полезности соседей на переходах02→03…08→09. Validation10, test13 после фиксации. Когорта дня02.')
        state(work,'complete',report=str(work/'final/REPORT.md'),**json.loads((work/'final/summary.json').read_text()))
    finally:
        if network.poll() is None:
            os.killpg(network.pid,signal.SIGTERM); network.wait()
        network_log.close()


if __name__=='__main__':
    try: main()
    except Exception as error:
        if '--work' in sys.argv:
            state(Path(sys.argv[sys.argv.index('--work')+1]),'failed',error=f'{type(error).__name__}: {error}')
        raise
