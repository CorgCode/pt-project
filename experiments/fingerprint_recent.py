"""Обновление source-профилей свежими наблюдениями: train≤12, validation13, test14."""
from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from experiments.fingerprint_donor_search import INCUMBENT,PREVIOUS,SOURCE,run
from experiments.fingerprint_expanded_search import slim,worker_env
from experiments.fingerprint_search import BASE,ROOT,dump,load,state,write_json
from experiments.multiday_finish import freeze_selection
from experiments.multiday_gain import evaluate
from pipeline.data import ProcessedData


VALIDATION=SOURCE/'fingerprint_memory'


def update_profiles(anchor,observations,method,asof=12,anchor_day=9):
    """Только известные past-ID обновляют training template; query не получает ID lookup."""
    if method not in {'last','ema1','ema3'}:raise ValueError('Неизвестный метод обновления')
    if len({day for day,_ in observations})!=len(observations):raise ValueError('День наблюдения повторяется')
    if len(set(anchor.index))!=len(anchor):raise ValueError('Повторные ID исходных шаблонов')
    for day,frame in observations:
        if not anchor_day<day<=asof:raise ValueError('Наблюдение за пределами прошлого')
        if not set(frame.index)<=set(anchor.index):raise ValueError('Нельзя расширять исходную когорту')
        if not frame.columns.equals(anchor.columns):raise ValueError('Разные координаты наблюдений')
        if len(set(frame.index))!=len(frame):raise ValueError('Повторные ID наблюдения')
    if method=='last':
        result=anchor.copy()
        for day,frame in sorted(observations,key=lambda x:x[0]):result.loc[frame.index]=frame
        return result
    half=1. if method=='ema1' else 3.
    initial=2.**(-(asof-anchor_day)/half)
    values=anchor.to_numpy(dtype=np.float64)*initial
    mass=np.full(len(anchor),initial)
    for day,frame in observations:
        rows=anchor.index.get_indexer(frame.index)
        weight=2.**(-(asof-day)/half)
        values[rows]+=frame.to_numpy(dtype=np.float64)*weight
        mass[rows]+=weight
    return pd.DataFrame((values/mass[:,None]).astype(np.float32),index=anchor.index,columns=anchor.columns)


class RecentEncoder:
    def __init__(self,base,templates,training_end_day=12):
        self.base,self.templates,self.training_end_day=base,templates,training_end_day

    def embed(self,data,window,hosts):
        if window=='train':return self.templates.loc[hosts]
        return self.base.embed(data,window,hosts)


def prepare_history(work):
    if (work/'history_ready.json').exists():return
    old=json.loads((PREVIOUS/'final/frozen_selection.json').read_text())['winner']
    anchor=pd.read_csv(Path(old['directory'])/'embeddings.csv',index_col=0)
    anchor.index=anchor.index.astype(str)
    anchor.to_csv(work/'anchor.csv',float_format='%.5g',index_label='host_id')
    cached={10:Path(old['directory'])/'later.csv',12:PREVIOUS/'final/winner/later.csv'}
    roots={10:VALIDATION/'day10',11:VALIDATION/'day11',12:PREVIOUS/'day12'}
    model=None;records=[]
    for day in [10,11,12]:
        saved=work/f'observation{day}.csv'
        data=ProcessedData(roots[day]);span=data.manifest['windows']['test']
        if span['end']>12*86400+12.5*3600:raise ValueError('История пересекает validation13')
        hosts=data.evaluable('test')
        if not set(hosts)<=set(anchor.index):raise ValueError('Историческая когорта не совпадает')
        if not saved.exists():
            if day in cached:
                frame=pd.read_csv(cached[day],index_col=0);frame.index=frame.index.astype(str)
                if set(frame.index)!=set(hosts):raise ValueError('Кэш другой target-когорты')
                frame=frame.loc[hosts]
            else:
                if psutil.virtual_memory().available<2*1024**3:raise MemoryError('Меньше2 ГБ RAM перед inference истории')
                if model is None:model=slim(load(INCUMBENT))
                frame=model.embed(data,'test',hosts)
            if not frame.columns.equals(anchor.columns):raise ValueError('Кэш не от того896-мерного энкодера')
            frame.to_csv(saved,float_format='%.5g',index_label='host_id')
        records.append({'day':day,'n_hosts':len(hosts),'interval':[span['start'],span['end']], 'data':str(roots[day])})
        print(json.dumps(records[-1]),flush=True)
        gc.collect()
    write_json(work/'history_ready.json',{'anchor':str(Path(old['directory'])/'embeddings.csv'),
        'anchor_asof_day':9,'updates':records,'fixed_cohort':len(anchor),'training_end_day':12,
        'validation_not_used':True})


def renew_token():
    result=subprocess.run(['curl','-fsS','--get','--data-urlencode','email=physcorgi@icloud.com',
        '--data-urlencode','usage=Research on temporal host embeddings; reproducible evaluation on anonymized NetFlow.',
        'https://csr.lanl.gov/data-fence/token'],capture_output=True,text=True,check=True)
    token=result.stdout.strip()
    if not 16<=len(token)<=4096 or any(c.isspace() for c in token) or '<' in token or '>' in token:
        raise ValueError('Форма LANL вернула неожиданный ответ; секретный ответ не выводится')
    path=SOURCE/'.lanl_token'
    descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(descriptor,'w') as stream:stream.write(token+'\n')
    path.chmod(0o600)


def download14(work):
    logpath=work/'download14.log'
    for attempt in range(2):
        env=dict(worker_env(1),LANL_TOKEN=(SOURCE/'.lanl_token').read_text().strip(),VERIFY_BZ2='0')
        with logpath.open('a') as log:
            result=subprocess.run(['/bin/bash','experiments/lanl/fetch.sh','14',str(SOURCE/'multiday/raw')],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode==0:return
        if attempt==0:renew_token()
    raise RuntimeError('Не удалось скачать14 после обновления токена; см.download14.log')


def selection(work):
    if (work/'selection_summary.json').exists():return
    anchor=pd.read_csv(work/'anchor.csv',index_col=0);anchor.index=anchor.index.astype(str)
    observations=[]
    for day in [10,11,12]:
        frame=pd.read_csv(work/f'observation{day}.csv',index_col=0);frame.index=frame.index.astype(str)
        observations.append((day,frame))
    data=ProcessedData(work/'paired_validation');ref=work/'reference_validation/validation'
    if list(anchor.index)!=data.cohort:raise ValueError('Изменился порядок исходной когорты')
    base=slim(load(INCUMBENT));records=[];paths=[]
    for method in ['incumbent','last','ema1','ema3']:
        out=work/'candidates'/method;out.mkdir(parents=True,exist_ok=True);paths.append(out)
        if (out/'best.json').exists():records.append(json.loads((out/'best.json').read_text()));continue
        model=base if method=='incumbent' else RecentEncoder(base,update_profiles(anchor,observations,method))
        state(work,'validation',candidate=method,completed=len(records),total=4)
        score,result=evaluate(model,data,'validation',ref,out,method)
        record={'name':method,'score':score,'identity':result['metrics']['identity_top1'],
                'partner_new_mrr':result['metrics']['partner_new_mrr']}
        write_json(out/'best.json',record);records.append(record)
        write_json(work/'validation_results.json',records);print(json.dumps(record),flush=True)
        gc.collect()
    winner=min(records,key=lambda r:(-r['score'],r['name']))
    control=next(r['score'] for r in records if r['name']=='incumbent')
    will_test=winner['name']!='incumbent' and winner['score']>=control+1
    chosen=work/'candidates'/winner['name']/'best.pkl'
    if winner['name']=='incumbent':
        if not chosen.exists():chosen.symlink_to(INCUMBENT)
    else:
        if will_test:
            # Правило выбрано по13. После выбора используем это уже прошлое
            # наблюдение для финального refit; закрытый14 не читается.
            observed=pd.read_csv(work/'candidates/incumbent/later.csv',index_col=0)
            observed.index=observed.index.astype(str)
            if data.manifest['windows']['validation']['end']>=13*86400+12.5*3600:
                raise ValueError('Refit13 пересекает test14')
            templates=update_profiles(anchor,[*observations,(13,observed)],winner['name'],asof=13)
            end=13
        else:templates=update_profiles(anchor,observations,winner['name']);end=12
        model=RecentEncoder(slim(base),templates,training_end_day=end)
        dump(model,chosen)
    winner['selection_training_end_day']=12
    winner['final_refit_training_end_day']=13 if will_test else 12
    winner['validation_score_is_pre_refit']=bool(will_test)
    write_json(work/'candidates'/winner['name']/'best.json',winner)
    write_json(work/'validation_results.json',records)
    freeze_selection(paths,work/'final')
    write_json(work/'selection_summary.json',{'winner':winner,'control':control,'validation_gain':winner['score']-control,
        'test_will_run':will_test,'test_gate':'gain>=1 over control;80 assessed only on blind test14','test_day':14,
        'final_refit_through_day':13 if will_test else 12,
        'protocol_changed':'select rule with train<=12 / validation13; refit selected rule through13; blind14'})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work',required=True,type=Path)
    ap.add_argument('--stage',choices=['all','history','download','selection'],default='all')
    args=ap.parse_args();work=args.work;work.mkdir(parents=True,exist_ok=True)
    if args.stage=='history':prepare_history(work);return
    if args.stage=='download':download14(work);return
    if args.stage=='selection':selection(work);return
    if (work/'final/REPORT.md').exists():
        state(work,'complete',**json.loads((work/'final/summary.json').read_text()));return
    protocol={'train_days':list(range(2,13)),'dimensions':[],'head_labels':[],'candidates':4,'validation_day':13,'test_day':14,
        'target_score':80,'anchor_encoder':'frozen32h encoder trained02–09','source_updates':[10,11,12],
        'update_window':'12:30–13:30, >=300 observed events','methods':['unchanged','last','EMA half1d','EMA half3d'],
        'source_contacts_day':2,'fixed_cohort':'original day02','selection':'validation13 only; freeze before test14',
        'test_gate':'improvement>=1 over control; goal80 is evaluated on blind test14',
        'final_refit':'after selecting rule on13, refit templates through13 before freeze; no14 data',
        'original_scoreboard_comparable':False,
        'warning':'new chronology; numbers cannot replace earlier day10/day12 scores; evaluate control on same new day'}
    if not (work/'protocol.json').exists():write_json(work/'protocol.json',protocol)
    log=(work/'network.log').open('a')
    network=subprocess.Popen([sys.executable,'-m','experiments.fingerprint_recent','--work',str(work),'--stage','download'],
        cwd=ROOT,env=worker_env(1),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    try:
        run(work,'history_updates','experiments.fingerprint_recent','--work',work,'--stage','history')
        if not (work/'day13/manifest.json').exists():
            run(work,'prepare_day13','experiments.multiday_prepare','--raw',SOURCE/'multiday/raw/netflow_day-13.bz2',
                '--base',BASE,'--out',work/'day13','--day',13)
        run(work,'reference_validation','experiments.multiday_reference','--base',BASE,'--target',work/'day13',
            '--out',work/'paired_validation','--reference',work/'reference_validation','--window','validation')
        run(work,'search','experiments.fingerprint_recent','--work',work,'--stage','selection')
        summary=json.loads((work/'selection_summary.json').read_text())
        if not summary['test_will_run']:
            state(work,'complete',result='no_material_validation_gain',**summary);return
        state(work,'waiting_for_download14')
        if network.wait():raise RuntimeError('Ошибка скачивания14; см.download14.log')
        if not (work/'day14/manifest.json').exists():
            run(work,'prepare_day14','experiments.multiday_prepare','--raw',SOURCE/'multiday/raw/netflow_day-14.bz2',
                '--base',BASE,'--out',work/'day14','--day',14)
        run(work,'reference_test','experiments.multiday_reference','--base',BASE,'--target',work/'day14',
            '--out',work/'paired_test','--reference',work/'reference_test','--window','test')
        from experiments.fingerprint_quick import candidate_directories
        run(work,'final_evaluation','experiments.multiday_finish','--candidates',*candidate_directories(work),
            '--data',work/'paired_test','--reference',work/'reference_test/test','--out',work/'final','--incumbent',INCUMBENT,
            '--test-label','day14,12:30–13:30','--protocol-text',
            'Кодировщик02–09; правило обновления выбрано по train<=12/validation13. Затем refit source-шаблонов через13 и фиксация перед новым test14. Source-контакты02; исходная когорта.')
        state(work,'complete',report=str(work/'final/REPORT.md'),**json.loads((work/'final/summary.json').read_text()))
    finally:
        if network.poll() is None:
            os.killpg(network.pid,signal.SIGTERM);network.wait()
        log.close()


if __name__=='__main__':
    try:main()
    except Exception as error:
        if '--work' in sys.argv:
            work=Path(sys.argv[sys.argv.index('--work')+1])
            if '--stage' in sys.argv and sys.argv[sys.argv.index('--stage')+1]=='download':
                write_json(work/'download14_error.json',{'error':f'{type(error).__name__}: {error}'})
            else:state(work,'failed',error=f'{type(error).__name__}: {error}')
        raise
