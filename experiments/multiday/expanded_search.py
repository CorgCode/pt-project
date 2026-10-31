"""32 часа train, validation10, новый однократный test12. Артефакты на SSD."""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil

from experiments.multiday.bag import BagHybrid
from experiments.multiday.search import (ArrayEmbeddings, BASE, ROOT, SOURCE, dump, load,
                                           slim_base, state, write_json)
from experiments.multiday.finish import freeze_selection
from experiments.multiday.gain import evaluate
from experiments.overnight.model import T
from pipeline.data import ProcessedData


PREVIOUS = SOURCE / 'fingerprint_memory'
INCUMBENT = PREVIOUS / 'final/model.pkl'


def slim(model):
    slim_base(model)
    if hasattr(model, 'base'):
        slim(model.base)
    if isinstance(model, BagHybrid):
        model._encoded = {}
        for key in ['memory', '_raw_memory', 'projection', 'class_projection']:
            model.bank.__dict__.pop(key, None)
    return model


def worker_env(threads=4):
    return dict(os.environ, PT_COMPACT='1', PT_TORCH_THREADS=str(threads), OMP_NUM_THREADS=str(threads),
                OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')


def command(work, label, module, *args):
    state(work, label)
    with (work / f'{label}.log').open('a') as log:
        subprocess.run([sys.executable, '-m', module, *map(str,args)], cwd=ROOT, env=worker_env(),
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def pool(work, stage, jobs, slots=2):
    if stage=='parallel_training' and psutil.virtual_memory().available < 7 * 1024**3:
        # Шире таблица EmbeddingBag и больше CSR-наблюдений: два AdamW могут
        # не поместиться вместе с рабочими приложениями. Потоки одной головы=4.
        slots=1
    pending, active = list(jobs), []
    try:
        while pending or active:
            if psutil.virtual_memory().available < 2 * 1024**3 or psutil.swap_memory().used > 1024**3:
                raise MemoryError('Резерв RAM меньше2 ГБ или swap больше1 ГБ; этап остановлен')
            while pending and len(active) < slots:
                label, module, args = pending.pop(0)
                log = (work / f'{label}.log').open('a')
                p = subprocess.Popen([sys.executable, '-m', module, *map(str,args)], cwd=ROOT,
                    env=worker_env(4 if stage=='parallel_training' else 2), stdout=log,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
                active.append((label,p,log))
                state(work, stage, workers={label:p.pid for label,p,_ in active}, pending=len(pending), slots=slots)
            for label,p,log in list(active):
                code=p.poll()
                if code is not None:
                    log.close(); active.remove((label,p,log))
                    if code:
                        raise RuntimeError(f'{label}: exit{code}, см.{label}.log')
            time.sleep(3)
    finally:
        for _,p,log in active:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM); p.wait()
            log.close()


def download(work):
    # Секрет только в окружении curl, его значение и URL в логи не попадают.
    token=(SOURCE/'.lanl_token').read_text().strip()
    env=dict(os.environ, LANL_TOKEN=token, VERIFY_BZ2='0')
    with (work/'download12.log').open('a') as log:
        subprocess.run(['/bin/bash','experiments/lanl/fetch.sh','12',str(SOURCE/'multiday/raw')],
                       cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    write_json(work/'download12_complete.json',{'day':12,'downloaded':True,'crc':'deferred to complete parsing'})


def teacher(work):
    if (work/'teacher.npy').exists():
        return
    model=slim(load(INCUMBENT))
    data=ProcessedData(BASE)
    np.save(work/'teacher.npy', model.embed(data,'train',data.cohort).to_numpy())


def search(work):
    data=ProcessedData(PREVIOUS/'paired_validation')
    ref=PREVIOUS/'reference_validation/validation'
    base=slim(load(INCUMBENT))
    bank=load(work/'bank.pkl')
    del bank.memory
    buckets=np.load(work/'buckets.npy')
    train_base=base.embed(data,'train',data.cohort)
    later_base=base.embed(data,'validation',data.evaluable('validation'))
    records, candidates=[],[]

    def assess(name, model, arrays=None):
        out=work/'candidates'/name
        out.mkdir(parents=True,exist_ok=True)
        candidates.append(out)
        if (out/'best.json').exists():
            records.append(json.loads((out/'best.json').read_text())); return
        if psutil.virtual_memory().available < 2*1024**3:
            raise MemoryError('Перед метриками осталось менее2 ГБ RAM')
        state(work,'validation',candidate=name,completed=len(records),total=18)
        score,result=evaluate(arrays or model,data,'validation',ref,out,name)
        record={'name':name,'score':score,'identity':result['metrics']['identity_top1']}
        # Совпадает с правилом разрешения равенств freeze_selection.
        best=min([(-r['score'],str(work/'candidates'/r['name'])) for r in records],default=(float('inf'),''))
        if (-score,str(out)) < best:
            for previous in candidates[:-1]:
                (previous/'best.pkl').unlink(missing_ok=True)
            if name=='incumbent':
                shutil.copy2(INCUMBENT,out/'best.pkl')
            else:
                dump(slim(model),out/'best.pkl')
        write_json(out/'best.json',record)
        records.append(record)
        write_json(work/'validation_results.json',records)
        print(json.dumps(record),flush=True)
        gc.collect()

    assess('incumbent',base,ArrayEmbeddings(train_base,later_base))
    baseline=T.TemporalEncoder().fit(data)
    assess('baseline',baseline); del baseline
    for dim in [128,256]:
        for epoch in [120,300]:
            head=load(work/f'bag{dim}/epoch{epoch}.pkl')
            model=BagHybrid(base,bank,PREVIOUS/'count_cache',buckets,head)
            for mode in ['neural','posterior']:
                model.mode,model.weight=mode,1.0
                train=model.embed(data,'train',data.cohort)
                later=model.embed(data,'validation',data.evaluable('validation'))
                for weight in [0.25,0.5]:
                    model.weight=weight
                    a,b=train.copy(),later.copy()
                    a.iloc[:,train_base.shape[1]:] *= weight
                    b.iloc[:,later_base.shape[1]:] *= weight
                    assess(f'expanded{dim}_e{epoch}_{mode}_w{weight}',model,ArrayEmbeddings(a,b))
            del model,head
            gc.collect()
    frozen=freeze_selection(candidates,work/'final')
    state(work,'winner_frozen',winner=frozen['winner'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work',required=True,type=Path)
    ap.add_argument('--stage',choices=['all','download','teacher','search'],default='all')
    args=ap.parse_args(); work=args.work; work.mkdir(parents=True,exist_ok=True)
    if args.stage=='download': download(work); return
    if args.stage=='teacher': teacher(work); return
    if args.stage=='search': search(work); return
    if (work/'final/REPORT.md').exists():
        state(work,'complete',**json.loads((work/'final/summary.json').read_text())); return
    protocol={'train_days':list(range(2,10)),'train_hours':[10,11,12,13],'train_hours_total':32,
        'validation_day':10,'test_day':12,'selection':'validation only; test12 once after freeze',
        'fixed_cohort':'original day02','incumbent':str(INCUMBENT),'width':262144,
        'dimensions':[128,256],'checkpoints':[120,300],'modes':['neural','posterior'],
        'fusion_weights':[0.25,0.5],'candidates':18,'target_score':80,
        'note':'Historical six-day encoder and eight-day head remain frozen components; new head sees32h. Separate protocol from original leaderboard.'}
    if not (work/'protocol.json').exists(): write_json(work/'protocol.json',protocol)
    # Скачивание сети перекрывается с потоковым извлечением двух TRAIN-дней.
    network_log=(work/'network.log').open('a')
    network=subprocess.Popen([sys.executable,'-m','experiments.multiday.expanded_search',
        '--work',str(work),'--stage','download'],cwd=ROOT,env=worker_env(1),stdout=network_log,
        stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    try:
        jobs=[(f'extract{day:02d}','experiments.multiday.expanded_data',
               ['--work',work,'--base',BASE,'--raw',SOURCE/f'multiday/raw/netflow_day-{day:02d}.bz2','--day',day])
              for day in range(2,10) if not (work/f'day{day:02d}/complete.json').exists()]
        pool(work,'extracting_32_hours',jobs)
        if not (work/'inputs_ready.json').exists():
            command(work,'fit_bank','experiments.multiday.expanded_data','--work',work,'--base',BASE,'--fit')
        command(work,'teacher','experiments.multiday.expanded_search','--work',work,'--stage','teacher')
        jobs=[(f'bag{dim}','experiments.multiday.expanded_train',['--work',work,'--dim',dim])
              for dim in [128,256] if not (work/f'bag{dim}/complete.json').exists()]
        pool(work,'parallel_training',jobs)
        command(work,'search','experiments.multiday.expanded_search','--work',work,'--stage','search')
        if not (work/'final/frozen_selection.json').exists(): raise RuntimeError('Модель не зафиксирована')
        state(work,'waiting_for_download12')
        if network.wait(): raise RuntimeError('Скачивание дня12 не завершено: см.download12.log')
        if not (work/'day12/manifest.json').exists():
            command(work,'prepare_day12','experiments.multiday.prepare','--raw',SOURCE/'multiday/raw/netflow_day-12.bz2',
                '--base',BASE,'--out',work/'day12','--day',12)
        command(work,'reference_test','experiments.multiday.reference','--base',BASE,'--target',work/'day12',
                '--out',work/'paired_test','--reference',work/'reference_test','--window','test')
        command(work,'final_evaluation','experiments.multiday.finish',
            '--candidates',*sorted((work/'candidates').iterdir()),'--data',work/'paired_test',
            '--reference',work/'reference_test/test','--out',work/'final','--incumbent',INCUMBENT,
            '--test-label','day12, 12:30–13:30','--protocol-text',
            'Новая голова: дни02–09,10:00–14:00 (32ч); validation10; test12. Замороженный победитель предыдущего этапа — контроль. Когорта дня02.')
        state(work,'complete',report=str(work/'final/REPORT.md'),**json.loads((work/'final/summary.json').read_text()))
    finally:
        if network.poll() is None:
            os.killpg(network.pid,signal.SIGTERM); network.wait()
        network_log.close()


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        if '--work' in sys.argv:
            state(Path(sys.argv[sys.argv.index('--work')+1]),'failed',error=f'{type(error).__name__}: {error}')
        raise
