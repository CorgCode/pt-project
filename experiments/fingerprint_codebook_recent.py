"""Source-коды с обучением через12, development13, однократный закрытый14."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from experiments.fingerprint_codebook import CONFIGS,CHECKPOINTS,CodebookEncoder,revision,verify
from experiments.fingerprint_donor_search import INCUMBENT,run
from experiments.fingerprint_donors import peer_matrix
from experiments.fingerprint_expanded_search import pool,slim
from experiments.fingerprint_memory import read_fingerprints
from experiments.fingerprint_search import SOURCE,BASE,ArrayEmbeddings,dump,load,state,write_json
from experiments.multiday_finish import freeze_selection
from experiments.multiday_gain import evaluate
from pipeline.data import ProcessedData


OLD=SOURCE/'fingerprint_aligned'
RECENT=SOURCE/'fingerprint_recent'


def inputs(work):
    if (work/'history/inputs_ready.json').exists():return
    root=work/'history';root.mkdir(parents=True,exist_ok=True)
    hosts=json.loads((OLD/'hosts.json').read_text())
    vocabulary=json.loads((SOURCE/'fingerprint_transfer/vocabulary.json').read_text())
    names=['anchor.npy','hosts.json']
    for day in range(3,10):names += [f'historical{day:02d}.npy',f'future{day:02d}.npz',f'future_active{day:02d}.npy']
    for name in names:
        destination=root/name
        if not destination.exists():destination.symlink_to(OLD/name)
    records=[]
    ready=json.loads((RECENT/'history_ready.json').read_text())
    cutoff=ProcessedData(RECENT/'paired_validation').manifest['windows']['validation']['start']
    for record in ready['updates']:
        day=record['day']
        if day not in [10,11,12]:raise ValueError('Неожиданный день обучающей истории')
        data=ProcessedData(Path(record['data']))
        span=data.manifest['windows']['test']
        if span['end']>cutoff:raise ValueError('Обучение пересекает development13')
        counts=read_fingerprints(data,'test',hosts,SOURCE/'fingerprint_memory/count_cache')
        active=counts['events']>=300
        frame=pd.read_csv(RECENT/f'observation{day}.csv',index_col=0);frame.index=frame.index.astype(str)
        if set(frame.index)!=set(np.asarray(hosts)[active]):raise ValueError('Несовпадение когорты и кэша наблюдений')
        values=np.zeros((len(hosts),896),dtype=np.float32)
        if frame.shape[1]!=896:raise ValueError('Кэш не от896-мерного кодировщика')
        values[pd.Index(hosts).get_indexer(frame.index)]=frame.to_numpy(dtype=np.float32)
        np.save(root/f'historical{day:02d}.npy',values)
        np.save(root/f'future_active{day:02d}.npy',active)
        peers=counts['counts']['peers']
        visible=peers.index.get_level_values('t').str[2:].isin(vocabulary)
        future=peer_matrix({'counts':{'peers':peers[visible]}},hosts,vocabulary)
        sp.save_npz(root/f'future{day:02d}.npz',future)
        records.append({'day':day,'hosts':int(active.sum()),'interval':[span['start'],span['end']],
            'data':str(data.root),'feature_source':str(RECENT/f'observation{day}.csv')})
        print(json.dumps(records[-1]),flush=True)
        del counts,future,values,frame;gc.collect()
    if sorted(r['day'] for r in records)!=[10,11,12]:raise ValueError('Неполная история10–12')
    source_files=[root/name for d in range(3,13) for name in
        [f'historical{d:02d}.npy',f'future{d:02d}.npz',f'future_active{d:02d}.npy']]
    write_json(root/'inputs_ready.json',{'train_target_days':list(range(3,13)),
        'n_hosts':len(hosts),'source_contacts_day':2,'training_end_day':12,'new_days':records,
        'development_start':cutoff,
        'sources_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        'development13_not_used':True,'blind14_not_used':True})


def verify_history(work):
    record=json.loads((work/'history/inputs_ready.json').read_text())
    if record['train_target_days']!=list(range(3,13)) or record['training_end_day']!=12:
        raise RuntimeError('Изменился состав обучающей истории')
    for path,digest in record['sources_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Изменились обучающие признаки или метки')


def selection(work):
    verify(work);verify_history(work)
    if (work/'selection_summary.json').exists():return
    hosts=json.loads((work/'history/hosts.json').read_text())
    data=ProcessedData(RECENT/'paired_validation');ref=RECENT/'reference_validation/validation'
    if data.cohort!=hosts:raise ValueError('Изменилась исходная когорта')
    b=pd.read_csv(RECENT/'candidates/incumbent/later.csv',index_col=0);b.index=b.index.astype(str)
    if set(b.index)!=set(data.evaluable('validation')):raise ValueError('Кэш другого окна development13')
    control=json.loads((RECENT/'candidates/incumbent/best.json').read_text())
    old_result=json.loads((RECENT/'candidates/incumbent/result.json').read_text())
    for name,digest in old_result['reference_info']['files_sha256'].items():
        if digest is not None and hashlib.sha256((ref/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Эталоны изменились относительно контрольного прогона')
    initial=work/'candidates/incumbent';initial.mkdir(parents=True,exist_ok=True)
    write_json(initial/'best.json',control)
    if not (initial/'best.pkl').exists():(initial/'best.pkl').symlink_to(INCUMBENT)
    paths=[initial];records=[control]
    for label,_ in CONFIGS:
        for epoch in CHECKPOINTS:
            name=f'{label}_e{epoch}';out=work/'candidates'/name
            saved=out/'assessment.json'
            if saved.exists():
                records.append(json.loads(saved.read_text()))
                if (out/'best.json').exists():paths.append(out)
                continue
            checkpoint=load(work/label/f'epoch{epoch}.pkl')
            if checkpoint['hosts']!=hosts:raise ValueError('Когорта checkpoint отличается')
            train=pd.DataFrame(checkpoint['source'],index=hosts).add_prefix('emb_')
            later=pd.DataFrame(checkpoint['adapter'].transform(b.to_numpy(dtype=np.float32)),index=b.index).add_prefix('emb_')
            state(work,'validation',candidate=name,completed=len(records)-1,total=4)
            score,result=evaluate(ArrayEmbeddings(train,later),data,'validation',ref,out,name)
            row={'name':name,'score':score,'identity':result['metrics']['identity_top1'],
                'partner_new_mrr':result['metrics']['partner_new_mrr'],'head':label,'epoch':epoch,
                'gain':score-control['score'],'adapter_blend':checkpoint['adapter'].blend,
                'training_end_day':12,'development_day':13}
            write_json(saved,row);records.append(row)
            if score>=control['score']:write_json(out/'best.json',row);paths.append(out)
            write_json(work/'validation_results.json',records);print(json.dumps(row),flush=True);gc.collect()
    winner=min([json.loads((p/'best.json').read_text()) for p in paths],key=lambda r:(-r['score'],r['name']))
    if winner['name']!='incumbent':
        checkpoint=load(work/winner['head']/f"epoch{winner['epoch']}.pkl")
        source=pd.DataFrame(checkpoint['source'],index=hosts).add_prefix('emb_')
        dump(CodebookEncoder(slim(load(INCUMBENT)),source,checkpoint['adapter']),work/'candidates'/winner['name']/'best.pkl')
    freeze_selection(paths,work/'final')
    result={'winner':winner,'control':control['score'],'validation_gain':winner['score']-control['score'],
        'test_will_run':winner['name']!='incumbent' and winner['score']>=control['score']+1.,
        'test_gate':'development13 gain>=1; fixed train<=12; blind14 once after freeze',
        'final_refit_through_day':12}
    write_json(work/'validation_results.json',records);write_json(work/'selection_summary.json',result)
    report=['# Source-профили с историей до дня12','',
        'Development13 уже открыт; test14 закрыт до фиксации. Эти числа не равны исходной таблице.',
        '', '| Модель | Development13 |','|---|---:|']
    for r in records:report.append(f"| {r['name']} | {r['score']:.3f} |")
    report+=['',f"Прирост к контролю: {result['validation_gain']:+.3f}. Допуск к test14: {result['test_will_run']}."]
    (work/'REPORT.md').write_text('\n'.join(report)+'\n')


def final_test(work):
    if (work/'final/REPORT.md').exists():
        state(work,'complete',report=str(work/'final/REPORT.md'),**json.loads((work/'final/summary.json').read_text()));return
    result=json.loads((work/'selection_summary.json').read_text())
    if not result['test_will_run']:
        state(work,'complete',result='no_material_validation_gain',**result);return
    raw=SOURCE/'multiday/raw/netflow_day-14.bz2'
    if not raw.exists() or raw.stat().st_size!=1463364672:
        raise RuntimeError('Архив14 отсутствует или неполный; новый download не запускается автоматически')
    if not (work/'day14/manifest.json').exists():
        run(work,'prepare_day14','experiments.multiday_prepare','--raw',raw,'--base',BASE,'--out',work/'day14','--day',14)
    run(work,'reference_test','experiments.multiday_reference','--base',BASE,'--target',work/'day14',
        '--out',work/'paired_test','--reference',work/'reference_test','--window','test')
    paths=sorted(p for p in (work/'candidates').iterdir() if p.is_dir() and (p/'best.json').exists())
    run(work,'final_evaluation','experiments.multiday_finish','--candidates',*paths,'--data',work/'paired_test',
        '--reference',work/'reference_test/test','--out',work/'final','--incumbent',INCUMBENT,
        '--test-label','day14,12:30–13:30','--protocol-text',
        'Source02; обучаемые профили с целями03–12 и отдельный адаптер трафика. Development13, test14 после фиксации. Рефита по13 нет. Исходная когорта6453.')
    state(work,'complete',report=str(work/'final/REPORT.md'),**json.loads((work/'final/summary.json').read_text()))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work',type=Path,required=True)
    ap.add_argument('--stage',choices=['all','inputs','train','selection'],default='all')
    ap.add_argument('--neighbor-weight',type=float,choices=[1.,4.],default=1.)
    args=ap.parse_args();work=args.work;work.mkdir(parents=True,exist_ok=True)
    if args.stage=='inputs':inputs(work);return
    if args.stage=='train':
        from experiments.fingerprint_codebook import train
        verify_history(work)
        train(work,args.neighbor_weight,history_root=work/'history',target_days=tuple(range(3,13)),half_life=3.);return
    if args.stage=='selection':selection(work);return
    if not (work/'protocol.json').exists():write_json(work/'protocol.json',{
        'train_days':list(range(2,13)),'head_labels':[k for k,_ in CONFIGS],
        'checkpoints':list(CHECKPOINTS),'candidates':5,'validation_day':13,'test_day':14,'target_score':80,
        'original_scoreboard_comparable':False,'model':'trainable source codebook + asymmetric ridge traffic adapter',
        'source_contacts_day':2,'training_target_days':list(range(3,13)),'query_identity_lookup':False,
        'training_recency_half_life_days':3.,'adapter_historical_holdout_days':[11,12],
        'test_policy':'gain>=1 on development13, blind14 once after freeze; no refit13'})
    inputs(work)
    verify_history(work)
    if not (work/'training_revision.json').exists():write_json(work/'training_revision.json',{
        'sources':revision(),'torch':str(torch.__version__)})
    if not (work/'training_configuration.json').exists():write_json(work/'training_configuration.json',{
        'history_root':str((work/'history').resolve()),'target_days':list(range(3,13)),'half_life':3.})
    verify(work)
    jobs=[(label,'experiments.fingerprint_codebook_recent',['--work',work,'--stage','train','--neighbor-weight',weight])
        for label,weight in CONFIGS if not (work/label/'complete.json').exists()]
    pool(work,'donor_training',jobs,slots=2)
    run(work,'search','experiments.fingerprint_codebook_recent','--work',work,'--stage','selection')
    final_test(work)


if __name__=='__main__':
    try:main()
    except Exception as error:
        if '--work' in sys.argv:state(Path(sys.argv[sys.argv.index('--work')+1]),'failed',error=f'{type(error).__name__}: {error}')
        raise
