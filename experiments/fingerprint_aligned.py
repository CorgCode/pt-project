"""Фиксированный source02 и обучение фактически экспортируемой геометрии."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sp
import torch
from sklearn.preprocessing import normalize, StandardScaler
from torch.nn import functional as F

from experiments.fingerprint_donors import DonorHead, peer_matrix
from experiments.fingerprint_memory import read_fingerprints
from experiments.fingerprint_donor_search import INCUMBENT, PREVIOUS, SOURCE, run, download
from experiments.fingerprint_expanded_search import pool, slim
from experiments.fingerprint_quick import candidate_directories
from experiments.fingerprint_search import ArrayEmbeddings, BASE, dump, load, state, write_json
from experiments.fingerprint_transfer import PartnerPair, sparse_partner_scores
from experiments.multiday_finish import freeze_selection
from experiments.multiday_gain import check_chronology, evaluate
from experiments.overnight.model import T
from pipeline.data import ProcessedData


TRANSFER=SOURCE/'fingerprint_transfer'
VALIDATION=SOURCE/'fingerprint_memory'


def block_tensor(values):
    centered=values-values.mean(0)
    return centered/centered.square().sum(1).mean().sqrt().clamp_min(1e-12)


def fused_tensor(base,head_values,weight):
    return F.normalize(torch.cat([block_tensor(base),block_tensor(head_values)*weight],dim=1),dim=1)


def fused_frame(base,head,weight):
    values=np.hstack([T._block(base.to_numpy()),T._block(head.encode(base.to_numpy()))*weight])
    return pd.DataFrame(values,index=base.index).add_prefix('emb_')


def gallery_labels(queries,population):
    indices=np.flatnonzero(population)
    positions=pd.Index(indices).get_indexer(queries)
    if (positions<0).any(): raise ValueError('Запрос отсутствует в исторической target-когорте')
    return indices,positions


class AlignedHybrid:
    def __init__(self,base,head,weight):
        self.base,self.head,self.weight=base,head,weight

    def embed(self,data,window,hosts):
        return fused_frame(self.base.embed(data,window,hosts),self.head,self.weight)


def history_input(base,source,out,window='test'):
    """Историческое окно под inference-window, без выбора будущей когорты."""
    (out/'flows').mkdir(parents=True,exist_ok=True)
    for original,target in [(source/'flows'/f'{window}.csv',out/'flows/validation.csv'),
                            (base/'flows/train.csv',out/'flows/train.csv')]:
        if not target.exists(): target.symlink_to(original.resolve())
    table=pd.read_csv(base/'cohort.csv',dtype={'host_id':str})
    table.to_csv(out/'cohort.csv',index=False)
    manifest=copy.deepcopy(json.loads((base/'manifest.json').read_text()))
    target=json.loads((source/'manifest.json').read_text())['windows'][window]
    manifest['windows']={'train':manifest['windows']['train'],'validation':target}
    manifest['config']['windows']={k:[v['start'],v['end']] for k,v in manifest['windows'].items()}
    manifest['historical_model_input_only']=True
    manifest['historical_source']=str(source)
    write_json(out/'manifest.json',manifest)
    return ProcessedData(out)


def inputs(work):
    if (work/'inputs_ready.json').exists(): return
    metadata=json.loads((VALIDATION/'bank_metadata.json').read_text())
    days=[ProcessedData(Path(p)) for p in metadata['training_sources']]
    check_chronology(days,9*86400+12.5*3600)
    old=json.loads((PREVIOUS/'final/frozen_selection.json').read_text())['winner']
    anchor=pd.read_csv(Path(old['directory'])/'embeddings.csv',index_col=0)
    anchor.index=anchor.index.astype(str)
    if list(anchor.index)!=days[0].cohort: raise ValueError('Изменился порядок когорты anchor')
    np.save(work/'anchor.npy',anchor.to_numpy(dtype=np.float32))
    write_json(work/'hosts.json',list(anchor.index))
    vocabulary=json.loads((TRANSFER/'vocabulary.json').read_text())
    model=slim(load(INCUMBENT))
    historical=[]
    pairs=[]
    for day,source in zip(range(2,10),days):
        saved=work/f'historical{day:02d}.npy'
        if saved.exists() and (work/f'future{day:02d}.npz').exists() and (work/f'future_active{day:02d}.npy').exists():
            values=np.load(saved)
        else:
            if psutil.virtual_memory().available<2*1024**3: raise MemoryError('Меньше2 ГБ RAM перед историческим inference')
            data=history_input(BASE,source.root,work/f'history/day{day:02d}')
            if data.manifest['windows']['validation']['end']>9*86400+12.5*3600:
                raise ValueError('Историческая цель пересекает validation10')
            values=model.embed(data,'validation',list(anchor.index)).to_numpy(dtype=np.float32)
            np.save(saved,values)
            counts=read_fingerprints(source,'test',list(anchor.index),VALIDATION/'count_cache')
            # Неизвестные source-словарю контакты не входят в кандидаты метрики.
            peers=counts['counts']['peers']
            visible=peers.index.get_level_values('t').str[2:].isin(vocabulary)
            following=peer_matrix({'counts':{'peers':peers[visible]}},list(anchor.index),vocabulary)
            sp.save_npz(work/f'future{day:02d}.npz',following)
            np.save(work/f'future_active{day:02d}.npy',counts['events']>=300)
            del counts,following
            slim(model); gc.collect()
        if values.shape!=anchor.shape: raise ValueError('Размер исторических признаков отличается от anchor')
        historical.append(values)
        if day>=3:
            pairs.append({'source_day':2,'target_day':day,
                          'active_queries':int(np.load(work/f'future_active{day:02d}.npy').sum())})
        print(json.dumps({'prepared_peer_day':day,'features':list(values.shape)}),flush=True)
    np.save(work/'historical.npy',np.stack(historical))
    ready=json.loads((TRANSFER/'inputs_ready.json').read_text())
    if ready['n_hosts']!=len(anchor): raise ValueError('Другая когорта у peer-меток')
    write_json(work/'inputs_ready.json',{'n_hosts':len(anchor),'pairs':pairs,
        'source_contacts':'fixed day02','features':'full frozen encoder, no validation input in training',
        'source_hours':'11:00–12:00','target_hours':'12:30–13:30',
        'training_end_day':9,'validation_used_for_training':False})


def aligned_pairs(work,hosts,vocabulary):
    source=sp.load_npz(TRANSFER/'source0.npz')
    source_active=np.diff(source.indptr)>0
    pairs=[]
    for day in range(3,10):
        population=np.load(work/f'future_active{day:02d}.npy')
        # Reference/partners ограничивает доноров хостами, активными в target.
        # Здесь это только исторические метки; будущая когорта13 не читается.
        matrix=source.multiply(population[:,None]).tocsr()
        matrix.eliminate_zeros()
        pairs.append(PartnerPair(matrix,sp.load_npz(work/f'future{day:02d}.npz'),hosts,vocabulary,
                                 source_active&population))
    return pairs


def train(work,weight,epochs=400,checkpoints=(100,400)):
    torch.set_num_threads(max(1,int(os.environ.get('PT_TORCH_THREADS','4'))))
    label='aligned05' if weight==0.5 else 'aligned10'
    out=work/label;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists(): return
    anchor=np.load(work/'anchor.npy')
    history=np.load(work/'historical.npy')
    populations=[np.load(work/f'future_active{day:02d}.npy') for day in range(2,10)]
    hosts=json.loads((work/'hosts.json').read_text())
    vocabulary=json.loads((TRANSFER/'vocabulary.json').read_text())
    pairs=aligned_pairs(work,hosts,vocabulary)
    raw=np.vstack([anchor,*[v[active] for v,active in zip(history,populations)]])
    scaler=StandardScaler().fit(normalize(raw));del raw
    head=DonorHead(scaler)
    x=torch.from_numpy(normalize(scaler.transform(normalize(anchor))).astype(np.float32))
    hx=[torch.from_numpy(normalize(scaler.transform(normalize(v))).astype(np.float32)) for v in history]
    base=torch.from_numpy(anchor);past=[torch.from_numpy(v) for v in history]
    base_unit=F.normalize(block_tensor(base),dim=1)
    optimizer=torch.optim.AdamW(head.net.parameters(),lr=5e-4,weight_decay=1e-3)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
    rng=np.random.default_rng(17)
    for epoch in range(1,epochs+1):
        head.net.train();i=int(rng.integers(7))
        queries,candidates,unique,inverse=pairs[i].sample(rng)
        q=torch.from_numpy(queries)
        z=fused_tensor(base,head.net(x),weight)
        scores=sparse_partner_scores(z,queries,pairs[i].scaled,unique)
        logits=scores.gather(1,torch.from_numpy(inverse)).clamp_min(1e-30).log()
        partner=F.cross_entropy(logits,torch.zeros(len(q),dtype=torch.long))
        # Сравнивается фактический source-anchor с историческим поздним окном.
        gallery,labels=gallery_labels(queries,populations[i+1])
        later=fused_tensor(past[i+1][gallery],head.net(hx[i+1][gallery]),weight)
        identity=F.cross_entropy(z[q]@later.T/0.1,torch.from_numpy(labels))
        other=torch.from_numpy(rng.integers(len(hosts),size=len(q)))
        preserve=F.mse_loss((z[q]*z[other]).sum(1),(base_unit[q]*base_unit[other]).sum(1))
        loss=partner+0.25*identity+0.1*preserve
        if not torch.isfinite(loss): raise FloatingPointError('Нечисловая потеря aligned')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(head.net.parameters(),3)
        if not torch.isfinite(norm): raise FloatingPointError('Нечисловой градиент aligned')
        optimizer.step();schedule.step()
        if epoch%10==0:
            print(json.dumps({'epoch':epoch,'loss':float(loss.detach()),'partner_loss':float(partner.detach()),
                'identity_loss':float(identity.detach()),'geometry_loss':float(preserve.detach()),'source_day':2,
                'target_day':i+3,'fusion_weight':weight}),flush=True)
        if epoch in checkpoints:dump(head,out/f'epoch{epoch}.pkl')
    write_json(out/'complete.json',{'epochs':list(checkpoints),'fusion_weight':weight})


def selection(work, configurations=None, checkpoints=(100,400)):
    if (work/'selection_summary.json').exists(): return
    old=json.loads((PREVIOUS/'final/frozen_selection.json').read_text())['winner']
    source=Path(old['directory'])
    train=pd.read_csv(source/'embeddings.csv',index_col=0);train.index=train.index.astype(str)
    cells={
        'day10':(VALIDATION/'paired_validation',VALIDATION/'reference_validation/validation','validation',source/'later.csv'),
        'day12':(PREVIOUS/'paired_test',PREVIOUS/'reference_test/test','test',PREVIOUS/'final/winner/later.csv')}
    baseline={'day10':old['score'],'day12':json.loads((PREVIOUS/'final/summary.json').read_text())['scores']['winner']}
    initial=work/'candidates/incumbent';initial.mkdir(parents=True,exist_ok=True)
    initialrow={'name':'incumbent','score':min(baseline.values()),'scores_by_day':baseline,
                'selection_metric':'minimum score across development10/12'}
    write_json(initial/'best.json',initialrow)
    if not (initial/'best.pkl').exists():(initial/'best.pkl').symlink_to(INCUMBENT)
    eligible=[initial];records=[initialrow]
    for label,weight in (configurations or [('aligned05',0.5),('aligned10',1.0)]):
        for epoch in checkpoints:
            name=f'{label}_e{epoch}';out=work/'candidates'/name
            saved=out/'assessment.json'
            if saved.exists():
                row=json.loads(saved.read_text());records.append(row)
                if (out/'best.json').exists(): eligible.append(out)
                continue
            head=load(work/label/f'epoch{epoch}.pkl')
            a=fused_frame(train,head,weight)
            scores,identities={},[]
            for day,(data_path,ref,window,later_file) in cells.items():
                completed=out/day/'scored.json'
                if completed.exists():
                    result=json.loads(completed.read_text());scores[day]=result['score'];identities.append(result['identity'])
                    continue
                if psutil.virtual_memory().available<2*1024**3: raise MemoryError('Меньше2 ГБ RAM перед оценкой')
                b=pd.read_csv(later_file,index_col=0);b.index=b.index.astype(str)
                data=ProcessedData(data_path)
                if list(train.index)!=data.cohort or set(b.index)!=set(data.evaluable(window)):
                    raise ValueError('Кэш другой когорты')
                state(work,'validation',candidate=f'{name}/{day}',completed=len(records)-1,total=4)
                score,result=evaluate(ArrayEmbeddings(a,fused_frame(b,head,weight)),data,window,ref,out/day,f'{name}_{day}')
                scores[day]=score;identities.append(result['metrics']['identity_top1'])
                write_json(completed,{'score':score,'identity':identities[-1]});gc.collect()
            row={'name':name,'score':min(scores.values()),'scores_by_day':scores,'identity':min(identities),
                 'gains_by_day':{day:scores[day]-baseline[day] for day in cells},'head':label,'epoch':epoch,'weight':weight,
                 'selection_metric':'minimum score across development10/12'}
            write_json(saved,row);records.append(row)
            if all(scores[day]>=baseline[day] for day in cells):
                write_json(out/'best.json',row);eligible.append(out)
            write_json(work/'validation_results.json',records)
            print(json.dumps(row),flush=True)
    winner=min([json.loads((p/'best.json').read_text()) for p in eligible],key=lambda r:(-r['score'],r['name']))
    if winner['name']!='incumbent':
        state(work,'export_model',winner=winner)
        head=load(work/winner['head']/f"epoch{winner['epoch']}.pkl")
        dump(AlignedHybrid(slim(load(INCUMBENT)),head,winner['weight']),work/'candidates'/winner['name']/'best.pkl')
    freeze_selection(eligible,work/'final')
    write_json(work/'validation_results.json',records)
    write_json(work/'selection_summary.json',{'winner':winner,'baseline':baseline,
        'validation_gain':min(winner['scores_by_day'][k]-baseline[k] for k in baseline),
        'test_will_run':passes_gate(winner['scores_by_day']),'test_gate':'both development scores >=80',
        'development12_previously_tested':True})


def passes_gate(scores):
    return set(scores)=={'day10','day12'} and all(np.isfinite(v) and 80<=v<=100 for v in scores.values())


def final_test(work):
    result=json.loads((work/'selection_summary.json').read_text())
    if not result['test_will_run']:
        state(work,'complete',result='below_80_on_development',**result);return
    # Выбор уже заморожен.13 скачан раньше, но до этой точки не анализируется.
    raw=SOURCE/'multiday/raw/netflow_day-13.bz2'
    if not raw.exists() or raw.stat().st_size!=1499152444: download(work)
    if not (work/'day13/manifest.json').exists():
        run(work,'prepare_day13','experiments.multiday_prepare','--raw',raw,'--base',BASE,'--out',work/'day13','--day',13)
    run(work,'reference_test','experiments.multiday_reference','--base',BASE,'--target',work/'day13',
        '--out',work/'paired_test','--reference',work/'reference_test','--window','test')
    from experiments.fingerprint_quick import candidate_directories
    candidates=[p for p in candidate_directories(work) if (p/'best.json').exists()]
    run(work,'final_evaluation','experiments.multiday_finish','--candidates',*candidates,
        '--data',work/'paired_test','--reference',work/'reference_test/test','--out',work/'final','--incumbent',INCUMBENT,
        '--test-label','day13,12:30–13:30','--protocol-text',
        'Source02 фиксирован; голова обучена на итоговой геометрии, исторические цели03–09. Подбор:min score дней10/12 (development); test13 закрыт до выбора.')
    state(work,'complete',report=str(work/'final/REPORT.md'),**json.loads((work/'final/summary.json').read_text()))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work',required=True,type=Path)
    ap.add_argument('--stage',choices=['all','inputs','train','selection'],default='all')
    ap.add_argument('--fusion-weight',type=float,choices=[0.5,1.0],default=0.5)
    args=ap.parse_args();work=args.work;work.mkdir(parents=True,exist_ok=True)
    if args.stage=='inputs':inputs(work);return
    if args.stage=='train':train(work,args.fusion_weight);return
    if args.stage=='selection':selection(work);return
    if (work/'final/REPORT.md').exists():
        state(work,'complete',**json.loads((work/'final/summary.json').read_text()));return
    protocol={'train_days':list(range(2,10)),'dimensions':[],'head_labels':['aligned05','aligned10'],
        'checkpoints':[100,400],'candidates':5,'validation_day':'10 и12 (development)','test_day':13,'target_score':80,
        'development_days':[10,12],
        'source_contacts_day':2,'targets_days':list(range(3,10)),'loss_geometry':'actual fused output',
        'source_hours':'11:00–12:00','target_hours':'12:30–13:30',
        'weights':[0.5,1.0],'selection_metric':'minimum score on development10 and12; both must improve',
        'test_gate':'both development scores >=80','day12_is_development_after_previous_final_test':True,
        'original_scoreboard_comparable':False}
    if not (work/'protocol.json').exists():write_json(work/'protocol.json',protocol)
    if not (work/'inputs_ready.json').exists():
        run(work,'donor_inputs','experiments.fingerprint_aligned','--work',work,'--stage','inputs')
    jobs=[(label,'experiments.fingerprint_aligned',['--work',work,'--stage','train','--fusion-weight',weight])
          for label,weight in [('aligned05',0.5),('aligned10',1.0)] if not (work/label/'complete.json').exists()]
    pool(work,'donor_training',jobs)
    run(work,'search','experiments.fingerprint_aligned','--work',work,'--stage','selection')
    final_test(work)


if __name__=='__main__':
    try:main()
    except Exception as error:
        if '--work' in sys.argv:
            state(Path(sys.argv[sys.argv.index('--work')+1]),'failed',error=f'{type(error).__name__}: {error}')
        raise
