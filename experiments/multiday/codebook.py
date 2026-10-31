"""Трансдуктивные профили источника + отдельный адаптер наблюдаемого трафика."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sp
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from torch.nn import functional as F

from experiments.multiday.aligned import TRANSFER,aligned_pairs
from experiments.multiday.bigraph import contact_scores
from experiments.multiday.donor_search import INCUMBENT,PREVIOUS,SOURCE,run
from experiments.multiday.expanded_search import pool,slim
from experiments.multiday.listwise import candidate_masks,listwise_loss
from experiments.multiday.transfer import PartnerPair
from experiments.multiday.search import ArrayEmbeddings,dump,load,state,write_json
from experiments.multiday.finish import freeze_selection
from experiments.multiday.gain import evaluate
from pipeline.data import ProcessedData


HISTORY=SOURCE/'fingerprint_aligned'
VALIDATION=SOURCE/'fingerprint_memory'
CONFIGS=[('codebook1',1.),('codebook4',4.)]
CHECKPOINTS=(100,400)


def revision():
    names=['codebook.py','listwise.py','aligned.py',
           'bigraph.py','transfer.py','codebook_recent.py']
    return {n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in names}


def verify(work):
    record=json.loads((work/'training_revision.json').read_text())
    if record['sources']!=revision() or record['torch']!=str(torch.__version__):
        raise RuntimeError('Код обучения изменился после фиксации версии')


def neighbor_loss(z,teacher,queries,temperature=0.05):
    q=torch.as_tensor(queries,dtype=torch.long)
    student=z[q]@z.T/temperature
    target=teacher[q]@teacher.T/temperature
    student[torch.arange(len(q)),q]=-1e9
    target[torch.arange(len(q)),q]=-1e9
    return F.kl_div(F.log_softmax(student,dim=1),F.softmax(target,dim=1),reduction='batchmean')


class TrafficAdapter:
    def __init__(self,ridge,blend):
        self.ridge,self.blend=ridge,float(blend)

    def transform(self,values):
        x=normalize(np.asarray(values,dtype=np.float32))
        return normalize((1-self.blend)*x+self.blend*normalize(self.ridge.predict(x))).astype(np.float32)


class CodebookEncoder:
    """ID нужен лишь для известного training-профиля. Query кодируется из трафика."""
    def __init__(self,base,source,adapter):
        self.base,self.source,self.adapter=base,source,adapter

    def embed(self,data,window,hosts):
        if window=='train':return self.source.loc[hosts].copy()
        values=self.base.embed(data,window,hosts)
        return pd.DataFrame(self.adapter.transform(values.to_numpy()),index=values.index).add_prefix('emb_')


def fit_adapter(history,populations,source,target_days=None,half_life=None):
    if len(history)<3:raise ValueError('Для адаптера нужны хотя бы три исторических окна')
    target_days=list(target_days or range(3,3+len(history)))
    def fit(indices):
        indices=list(indices)
        xs=[normalize(history[i][populations[i]]) for i in indices]
        ys=[source[populations[i]] for i in indices]
        weights=None
        if half_life is not None:
            weights=np.concatenate([np.full(len(x),2.**(-(max(target_days)-target_days[i])/half_life)) for i,x in zip(indices,xs)])
            weights*=len(weights)/weights.sum()
        return Ridge(alpha=1.).fit(np.vstack(xs),np.vstack(ys),sample_weight=weights)
    # Только обучающая история. Это диагностика, не независимый validation.
    early=fit(range(len(history)-2))
    records=[]
    for blend in [0.,0.5,1.]:
        adapter=TrafficAdapter(early,blend);scores=[]
        for i in [len(history)-2,len(history)-1]:
            pop=populations[i];later=adapter.transform(history[i][pop]);a=source[pop]
            sim=a@later.T
            scores.append(float((sim.argmax(1)==np.arange(len(a))).mean()))
        records.append({'blend':blend,'historical_holdout_days':target_days[-2:],
            'historical_holdout_raw_identity':float(np.mean(scores))})
    best=min(records,key=lambda r:(-r['historical_holdout_raw_identity'],r['blend']))
    return TrafficAdapter(fit(range(len(history))),best['blend']),records


def training_pairs(history,hosts,vocabulary,target_days):
    source=sp.load_npz(TRANSFER/'source0.npz')
    source_active=np.diff(source.indptr)>0
    pairs=[]
    for day in target_days:
        population=np.load(history/f'future_active{day:02d}.npy')
        matrix=source.multiply(population[:,None]).tocsr();matrix.eliminate_zeros()
        pairs.append(PartnerPair(matrix,sp.load_npz(history/f'future{day:02d}.npz'),hosts,vocabulary,source_active&population))
    return pairs


def train(work,regularization,epochs=400,history_root=None,target_days=tuple(range(3,10)),half_life=None):
    verify(work);torch.set_num_threads(4);torch.manual_seed(17)
    root=Path(history_root or HISTORY)
    configuration={'history_root':str(root.resolve()),'target_days':list(target_days),'half_life':half_life}
    if (work/'training_configuration.json').exists():
        if json.loads((work/'training_configuration.json').read_text())!=configuration:
            raise RuntimeError('Конфигурация истории изменилась после фиксации')
    label=dict((v,k) for k,v in CONFIGS)[regularization]
    out=work/label;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    hosts=json.loads((root/'hosts.json').read_text())
    vocab=json.loads((TRANSFER/'vocabulary.json').read_text())
    anchor=normalize(np.load(root/'anchor.npy')).astype(np.float32)
    teacher=torch.from_numpy(anchor)
    codes=torch.nn.Parameter(teacher.clone())
    optimizer=torch.optim.AdamW([codes],lr=3e-4,weight_decay=0.)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
    pairs=training_pairs(root,hosts,vocab,target_days)
    history=[np.load(root/f'historical{d:02d}.npy') for d in target_days]
    populations=[np.load(root/f'future_active{d:02d}.npy') for d in target_days]
    donors=[]
    for pair in pairs:
        m=pair.source.copy();m.data=np.log1p(m.data)
        m=m.multiply(1/np.maximum(np.asarray(m.sum(0)).ravel(),1e-12)[None,:]).tocsr()
        donors.append(m)
    rng=np.random.default_rng(17)
    probability=None
    if half_life is not None:
        if half_life<=0:raise ValueError('Half-life должен быть положительным')
        probability=2.**(-(max(target_days)-np.asarray(target_days))/half_life)
        probability/=probability.sum()
    for epoch in range(1,epochs+1):
        if psutil.virtual_memory().available<1.5*1024**3:raise MemoryError('Свободная RAM меньше1.5 ГБ')
        i=int(rng.integers(len(pairs)) if probability is None else rng.choice(len(pairs),p=probability));pair=pairs[i]
        queries=rng.choice(pair.active,min(64,len(pair.active)),replace=False)
        valid,positive=candidate_masks(pair,queries)
        keep=positive.any(1).numpy();queries=queries[keep];valid=valid[keep];positive=positive[keep]
        if not len(queries):raise ValueError('Нет допустимых новых целей')
        q=torch.from_numpy(queries);z=F.normalize(codes,dim=1)
        scores=contact_scores(z,queries,pair.scaled,np.arange(len(vocab)),hard_mix=0.9)
        partner=listwise_loss(scores,valid,positive)
        preserve=neighbor_loss(z,teacher,queries)
        targets=(pair.new[queries]@donors[i].T).toarray()
        targets[np.arange(len(q)),queries]=0
        targets=normalize(targets,norm='l1').astype(np.float32)
        logits=z[q]@z.T/0.1;logits[torch.arange(len(q)),q]=-1e9
        donor=-(torch.from_numpy(targets)*F.log_softmax(logits,dim=1)).sum(1).mean()
        anchor_loss=(1-(z*teacher).sum(1)).mean()
        loss=partner+regularization*preserve+0.25*donor+0.1*anchor_loss
        if not torch.isfinite(loss):raise FloatingPointError('Нечисловая потеря codebook')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        if not torch.isfinite(codes.grad).all():raise FloatingPointError('Нечисловой градиент codebook')
        torch.nn.utils.clip_grad_norm_([codes],3);optimizer.step();schedule.step()
        if epoch%10==0:
            print(json.dumps({'epoch':epoch,'loss':float(loss.detach()),'partner_loss':float(partner.detach()),
                'neighbor_kl':float(preserve.detach()),'donor_loss':float(donor.detach()),
                'anchor_loss':float(anchor_loss.detach()),'neighbor_weight':regularization}),flush=True)
        if epoch in CHECKPOINTS:
            source=F.normalize(codes,dim=1).detach().numpy()
            with threadpool_limits(4):adapter,records=fit_adapter(history,populations,source,target_days,half_life)
            dump({'source':source,'adapter':adapter,'hosts':hosts},out/f'epoch{epoch}.pkl')
            write_json(out/f'adapter{epoch}.json',records)
    write_json(out/'complete.json',{'epochs':list(CHECKPOINTS),'neighbor_weight':regularization,
        'trainable_source_parameters':int(codes.numel()),'query_input':'observed traffic vectors, no ID lookup'})


def selection(work):
    verify(work)
    if (work/'selection_summary.json').exists():return
    old=json.loads((PREVIOUS/'final/frozen_selection.json').read_text())['winner']
    directory=Path(old['directory'])
    hosts=json.loads((HISTORY/'hosts.json').read_text())
    cells={
        'day10':(VALIDATION/'paired_validation',VALIDATION/'reference_validation/validation','validation',directory/'later.csv'),
        'day12':(PREVIOUS/'paired_test',PREVIOUS/'reference_test/test','test',PREVIOUS/'final/winner/later.csv')}
    baseline={'day10':old['score'],'day12':json.loads((PREVIOUS/'final/summary.json').read_text())['scores']['winner']}
    control=work/'candidates/incumbent';control.mkdir(parents=True,exist_ok=True)
    row={'name':'incumbent','score':min(baseline.values()),'scores_by_day':baseline}
    write_json(control/'best.json',row)
    if not (control/'best.pkl').exists():(control/'best.pkl').symlink_to(INCUMBENT)
    paths=[control];records=[row]
    for label,_ in CONFIGS:
        for epoch in CHECKPOINTS:
            name=f'{label}_e{epoch}';out=work/'candidates'/name
            saved=out/'assessment.json'
            if saved.exists():
                records.append(json.loads(saved.read_text()))
                if (out/'best.json').exists():paths.append(out)
                continue
            checkpoint=load(work/label/f'epoch{epoch}.pkl')
            if checkpoint['hosts']!=hosts:raise ValueError('Когорта source-профилей изменилась')
            train=pd.DataFrame(checkpoint['source'],index=hosts).add_prefix('emb_')
            scores={};identities={};partners={}
            for day,(data_path,ref,window,later_file) in cells.items():
                completed=out/day/'scored.json'
                if completed.exists():
                    result=json.loads(completed.read_text())
                else:
                    if psutil.virtual_memory().available<2*1024**3:raise MemoryError('Перед оценкой свободно меньше2 ГБ')
                    data=ProcessedData(data_path)
                    b=pd.read_csv(later_file,index_col=0);b.index=b.index.astype(str)
                    if data.cohort!=hosts or set(b.index)!=set(data.evaluable(window)):
                        raise ValueError('Кэш не соответствует фиксированной когорте')
                    values=checkpoint['adapter'].transform(b.to_numpy(dtype=np.float32))
                    later=pd.DataFrame(values,index=b.index).add_prefix('emb_')
                    state(work,'validation',candidate=f'{name}/{day}',completed=len(records)-1,total=4)
                    score,r=evaluate(ArrayEmbeddings(train,later),data,window,ref,out/day,f'{name}_{day}')
                    result={'score':score,'identity':r['metrics']['identity_top1'],'partner_new_mrr':r['metrics']['partner_new_mrr']}
                    write_json(completed,result)
                scores[day]=result['score'];identities[day]=result['identity'];partners[day]=result['partner_new_mrr']
                gc.collect()
            row={'name':name,'score':min(scores.values()),'scores_by_day':scores,'identity':min(identities.values()),
                'identity_by_day':identities,'partner_mrr_by_day':partners,'head':label,'epoch':epoch,
                'gains_by_day':{d:scores[d]-baseline[d] for d in baseline},
                'adapter_blend':checkpoint['adapter'].blend}
            write_json(saved,row);records.append(row)
            if all(scores[d]>=baseline[d] for d in baseline):
                write_json(out/'best.json',row);paths.append(out)
            write_json(work/'validation_results.json',records);print(json.dumps(row),flush=True)
    winner=min([json.loads((p/'best.json').read_text()) for p in paths],key=lambda r:(-r['score'],r['name']))
    if winner['name']!='incumbent':
        checkpoint=load(work/winner['head']/f"epoch{winner['epoch']}.pkl")
        source=pd.DataFrame(checkpoint['source'],index=hosts).add_prefix('emb_')
        dump(CodebookEncoder(slim(load(INCUMBENT)),source,checkpoint['adapter']),work/'candidates'/winner['name']/'best.pkl')
    freeze_selection(paths,work/'final')
    write_json(work/'validation_results.json',records)
    result={'winner':winner,'baseline':baseline,
        'validation_gain':min(winner['scores_by_day'][d]-baseline[d] for d in baseline),
        'test_will_run':False,'test_gate':'diagnostic development10/12; blind14 untouched'}
    write_json(work/'selection_summary.json',result)
    report=['# Обучаемые source-профили и временной адаптер','',
        'Обучение03–09, source-контакты02. Development10/12 уже открыт. Это не исходная таблица.',
        'Закрытый14 не читался.','', '| Модель | День10 | День12 |','|---|---:|---:|']
    for r in records:report.append(f"| {r['name']} | {r['scores_by_day']['day10']:.3f} | {r['scores_by_day']['day12']:.3f} |")
    report+=['',f"Выбрана {winner['name']}. Минимальный прирост к контролю: {result['validation_gain']:+.3f}."]
    (work/'REPORT.md').write_text('\n'.join(report)+'\n')
    state(work,'complete',**result)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work',type=Path,required=True)
    ap.add_argument('--stage',choices=['all','train','selection'],default='all')
    ap.add_argument('--neighbor-weight',type=float,choices=[1.,4.],default=1.)
    args=ap.parse_args();work=args.work;work.mkdir(parents=True,exist_ok=True)
    if args.stage=='train':train(work,args.neighbor_weight);return
    if args.stage=='selection':selection(work);return
    if not (work/'protocol.json').exists():write_json(work/'protocol.json',{
        'train_days':list(range(2,10)),'head_labels':[k for k,_ in CONFIGS],
        'checkpoints':list(CHECKPOINTS),'candidates':5,'validation_day':'10 и12 (development)','test_day':14,
        'target_score':80,'original_scoreboard_comparable':False,
        'model':'trainable source codebook + asymmetric ridge traffic adapter',
        'source_contacts_day':2,'training_target_days':list(range(3,10)),
        'query_identity_lookup':False,'adapter_alpha':1.,'adapter_blend_selection':'historical08/09 only',
        'neighbor_weights':[1.,4.],'test_policy':'diagnostic only; blind14 untouched'} )
    if not (work/'training_revision.json').exists():write_json(work/'training_revision.json',{'sources':revision(),'torch':str(torch.__version__)})
    verify(work)
    jobs=[(label,'experiments.multiday.codebook',['--work',work,'--stage','train','--neighbor-weight',weight])
        for label,weight in CONFIGS if not (work/label/'complete.json').exists()]
    pool(work,'donor_training',jobs,slots=2)
    run(work,'search','experiments.multiday.codebook','--work',work,'--stage','selection')


if __name__=='__main__':
    try:main()
    except Exception as error:
        if '--work' in sys.argv:state(Path(sys.argv[sys.argv.index('--work')+1]),'failed',error=f'{type(error).__name__}: {error}')
        raise
