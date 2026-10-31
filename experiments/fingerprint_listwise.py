"""Все кандидаты и любая правильная цель вместо случайных 16 отрицательных."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import psutil
import scipy.sparse as sp
import torch
from sklearn.preprocessing import StandardScaler, normalize
from torch.nn import functional as F

from experiments.fingerprint_aligned import (
    TRANSFER, aligned_pairs, block_tensor, fused_tensor, gallery_labels, selection,
)
from experiments.fingerprint_bigraph import contact_scores
from experiments.fingerprint_donors import DonorHead
from experiments.fingerprint_donor_search import SOURCE, run
from experiments.fingerprint_expanded_search import pool
from experiments.fingerprint_search import dump, state, write_json


HISTORY=SOURCE/'fingerprint_aligned'
CONFIGS=[('listwise05',0.5),('listwise10',1.0)]
CHECKPOINTS=(100,300)


def candidate_masks(pair,queries):
    valid=np.broadcast_to(pair.available,(len(queries),len(pair.available))).copy()
    for row,q in enumerate(queries):
        valid[row,pair.source.indices[pair.source.indptr[q]:pair.source.indptr[q+1]]]=False
        if pair.own[q]>=0:valid[row,pair.own[q]]=False
    positive=pair.new[queries].toarray()>0
    positive &= valid
    return torch.from_numpy(valid),torch.from_numpy(positive)


def listwise_loss(scores,valid,positive,temperature=0.2):
    if scores.shape!=valid.shape or valid.shape!=positive.shape:
        raise ValueError('Разные размеры скоров и масок')
    if not torch.all(positive.any(1)) or torch.any(positive & ~valid):
        raise ValueError('Каждый запрос должен иметь допустимую новую цель')
    logits=scores.clamp_min(1e-30).log()/temperature
    # Цель официального MRR — первый из любых правильных новых партнёров.
    return (torch.logsumexp(logits.masked_fill(~valid,-torch.inf),1)-
            torch.logsumexp(logits.masked_fill(~positive,-torch.inf),1)).mean()


def train(work,weight,epochs=300):
    torch.set_num_threads(4)
    torch.manual_seed(17)
    revision=work/'training_revision.json'
    if revision.exists():
        for name,digest in json.loads(revision.read_text())['sources'].items():
            if hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('Код изменился после фиксации обучения')
    label=dict((w,l) for l,w in CONFIGS)[weight]
    out=work/label;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    anchor=np.load(HISTORY/'anchor.npy');history=np.load(HISTORY/'historical.npy')
    populations=[np.load(HISTORY/f'future_active{d:02d}.npy') for d in range(2,10)]
    hosts=json.loads((HISTORY/'hosts.json').read_text())
    vocab=json.loads((TRANSFER/'vocabulary.json').read_text())
    pairs=aligned_pairs(HISTORY,hosts,vocab)
    raw=np.vstack([anchor,*[v[a] for v,a in zip(history,populations)]])
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
        if psutil.virtual_memory().available<1.5*1024**3:
            raise MemoryError('Меньше1.5 ГБ свободной RAM')
        head.net.train();i=int(rng.integers(7));pair=pairs[i]
        queries=rng.choice(pair.active,min(64,len(pair.active)),replace=False)
        valid,positive=candidate_masks(pair,queries)
        keep=positive.any(1).numpy();queries=queries[keep];valid=valid[keep];positive=positive[keep]
        if not len(queries):raise ValueError('Нет допустимых новых целей в batch')
        q=torch.from_numpy(queries)
        z=fused_tensor(base,head.net(x),weight)
        scores=contact_scores(z,queries,pair.scaled,np.arange(len(vocab)),hard_mix=0.9)
        partner=listwise_loss(scores,valid,positive)
        gallery,labels=gallery_labels(queries,populations[i+1])
        later=fused_tensor(past[i+1][gallery],head.net(hx[i+1][gallery]),weight)
        identity=F.cross_entropy(z[q]@later.T/0.1,torch.from_numpy(labels))
        preserve=F.mse_loss(z[q]@z.T,base_unit[q]@base_unit.T)
        loss=partner+0.25*identity+preserve
        if not torch.isfinite(loss):raise FloatingPointError('Нечисловая listwise потеря')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(head.net.parameters(),3)
        if not torch.isfinite(norm):raise FloatingPointError('Нечисловой listwise градиент')
        optimizer.step();schedule.step()
        if epoch%10==0:
            best=scores.detach().masked_fill(~positive,-torch.inf).max(1).values
            ranks=1+((scores.detach()>best[:,None])&valid).sum(1)
            print(json.dumps({'epoch':epoch,'loss':float(loss.detach()),'partner_loss':float(partner.detach()),
                'identity_loss':float(identity.detach()),'geometry_loss':float(preserve.detach()),
                'training_full_candidate_mrr':float((1/ranks.float()).mean()),
                'candidate_count':len(vocab),'fusion_weight':weight,
                'note':'training, not validation; optimistic tie handling'}),flush=True)
        if epoch in CHECKPOINTS:dump(head,out/f'epoch{epoch}.pkl')
    write_json(out/'complete.json',{'epochs':list(CHECKPOINTS),'weight':weight,'all_candidate_count':len(vocab)})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work',required=True,type=Path)
    ap.add_argument('--stage',choices=['all','train','selection'],default='all')
    ap.add_argument('--weight',type=float,choices=[0.5,1.0],default=0.5)
    args=ap.parse_args();work=args.work;work.mkdir(parents=True,exist_ok=True)
    if args.stage=='train':train(work,args.weight);return
    if args.stage=='selection':selection(work,CONFIGS,CHECKPOINTS);return
    write_json(work/'protocol.json',{'train_days':list(range(2,10)),
        'head_labels':[v[0] for v in CONFIGS],'checkpoints':list(CHECKPOINTS),
        'validation_day':'10 и12 (development)','test_day':14,'target_score':80,
        'loss':'full 29091 candidate listwise, any true new peer, 90% official top10 neighbor weights',
        'source_contacts_day':2,'target_training_days':list(range(3,10)),
        'test_policy':'no test in this diagnostic; day14 remains sealed',
        'original_scoreboard_comparable':False})
    if not (work/'training_revision.json').exists():
        modules=['fingerprint_listwise.py','fingerprint_aligned.py','fingerprint_bigraph.py','fingerprint_donors.py']
        write_json(work/'training_revision.json',{'sources':{n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in modules}})
    jobs=[(label,'experiments.fingerprint_listwise',['--work',work,'--stage','train','--weight',w])
          for label,w in CONFIGS if not (work/label/'complete.json').exists()]
    pool(work,'donor_training',jobs,slots=2)
    run(work,'search','experiments.fingerprint_listwise','--work',work,'--stage','selection')
    result=json.loads((work/'selection_summary.json').read_text())
    result['test_will_run']=False
    result['test_gate']='diagnostic only; blind14 untouched'
    write_json(work/'selection_summary.json',result)
    state(work,'complete',**result)


if __name__=='__main__':
    try:main()
    except Exception as error:
        import sys
        if '--work' in sys.argv:state(Path(sys.argv[sys.argv.index('--work')+1]),'failed',error=f'{type(error).__name__}: {error}')
        raise
