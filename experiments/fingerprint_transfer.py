"""Прямой sampled ranking новых собеседников через дифференцируемый перенос контактов."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.preprocessing import normalize, StandardScaler
from torch.nn import functional as F

from experiments.fingerprint_donors import DonorHead, peer_matrix
from experiments.fingerprint_donor_search import (INCUMBENT, SOURCE, download,
                                                  run, search)
from experiments.fingerprint_expanded_search import pool
from experiments.fingerprint_memory import read_fingerprints
from experiments.fingerprint_search import dump, state, write_json
from experiments.multiday_gain import check_chronology
from pipeline.data import ProcessedData


DONORS=SOURCE/'fingerprint_donors'
ORIGINAL=SOURCE/'fingerprint_memory'


class PartnerPair:
    def __init__(self, source, following, hosts, vocabulary, active=None):
        self.source=source.tocsr().astype(np.float32)
        self.following=following.tocsr().astype(np.float32)
        new=self.following.sign()-self.following.sign().multiply(self.source.sign())
        new.eliminate_zeros()
        mass=np.asarray(self.source.sum(0),dtype=np.float64).ravel()
        self.available=mass>0
        new=new.multiply(self.available[None,:]).tocsr(); new.eliminate_zeros()
        self.new=new
        usable=np.diff(new.indptr)>0
        if active is not None: usable &= active
        self.active=np.flatnonzero(usable)
        self.own=pd.Index(vocabulary).get_indexer(hosts)
        if not len(self.active): raise ValueError('Нет прогнозируемых новых собеседников')
        uniform=self.available.astype(float)/self.available.sum()
        popularity=mass**0.75; popularity/=popularity.sum()
        self.distribution=0.5*uniform+0.5*popularity
        self.distribution/=self.distribution.sum(dtype=np.float64)
        # Один общий множитель сохраняет ранги официального W@M.
        self.scaled=self.source/max(float(self.source.data.max()),1.0)

    def sample(self,rng,batch=128,negatives=16):
        queries=rng.choice(self.active,min(batch,len(self.active)),replace=False)
        positive=np.array([rng.choice(self.new.indices[self.new.indptr[q]:self.new.indptr[q+1]]) for q in queries])
        forbidden=[]
        for q in queries:
            values=set(self.source.indices[self.source.indptr[q]:self.source.indptr[q+1]])
            values.update(self.following.indices[self.following.indptr[q]:self.following.indptr[q+1]])
            if self.own[q]>=0: values.add(int(self.own[q]))
            forbidden.append(values)
        neg=rng.choice(self.source.shape[1],(len(queries),negatives),p=self.distribution)
        for _ in range(128):
            bad=np.array([[int(p) in forbidden[i] for p in row] for i,row in enumerate(neg)])
            if not bad.any(): break
            neg[bad]=rng.choice(self.source.shape[1],int(bad.sum()),p=self.distribution)
        else: raise ValueError('Не удалось набрать отрицательных собеседников')
        candidates=np.column_stack([positive,neg])
        unique,inverse=np.unique(candidates,return_inverse=True)
        return queries, candidates, unique, inverse.reshape(candidates.shape)


def sparse_partner_scores(z,queries,source,partners,temperature=0.1):
    """Все source-доноры → выбранные партнёры, с градиентом до cosine геометрии."""
    queries=torch.as_tensor(queries,dtype=torch.long)
    logits=z[queries]@z.T/temperature
    logits[torch.arange(len(queries)),queries]=-1e9
    weights=torch.softmax(logits,dim=1)
    matrix=source[:,partners].T.tocoo()
    sparse=torch.sparse_coo_tensor(np.vstack([matrix.row,matrix.col]),
                                  matrix.data.astype(np.float32),matrix.shape,check_invariants=True).coalesce()
    return torch.sparse.mm(sparse,weights.T).T


def inputs(work):
    if (work/'inputs_ready.json').exists(): return
    metadata=json.loads((ORIGINAL/'bank_metadata.json').read_text())
    training=[ProcessedData(Path(p)) for p in metadata['training_sources']]
    check_chronology(training,9*86400+12.5*3600)
    hosts=training[0].cohort
    reference_hosts=json.loads((DONORS/'hosts.json').read_text())
    if hosts!=reference_hosts: raise ValueError('Изменился порядок исходной когорты')
    series,events=[],[]
    for day,data in zip(range(2,10),training):
        counts=read_fingerprints(data,'train',hosts,ORIGINAL/'count_cache')
        series.append({'counts':{'peers':counts['counts']['peers']}})
        events.append(counts['events'])
        del counts; gc.collect()
        print(json.dumps({'prepared_peer_day':day}),flush=True)
    vocabulary=sorted(set().union(*[set(v['counts']['peers'].index.get_level_values('t').str[2:]) for v in series]))
    matrices=[peer_matrix(v,hosts,vocabulary) for v in series]
    del series
    pairs=[]
    for i in range(7):
        source=matrices[i].multiply((events[i]>=20)[:,None]).tocsr()
        active=(events[i]>=20)&(events[i+1]>=20)
        pair=PartnerPair(source,matrices[i+1],hosts,vocabulary,active)
        sp.save_npz(work/f'source{i}.npz',source)
        sp.save_npz(work/f'future{i}.npz',matrices[i+1])
        np.save(work/f'active{i}.npy',active)
        pairs.append({'source_day':i+2,'target_day':i+3,'active_queries':len(pair.active)})
    # Уже подготовленные исторические входы; ни validation, ни test сюда не читаются.
    for name in ['features.npy','hosts.json']:
        target=work/name
        if not target.exists(): target.symlink_to(DONORS/name)
    write_json(work/'vocabulary.json',vocabulary)
    write_json(work/'inputs_ready.json',{'n_hosts':len(hosts),'pairs':pairs,
        'validation_used_for_training':False,'objective':'sampled new partner ranking through sparse source contacts',
        'feature_inputs':str(DONORS/'features.npy')})


def train(work,identity_weight,epochs=400,checkpoints=(100,400)):
    torch.set_num_threads(max(1,int(os.environ.get('PT_TORCH_THREADS','4'))))
    label='donor0' if identity_weight==0 else 'donor025'
    out=work/label; out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists(): return
    raw=np.load(work/'features.npy')
    scaler=StandardScaler().fit(raw.reshape(-1,raw.shape[-1]))
    head=DonorHead(scaler)
    xs=[torch.from_numpy(normalize(scaler.transform(x)).astype(np.float32)) for x in raw]
    hosts=json.loads((work/'hosts.json').read_text()); vocab=json.loads((work/'vocabulary.json').read_text())
    pairs=[PartnerPair(sp.load_npz(work/f'source{i}.npz'),sp.load_npz(work/f'future{i}.npz'),hosts,vocab,
                       np.load(work/f'active{i}.npy')) for i in range(len(xs)-1)]
    optimizer=torch.optim.AdamW(head.net.parameters(),lr=5e-4,weight_decay=1e-3)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
    rng=np.random.default_rng(17)
    for epoch in range(1,epochs+1):
        head.net.train(); i=int(rng.integers(len(pairs)))
        queries,candidates,unique,inverse=pairs[i].sample(rng)
        z=head.net(xs[i])
        scores=sparse_partner_scores(z,queries,pairs[i].scaled,unique)
        sampled=scores.gather(1,torch.from_numpy(inverse))
        logits=sampled.clamp_min(1e-30).log()
        partner=F.cross_entropy(logits,torch.zeros(len(queries),dtype=torch.long))
        identity=torch.zeros(())
        if identity_weight:
            next_z=head.net(xs[i+1]); q=torch.from_numpy(queries)
            identity=F.cross_entropy(z[q]@next_z.T/0.1,q)
        loss=partner+identity_weight*identity
        if not torch.isfinite(loss): raise FloatingPointError('Нечисловая потеря прямого transfer')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        nnorm=torch.nn.utils.clip_grad_norm_(head.net.parameters(),3)
        if not torch.isfinite(nnorm): raise FloatingPointError('Нечисловой градиент transfer')
        optimizer.step(); schedule.step()
        if epoch%10==0:
            rank=1+(sampled.detach()[:,1:]>sampled.detach()[:,:1]).sum(1)
            print(json.dumps({'epoch':epoch,'loss':float(loss.detach()),'partner_loss':float(partner.detach()),
                'identity_loss':float(identity.detach()),'training_sample_mrr':float((1/rank.float()).mean()),
                'note':'training subset of17 candidates, not validation MRR'}),flush=True)
        if epoch in checkpoints: dump(head,out/f'epoch{epoch}.pkl')
    write_json(out/'complete.json',{'epochs':list(checkpoints),'identity_weight':identity_weight})


def finish(work):
    summary=json.loads((work/'selection_summary.json').read_text())
    if not summary['test_will_run']:
        state(work,'complete',result='no_material_validation_gain',**summary); return
    # Предыдущий прогон скачал13 полностью; проверяем размер по его успешному логу.
    import re
    raw=SOURCE/'multiday/raw/netflow_day-13.bz2'
    text=(DONORS/'download13.log').read_text() if (DONORS/'download13.log').exists() else ''
    match=re.search(r'netflow_day-13\.bz2: (\d+) байт',text)
    if not (match and raw.exists() and raw.stat().st_size==int(match.group(1))):
        state(work,'download13'); download(work)
    if not (work/'day13/manifest.json').exists():
        run(work,'prepare_day13','experiments.multiday_prepare','--raw',raw,'--base',
            SOURCE/'multiday/processed','--out',work/'day13','--day',13)
    run(work,'reference_test','experiments.multiday_reference','--base',SOURCE/'multiday/processed',
        '--target',work/'day13','--out',work/'paired_test','--reference',work/'reference_test','--window','test')
    from experiments.fingerprint_quick import candidate_directories
    run(work,'final_evaluation','experiments.multiday_finish','--candidates',*candidate_directories(work),
        '--data',work/'paired_test','--reference',work/'reference_test/test','--out',work/'final',
        '--incumbent',INCUMBENT,'--test-label','day13, 12:30–13:30','--protocol-text',
        'Замороженный энкодер32ч + прямая transfer-голова на исторических02→03…08→09. Validation10, test13 после фиксации.')
    state(work,'complete',report=str(work/'final/REPORT.md'),**json.loads((work/'final/summary.json').read_text()))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work',required=True,type=Path)
    ap.add_argument('--stage',choices=['all','inputs','train','search'],default='all')
    ap.add_argument('--identity-weight',type=float,choices=[0.0,0.25],default=0)
    args=ap.parse_args(); work=args.work; work.mkdir(parents=True,exist_ok=True)
    if args.stage=='inputs': inputs(work); return
    if args.stage=='train': train(work,args.identity_weight); return
    if args.stage=='search': search(work); return
    if (work/'final/REPORT.md').exists():
        state(work,'complete',**json.loads((work/'final/summary.json').read_text())); return
    protocol={'train_days':list(range(2,10)),'dimensions':[],'head_labels':['donor0','donor025'],
        'checkpoints':[100,400],'candidates':13,'validation_day':10,'test_day':13,'target_score':80,
        'objective':'sampled direct ranking new partners using differentiable weighted neighbor contacts',
        'metric_difference':'training softmax over all donor hosts; evaluation original top10 positive-cosine donors',
        'sampling':'16 negatives exclude own host, all known source and future contacts',
        'test_gate':{'validation_score_gain':1.0,'partner_new_mrr_gain':0.01},
        'test_policy':'day13 once after freeze if gate passes','original_scoreboard_comparable':False}
    if not (work/'protocol.json').exists(): write_json(work/'protocol.json',protocol)
    if not (work/'inputs_ready.json').exists():
        run(work,'donor_inputs','experiments.fingerprint_transfer','--work',work,'--stage','inputs')
    jobs=[(label,'experiments.fingerprint_transfer',['--work',work,'--stage','train','--identity-weight',weight])
          for label,weight in [('donor0',0),('donor025',0.25)] if not (work/label/'complete.json').exists()]
    pool(work,'donor_training',jobs)
    run(work,'search','experiments.fingerprint_transfer','--work',work,'--stage','search')
    finish(work)


if __name__=='__main__':
    try: main()
    except Exception as error:
        if '--work' in sys.argv:
            state(Path(sys.argv[sys.argv.index('--work')+1]),'failed',error=f'{type(error).__name__}: {error}')
        raise
