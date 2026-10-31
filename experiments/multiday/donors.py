"""Учить геометрию соседей по их полезности для будущих НОВЫХ собеседников."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.preprocessing import normalize, StandardScaler
from torch import nn
from torch.nn import functional as F

from experiments.multiday.memory import read_fingerprints
from experiments.multiday.search import BASE, SOURCE, dump, load, write_json
from experiments.multiday.gain import check_chronology
from experiments.overnight.model import T
from pipeline.data import ProcessedData


EXPANDED=SOURCE/'fingerprint_expanded'
ORIGINAL=SOURCE/'fingerprint_memory'


def peer_matrix(counts, hosts, vocabulary):
    series=counts['counts']['peers']
    tokens=series.index.get_level_values('t').astype(str)
    if not tokens.str.startswith(('o:','i:')).all():
        raise ValueError('Неизвестный формат направленного собеседника')
    rows=pd.Index(hosts).get_indexer(series.index.get_level_values('h'))
    columns=pd.Index(vocabulary).get_indexer(tokens.str[2:])
    if (rows<0).any() or (columns<0).any():
        raise ValueError('Агрегат вне обучающего словаря/когорты')
    return sp.csr_matrix((series.to_numpy(dtype=np.float32),(rows,columns)),shape=(len(hosts),len(vocabulary)))


def donor_targets(source, following, topk=32):
    """R[i,j]>0, если j знает нового будущего собеседника i. Диагональ запрещена."""
    if source.shape!=following.shape:
        raise ValueError('Разные когорты или словари соседних обучающих окон')
    source=source.tocsr().astype(np.float32)
    following=following.tocsr().astype(np.float32)
    new=following.sign()-following.sign().multiply(source.sign())
    new.eliminate_zeros()
    # Уменьшаем доминирование популярных контактов и огромных счётчиков.
    weighted=source.copy(); weighted.data=np.log1p(weighted.data)
    mass=np.asarray(weighted.sum(0)).ravel()
    weighted=weighted@sp.diags(1/np.maximum(mass,1e-12))
    relevance=(new@weighted.T).tocsr()
    relevance.setdiag(0); relevance.eliminate_zeros()
    for row in range(relevance.shape[0]):
        start,end=relevance.indptr[row:row+2]
        if end-start>topk:
            values=relevance.data[start:end]
            keep=np.argpartition(values,-topk)[-topk:]
            mask=np.ones(len(values),dtype=bool); mask[keep]=False
            values[mask]=0
    relevance.eliminate_zeros()
    return normalize(relevance,norm='l1').astype(np.float32)


def transfer_loss(z, queries, targets, temperature=0.1):
    """Вероятность полезных доноров среди всех других хостов; без прямого ID lookup."""
    logits=z[queries]@z.T/temperature
    logits[torch.arange(len(queries)),queries]=-1e9
    return -(targets*F.log_softmax(logits,dim=1)).sum(dim=1).mean()


class DonorNetwork(nn.Module):
    def __init__(self, inputs=256, dim=128):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(inputs,256),nn.LayerNorm(256),nn.GELU(),
                                  nn.Dropout(0.1),nn.Linear(256,dim))

    def forward(self,x):
        return F.normalize(self.layers(x),dim=1)


class DonorHead:
    def __init__(self, scaler, seed=17):
        torch.manual_seed(seed)
        self.scaler=scaler
        self.net=DonorNetwork(scaler.n_features_in_)

    def encode(self,values):
        self.net.eval()
        x=normalize(self.scaler.transform(normalize(np.asarray(values,dtype=np.float32)))).astype(np.float32)
        with torch.no_grad():
            result=self.net(torch.from_numpy(x)).numpy()
        return result


class DonorHybrid:
    def __init__(self,base,head,weight=0.5,input_dim=256):
        self.base,self.head,self.weight,self.input_dim=base,head,weight,input_dim

    def embed(self,data,window,hosts):
        anchor=self.base.embed(data,window,hosts)
        if not self.weight: return anchor
        features=self.head.encode(anchor.iloc[:,-self.input_dim:].to_numpy())
        values=np.hstack([T._block(anchor.to_numpy()),T._block(features)*self.weight])
        return pd.DataFrame(values,index=anchor.index).add_prefix('emb_')


def inputs(work):
    if (work/'inputs_ready.json').exists(): return
    metadata=json.loads((ORIGINAL/'bank_metadata.json').read_text())
    training=[ProcessedData(Path(p)) for p in metadata['training_sources']]
    validation_start=9*86400+12.5*3600
    check_chronology(training,validation_start)
    hosts=training[0].cohort
    if len(set(hosts))!=len(hosts): raise ValueError('Повторные ID в исходной когорте')
    # cohort.csv старого дня03 относится к его собственной активности.
    # Потоки в pipeline.prepare сохраняются ДО выбора когорты, без фильтрации.
    # Поэтому все агрегаты ниже явно строятся по исходным hosts дня02,
    # без объединения/пересечения с таблицами последующих дней.
    # Сначала только небольшие peer-агрегаты; большие context освобождаются по дням.
    peer_series,events=[],[]
    for day,data in zip(range(2,10),training):
        counts=read_fingerprints(data,'train',hosts,ORIGINAL/'count_cache')
        peer_series.append({'counts':{'peers':counts['counts']['peers']}})
        events.append(counts['events'])
        del counts; gc.collect()
        print(json.dumps({'prepared_peer_day':day}),flush=True)
    vocabulary=sorted(set().union(*[set(v['counts']['peers'].index.get_level_values('t').str[2:]) for v in peer_series]))
    matrices=[peer_matrix(v,hosts,vocabulary) for v in peer_series]
    del peer_series
    pairs=[]
    for i in range(len(matrices)-1):
        active=(events[i]>=20)&(events[i+1]>=20)
        source=matrices[i].multiply((events[i]>=20)[:,None]).tocsr()
        labels=donor_targets(source,matrices[i+1])
        labels=labels.multiply(active[:,None]).tocsr(); labels.eliminate_zeros()
        if not labels.nnz: raise ValueError(f'Нет новых собеседников в паре {i+2}→{i+3}')
        sp.save_npz(work/f'targets{i}.npz',labels)
        pairs.append({'source_day':i+2,'target_day':i+3,'active_queries':int((np.diff(labels.indptr)>0).sum()),'positive_donors':labels.nnz})
        print(json.dumps(pairs[-1]),flush=True)
    del matrices; gc.collect()
    # Голова обучается по историческим признакам уже замороженного энкодера.
    bank=load(EXPANDED/'bank.pkl')
    if bank.hosts!=hosts: raise ValueError('Когорта bank отличается')
    if any(m['interval'][1]>validation_start for m in bank.training_manifests): raise ValueError('Будущее в pretrained bank')
    encoder=load(EXPANDED/'bag256/epoch300.pkl')
    xs=[]; n=len(hosts)
    for day in range(8):
        view=day*4+1  # Исторический час11:00–12:00, как у peer-меток.
        values=encoder.encode(bank.memory[view*n:(view+1)*n])
        xs.append(normalize(T._block(values)).astype(np.float32))
    np.save(work/'features.npy',np.stack(xs))
    write_json(work/'hosts.json',hosts)
    write_json(work/'inputs_ready.json',{'training_days':list(range(2,10)), 'n_hosts':n,
        'pairs':pairs,'validation_used_for_training':False,'features':'frozen 32h head256, historical hour11',
        'objective':'neighbors that expose new next-day partners; self and old contacts excluded'})


def train(work,identity_weight,epochs=400,checkpoints=(100,400)):
    torch.set_num_threads(max(1,int(os.environ.get('PT_TORCH_THREADS','4'))))
    label='donor0' if identity_weight==0 else 'donor025'
    out=work/label; out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists(): return
    raw=np.load(work/'features.npy')
    scaler=StandardScaler().fit(raw.reshape(-1,raw.shape[-1]))
    head=DonorHead(scaler)
    xs=[torch.from_numpy(normalize(scaler.transform(x)).astype(np.float32)) for x in raw]
    labels=[sp.load_npz(work/f'targets{i}.npz') for i in range(len(xs)-1)]
    active=[np.flatnonzero(np.diff(y.indptr)>0) for y in labels]
    optimizer=torch.optim.AdamW(head.net.parameters(),lr=5e-4,weight_decay=1e-3)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
    rng=np.random.default_rng(17)
    for epoch in range(1,epochs+1):
        head.net.train(); i=int(rng.integers(len(labels)))
        chosen=rng.choice(active[i],min(512,len(active[i])),replace=False)
        queries=torch.from_numpy(chosen)
        target=torch.from_numpy(labels[i][chosen].toarray().astype(np.float32))
        z=head.net(xs[i])
        partner=transfer_loss(z,queries,target)
        identity=torch.zeros(())
        if identity_weight:
            next_z=head.net(xs[i+1])
            identity=F.cross_entropy(z[queries]@next_z.T/0.1,queries)
        loss=partner+identity_weight*identity
        if not torch.isfinite(loss): raise FloatingPointError('Нечисловая потеря donor head')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(head.net.parameters(),3)
        optimizer.step(); schedule.step()
        if epoch%10==0:
            print(json.dumps({'epoch':epoch,'loss':float(loss.detach()),'partner_loss':float(partner.detach()),
                              'identity_loss':float(identity.detach()),'pair':i,'identity_weight':identity_weight}),flush=True)
        if epoch in checkpoints: dump(head,out/f'epoch{epoch}.pkl')
    write_json(out/'complete.json',{'epochs':list(checkpoints),'identity_weight':identity_weight})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work',required=True,type=Path)
    ap.add_argument('--stage',choices=['inputs','train'],required=True)
    ap.add_argument('--identity-weight',type=float,choices=[0.0,0.25],default=0)
    args=ap.parse_args(); args.work.mkdir(parents=True,exist_ok=True)
    if args.stage=='inputs': inputs(args.work)
    else: train(args.work,args.identity_weight)


if __name__=='__main__': main()
