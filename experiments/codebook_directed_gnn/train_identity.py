"""Первый этап: late-поправка при неизменном оригинальном source."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from .base import load_base,sha,EXPECTED
from .model import CodebookDirectedGNN
from experiments.temporal_gatv2.objectives import episodic_identity


TRAIN_SPANS={2:(126000,129600),10:(822600,826200),11:(909000,912600),12:(995400,999000)}


def load_graph(path,hosts,*,directionless_bytes=False,expected_span=None):
    d=joblib.load(path)
    if d['nodes'][:len(hosts)]!=hosts or d['manifest']['parts']!=6:
        raise ValueError('Неверный порядок узлов или число частей')
    if expected_span is not None and (d['manifest']['start'],d['manifest']['end'])!=expected_span:
        raise ValueError('Граф не соответствует разрешённому train-окну')
    if d['totals'].shape[1:]!=(6,8) or len(d['graphs'])!=6:raise ValueError('Неверная схема графа')
    x=torch.tensor(d['totals'],dtype=torch.float32)
    if directionless_bytes:x[:,:,:2]=x[:,:,:2].sum(-1,keepdim=True)/2
    graphs=[]
    for edges in d['graphs']:
        item={}
        for key,value in edges.items():
            value=value.copy()
            if key.endswith('_attr'):
                if directionless_bytes:
                    mean=np.log1p(np.expm1(value[:,1:3]).sum(1)/2)
                    value[:,1]=mean;value[:,2]=mean
                value[:,:6]/=10.
            item[key]=torch.from_numpy(value)
        graphs.append(item)
    return x,graphs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['data','windows','output','archive']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--steps',type=int,default=12);p.add_argument('--batch',type=int,default=256)
    p.add_argument('--seed',type=int,default=17);p.add_argument('--no-graph',action='store_true')
    p.add_argument('--directionless-bytes',action='store_true');a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);torch.set_num_threads(3);torch.manual_seed(a.seed)
    rng=np.random.default_rng(a.seed);base=load_base(a.data,a.archive)
    hosts=pd.read_csv(a.data/'late_test/processed/cohort.csv',dtype=str).host_id.tolist()
    if base.source.index.tolist()!=hosts:raise ValueError('Порядок indexed-истории отличается от оригинальной когорты')
    source=base.source.loc[hosts].to_numpy(dtype=np.float32);frames={};active={}
    for day in (10,11,12):
        raw=np.load(a.data/f'fingerprint_codebook_recent/history/historical{day:02}.npy',mmap_mode='r')
        active[day]=np.flatnonzero(np.load(a.data/f'fingerprint_codebook_recent/history/future_active{day:02}.npy'))
        frames[day]=base.adapter.transform(raw).astype(np.float32)
    del base
    model=CodebookDirectedGNN();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    history=[];protocol={'base_sha256':EXPECTED,'train_days':[10,11,12],'steps':a.steps,
        'batch':a.batch,'seed':a.seed,'use_graph':not a.no_graph,'directionless_bytes':a.directionless_bytes,
        'development':13,'test14_16_used':False,'base_previously_fitted_until_day12':True,
        'phase':'late Identity residual only; source residual zero'}
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for step in range(1,a.steps+1):
        day=(10,11,12)[(step-1)%3];x,edges=load_graph(a.windows/f'day{day:02}.joblib',hosts,
            directionless_bytes=a.directionless_bytes,expected_span=TRAIN_SPANS[day])
        indices=rng.choice(active[day],min(a.batch,len(active[day])),replace=False)
        q=torch.from_numpy(indices);target=torch.from_numpy(source[indices]);old=torch.from_numpy(frames[day][indices])
        model.train();opt.zero_grad(set_to_none=True)
        encoded=model.graph(x,edges,n_output=len(hosts),use_graph=not a.no_graph)
        predicted=model.late(old,encoded[q])
        loss=episodic_identity(target,predicted)+10.*(predicted-old).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('Нечисловая потеря')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        history.append({'step':step,'day':day,'loss':float(loss.detach())});print(history[-1],flush=True)
    cp=a.output/'model.pt';torch.save({'state_dict':model.state_dict(),'protocol':protocol},cp)
    (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    (a.output/'completed.json').write_text(json.dumps({'checkpoint_sha256':sha(cp),
        'steps':a.steps,'dev_used':False,'test_used':False},indent=2)+'\n')


if __name__=='__main__':main()
