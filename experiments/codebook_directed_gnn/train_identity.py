"""Train only a late residual, keeping the original source geometry fixed."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from .model import CodebookDirectedGNN
from experiments.temporal_gatv2.objectives import episodic_identity
from experiments.temporal_donor_identity.train_identity import LegacyUnpickler

EXPECTED='a1c4bfa15ed0c227aefd981f59b87769aece146efb232da90617e4f561b10211'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def load_graph(path, hosts, *, directionless_bytes=False):
    d=joblib.load(path)
    if d['nodes'][:len(hosts)]!=hosts or d['manifest']['parts']!=6:
        raise ValueError('graph host order or interval count mismatch')
    x=torch.tensor(d['totals'],dtype=torch.float32)
    if directionless_bytes:
        mean=x[:,:,:2].sum(dim=-1,keepdim=True)/2
        x[:,:,:2]=mean
    graphs=[]
    for e in d['graphs']:
        item={}
        for k,v in e.items():
            if k.endswith('_attr'):
                v=v.copy()
                if directionless_bytes:
                    mean=np.log1p(np.expm1(v[:,1:3]).sum(axis=1)/2)
                    v[:,1]=mean;v[:,2]=mean
                v[:,:6]/=10.
            item[k]=torch.from_numpy(v)
        graphs.append(item)
    return x,graphs


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True);p.add_argument('--windows',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=24)
    p.add_argument('--batch',type=int,default=256);p.add_argument('--seed',type=int,default=17)
    p.add_argument('--no-graph',action='store_true');p.add_argument('--archive',type=Path,default=Path('/private/tmp/pt-original-multiday'))
    p.add_argument('--directionless-bytes',action='store_true')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);torch.set_num_threads(3);torch.manual_seed(a.seed)
    rng=np.random.default_rng(a.seed)
    import experiments
    experiments.__path__.insert(0,str(a.archive/'experiments'));sys.path.insert(0,str(a.archive))
    os.environ['PT_ML_ROOT']=str(a.data)
    cp=a.data/'fingerprint_codebook_recent/final/model.pkl'
    if sha(cp)!=EXPECTED:raise ValueError('wrong baseline checkpoint')
    with cp.open('rb') as f:base=LegacyUnpickler(f).load()
    hosts=pd.read_csv(a.data/'late_test/processed/cohort.csv',dtype=str).host_id.tolist()
    ids=pd.Index(hosts);source=base.source.loc[ids].to_numpy(dtype=np.float32)
    frames={};active={}
    for day in (10,11,12):
        raw=np.load(a.data/f'fingerprint_codebook_recent/history/historical{day:02}.npy',mmap_mode='r')
        active[day]=np.flatnonzero(np.load(a.data/f'fingerprint_codebook_recent/history/future_active{day:02}.npy'))
        frames[day]=base.adapter.transform(raw).astype(np.float32)
    del base
    model=CodebookDirectedGNN();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    history=[]
    protocol={'base_sha256':EXPECTED,'train_days':[10,11,12], 'source':'frozen original codebook',
      'steps':a.steps,'batch':a.batch,'seed':a.seed,'use_graph':not a.no_graph,'directionless_bytes':a.directionless_bytes,
      'development':13,'test14_16_used':False,'base_previously_fitted_until_day12':True,
      'calibration':'batch stop-gradient procrustes; calibration rank at most batch/2 - 1',
      'phase':'late Identity residual only; donor/source residual stays exactly zero'}
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for step in range(1,a.steps+1):
        day=(10,11,12)[(step-1)%3]
        graph_path=a.windows/f'day{day:02}.joblib'
        x,edges=load_graph(graph_path,hosts,directionless_bytes=a.directionless_bytes)
        positions=rng.choice(active[day],min(a.batch,len(active[day])),replace=False)
        q=torch.from_numpy(positions); target=torch.from_numpy(source[positions]);old=torch.from_numpy(frames[day][positions])
        model.train();opt.zero_grad(set_to_none=True)
        encoded=model.graph(x,edges,n_output=len(hosts),use_graph=not a.no_graph)
        predicted=model.late(old,encoded[q])
        loss=episodic_identity(target,predicted)+10.*(predicted-old).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        row={'step':step,'day':day,'loss':float(loss.detach())};history.append(row)
        print('training',row,flush=True)
        del x,edges,encoded,loss
    checkpoint=a.output/'model.pt'
    torch.save({'state_dict':model.state_dict(),'protocol':protocol},checkpoint)
    (a.output/'history.json').write_text(json.dumps(history,indent=2))
    (a.output/'completed.json').write_text(json.dumps({'checkpoint_sha256':sha(checkpoint),
       'steps':a.steps,'dev_used':False,'test_used':False},indent=2))


if __name__=='__main__':main()
