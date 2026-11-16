"""Train-only sampled soft/hard voting gap; not an oracle or a score bound."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.nn import functional as F
from .common.data import Window,read_window,contacts
from .run import soft_topk


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
    torch.set_num_threads(2);saved=torch.load(a.run/'model.pt',weights_only=False);hosts=saved['hosts'];idx=pd.Index(hosts)
    source=F.normalize(saved['state']['source'],dim=1).detach()
    obs=read_window(Window('source02',2,126000,129600,'runs/temporal_stable_hybrid_v2_exact/processed/flows/train.csv'),hosts,'runs/frozen_identity_local/cache')
    rows=contacts(obs);vocab=pd.Index(sorted({p for r in rows for p in r}));rr=[];cc=[];vv=[]
    for i,row in enumerate(rows):
        for p,v in row.items():rr.append(i);cc.append(vocab.get_loc(p));vv.append(v)
    matrix=sp.csr_matrix((vv,(rr,cc)),shape=(len(hosts),len(vocab)),dtype=float)
    results=[];rng=np.random.default_rng(17)
    for day,path in [(11,'fingerprint_memory/reference_test/test'),(12,'fingerprint_expanded/reference_test/test')]:
        ref=Path('PT_PROJECT_ML')/path;active=idx.get_indexer(pd.read_csv(ref/'eval_hosts.csv').host_id)
        pp=pd.read_csv(ref/'partners.csv');pp=pp[pp.window=='target'];targets={}
        for h,g in pp.groupby('host_id'):
            q=idx.get_loc(h);pos=vocab.get_indexer(g.partner);pos=pos[pos>=0];pos=np.setdiff1d(pos,matrix[q].indices)
            own=vocab.get_indexer([h])[0];pos=pos[pos!=own]
            if len(pos):targets[q]=pos
        queries=rng.choice(sorted(targets),min(64,len(targets)),replace=False)
        sim=source[queries]@source[active].T;sim=sim.masked_fill(torch.tensor(queries[:,None]==active[None,:]),-20)
        with torch.no_grad():soft=soft_topk(sim).numpy()
        top=torch.topk(sim,10,dim=1).indices.numpy();weights=np.maximum(np.take_along_axis(sim.numpy(),top,axis=1),0)
        hard=np.zeros_like(soft);np.put_along_axis(hard,top,weights,axis=1)
        values={}
        for name,w in [('soft',soft),('hard',hard)]:
            scores=sp.csr_matrix(w)@matrix[active];rrs=[]
            for i,q in enumerate(queries):
                valid=np.ones(len(vocab),bool);valid[matrix[q].indices]=False
                own=vocab.get_indexer([hosts[q]])[0]
                if own>=0:valid[own]=False
                row=scores[i].toarray().ravel();best=row[targets[q]].max()
                greater=((row>best)&valid).sum();ties=((row==best)&valid).sum()
                rrs.append(1/(greater+(ties+1)/2))
            values[name+'_mrr']=float(np.mean(rrs))
        results.append({'day':day,'queries':len(queries),**values})
    (a.run/'train_voting_gap.json').write_text(json.dumps(results,indent=2));print(results)

if __name__=='__main__':main()
