"""Второй этап: source-проекция по полезным историческим заменам."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from .model import CodebookDirectedGNN
from .base import sha
from .train_identity import load_graph,TRAIN_SPANS
from .swaps import replacement_pairs,swap_loss
from experiments.temporal_gatv2.episode import Episode


def history_profile(encoded,presence,days):
    cutoff=max(days);weight=torch.tensor([.5**((cutoff-d)/3.) for d in days])
    mask=torch.stack([presence[d] for d in days],1).float()*weight[None,:]
    return (torch.stack([encoded[d] for d in days],1)*mask[:,:,None]).sum(1)/mask.sum(1).clamp_min(1e-9)[:,None]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['data','windows','training','zero','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--steps',type=int,default=6);p.add_argument('--batch',type=int,default=4)
    p.add_argument('--seed',type=int,default=17);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    cp=a.training/'model.pt';f=json.loads((a.training/'completed.json').read_text())
    if sha(cp)!=f['checkpoint_sha256'] or f['dev_used'] or f['test_used']:raise ValueError('Неверный checkpoint')
    a.output.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
    state=torch.load(cp,map_location='cpu',weights_only=True);model=CodebookDirectedGNN()
    model.load_state_dict(state['state_dict']);model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    model.source_residual.weight.requires_grad_(True)
    if torch.count_nonzero(model.source_residual.weight):raise ValueError('Нужна нулевая source-проекция')
    hosts=pd.read_csv(a.data/'late_test/processed/cohort.csv',dtype=str).host_id.tolist()
    base=pd.read_csv(a.zero/'source.csv',index_col=0,float_precision='round_trip')
    if base.index.tolist()!=hosts:raise ValueError('Порядок source не совпал с indexed-историей')
    source=torch.tensor(base.to_numpy(),dtype=torch.float32);encoded={};presence={}
    with torch.no_grad():
        for day in (2,10,11,12):
            path=a.windows/f'day{day:02}.joblib';d=joblib.load(path)
            presence[day]=torch.tensor(d['totals'][:len(hosts),:,4:6].sum((1,2))>0)
            x,edges=load_graph(path,hosts,expected_span=TRAIN_SPANS[day])
            encoded[day]=model.graph(x,edges,n_output=len(hosts),use_graph=state['protocol']['use_graph']).detach()
    profiles={10:history_profile(encoded,presence,[2]),11:history_profile(encoded,presence,[2,10]),
              12:history_profile(encoded,presence,[2,10,11])}
    episodes={d:Episode(a.data,d,np.load(a.data/f'fingerprint_codebook_recent/history/future_active{d:02}.npy'))
              for d in (10,11,12)}
    if any(ep.hosts.tolist()!=hosts for ep in episodes.values()):raise ValueError('Порядок словаря хостов изменён')
    opt=torch.optim.AdamW([model.source_residual.weight],lr=.0003,weight_decay=.0001);history=[]
    for step in range(1,a.steps+1):
        day=(10,11,12)[(step-1)%3];ep=episodes[day]
        q=rng.choice(ep.eligible,min(a.batch,len(ep.eligible)),replace=False)
        z=model.source(source,profiles[day]);triples=replacement_pairs(z.detach().numpy(),q,ep)
        loss=swap_loss(z,triples)+10*(z-source).square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        history.append({'step':step,'day':day,'pairs':len(triples),'loss':float(loss.detach())})
        print(history[-1],flush=True)
    protocol={'steps':a.steps,'batch':a.batch,'seed':a.seed,'base_fitted_through_day12':True,
              'source_train_prefixes':{'10':[2],'11':[2,10],'12':[2,10,11]},
              'source_final_windows':[2,10,11,12],'day13_development_used':False,'test_used':False,
              'stage1_checkpoint_sha256':sha(cp),'frozen_graph_encoder':True}
    protocol['partner_order_sha256']={str(day):ep.order_sha256 for day,ep in episodes.items()}
    cp=a.output/'model.pt';torch.save({'state_dict':model.state_dict(),'protocol':protocol},cp)
    with torch.no_grad():values=model.source(torch.tensor(base.to_numpy()),history_profile(encoded,presence,[2,10,11,12])).numpy()
    pd.DataFrame(values,index=base.index,columns=base.columns).to_csv(a.output/'source.csv')
    (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    (a.output/'completed.json').write_text(json.dumps({'checkpoint_sha256':sha(cp),
        'source_sha256':sha(a.output/'source.csv'),'protocol':protocol},indent=2)+'\n')


if __name__=='__main__':main()
