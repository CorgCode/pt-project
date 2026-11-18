"""Small historical donor-swap pilot over fixed graph encoder and codebook."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from .model import CodebookDirectedGNN
from .train_identity import load_graph,sha
from .swaps import replacement_pairs,swap_loss
from experiments.temporal_gatv2.episode import Episode


def main():
    p=argparse.ArgumentParser()
    for n in ('data','windows','training','zero','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--steps',type=int,default=6);p.add_argument('--batch',type=int,default=4)
    p.add_argument('--seed',type=int,default=17)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    complete=json.loads((a.training/'completed.json').read_text());cp=a.training/'model.pt'
    if sha(cp)!=complete['checkpoint_sha256'] or complete['dev_used'] or complete['test_used']:
        raise ValueError('invalid stage-one provenance')
    a.output.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(a.seed)
    rng=np.random.default_rng(a.seed)
    state=torch.load(cp,map_location='cpu',weights_only=True)
    model=CodebookDirectedGNN();model.load_state_dict(state['state_dict']);model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    model.source_residual.weight.requires_grad_(True)
    if torch.count_nonzero(model.source_residual.weight):raise ValueError('source residual must start at zero')
    hosts=pd.read_csv(a.data/'late_test/processed/cohort.csv',dtype=str).host_id.tolist()
    base=pd.read_csv(a.zero/'source.csv',dtype={'host_id':str},float_precision='round_trip').set_index('host_id').loc[hosts]
    source=torch.tensor(base.to_numpy(),dtype=torch.float32)
    enc={};available={}
    with torch.no_grad():
        for day in (2,10,11,12):
            path=a.windows/f'day{day:02}.joblib'
            data=joblib.load(path)
            available[day]=torch.tensor(data['totals'][:len(hosts),:,4:6].sum((1,2))>0)
            del data
            x,edges=load_graph(path,hosts)
            enc[day]=model.graph(x,edges,n_output=len(hosts),use_graph=state['protocol']['use_graph']).detach()
            print('encoded training day',day,flush=True)
            del x,edges
    def profile(days):
        cutoff=max(days); weights=torch.tensor([.5**((cutoff-d)/3.) for d in days])
        mask=torch.stack([available[d] for d in days],1).float()*weights[None,:]
        return (torch.stack([enc[d] for d in days],1)*mask[:,:,None]).sum(1)/mask.sum(1).clamp_min(1e-9)[:,None]
    profiles={10:profile([2]),11:profile([2,10]),12:profile([2,10,11])}
    episodes={}
    for day in (10,11,12):
        active=np.load(a.data/f'fingerprint_codebook_recent/history/future_active{day:02}.npy')
        episodes[day]=Episode(a.data,day,active)
        if episodes[day].hosts.tolist()!=hosts:raise ValueError('episode host order mismatch')
    opt=torch.optim.AdamW([model.source_residual.weight],lr=.0003,weight_decay=.0001)
    history=[]
    for step in range(1,a.steps+1):
        day=(10,11,12)[(step-1)%3];ep=episodes[day]
        q=rng.choice(ep.eligible,min(a.batch,len(ep.eligible)),replace=False)
        z=model.source(source,profiles[day])
        triples=replacement_pairs(z.detach().numpy(),q,ep,near=5,positive_cap=5)
        loss=swap_loss(z,triples)+10*(z-source).square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        row={'step':step,'day':day,'pairs':len(triples),'loss':float(loss.detach())}
        history.append(row);print('donor training',row,flush=True)
    checkpoint=a.output/'model.pt'
    protocol={'steps':a.steps,'batch':a.batch,'seed':a.seed,'base_fitted_through_day12':True,
              'source_train_prefixes':{'10':[2],'11':[2,10],'12':[2,10,11]},
              'source_final_windows':[2,10,11,12],'day13_development_used':False,'test_used':False,
              'stage1_checkpoint_sha256':sha(cp),'frozen_graph_encoder':True,
              'source_contacts':'day02 counts','targets':'historical day10-12 only'}
    torch.save({'state_dict':model.state_dict(),'protocol':protocol},checkpoint)
    (a.output/'history.json').write_text(json.dumps(history,indent=2))
    final_profile=profile([2,10,11,12])
    with torch.no_grad():source_out=model.source(torch.tensor(base.to_numpy()),final_profile).numpy()
    pd.DataFrame(source_out,index=base.index,columns=base.columns).to_csv(a.output/'source.csv')
    (a.output/'completed.json').write_text(json.dumps({'checkpoint_sha256':sha(checkpoint),
       'source_sha256':sha(a.output/'source.csv'),'protocol':protocol},indent=2))


if __name__=='__main__':main()
