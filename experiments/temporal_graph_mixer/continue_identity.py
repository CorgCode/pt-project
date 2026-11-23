"""Обучение с полной галереей классов; proxy не используется при inference."""
import argparse,json,shutil
from pathlib import Path
import joblib,numpy as np,torch
from torch.nn import functional as F
from .data import sha,prepare
from .evaluate import checkpoint,frame
from .model import encode,pooled_source
from .fit import code_shas,verify_training


def train_full_gallery(model,features,active,steps=600,seed=17):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);n=len(features[0])
    # Proxy нужен только для supervision и не включается в embedding.
    proxy=torch.nn.Parameter(torch.randn(n,256)*.02)
    params=[*model.parameters(),proxy];opt=torch.optim.AdamW(params,lr=5e-4,weight_decay=1e-4)
    present=np.stack(active);eligible=np.flatnonzero(present.sum(0)>=2);log=[]
    for step in range(steps):
        ids=rng.choice(eligible,min(128,len(eligible)),replace=False)
        pairs=np.array([rng.choice(np.flatnonzero(present[:,i]),2,replace=False) for i in ids])
        a=torch.from_numpy(np.stack([features[d][i] for d,i in zip(pairs[:,0],ids)]));b=torch.from_numpy(np.stack([features[d][i] for d,i in zip(pairs[:,1],ids)]))
        a=model.inputs(a);b=model.inputs(b);s,d=model(a);other,_=model(b)
        labels=torch.tensor(ids);logits=torch.cat([s,other])@F.normalize(proxy,dim=1).T/.1
        classification=F.cross_entropy(logits,torch.cat([labels,labels]))
        target=torch.arange(len(ids));contrast=.5*(F.cross_entropy(s@other.T/.12,target)+F.cross_entropy(other@s.T/.12,target))
        reconstruction=F.smooth_l1_loss(model.reconstruct(torch.cat([s,d],1)),a.mean(1).flatten(1))*model.dim
        terms=[]
        for h in [1,2,3]:
            patch=model.input(a[:,:-h].flatten(2));latent=torch.cat([F.normalize(model.stable(patch),dim=2),F.normalize(model.dynamic(patch),dim=2)],2)
            terms.append(F.smooth_l1_loss(model.next_patch(latent),a[:,h:].flatten(2))*model.dim)
        forecast=torch.stack(terms).mean()
        own=F.normalize(a[:,:,0,:].mean(1),dim=1);relation=F.mse_loss(s@s.T,own@own.T)
        loss=classification+.25*contrast+.5*reconstruction+.3*forecast+.5*relation
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite full-gallery loss')
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,3);opt.step()
        if (step+1)%50==0:
            row={'step':step+1,'classification':float(classification.detach()),'contrast':float(contrast.detach()),'reconstruction':float(reconstruction.detach()),'forecast':float(forecast.detach())}
            log.append(row);print('global-gallery',model.mode,row,flush=True)
    model.eval();return model,log


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trained',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--steps',type=int,default=600);a=p.parse_args();torch.set_num_threads(2)
    f=verify_training(a.trained);c=f['config'];hosts=f['hosts']
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    for p in a.trained.iterdir():
        if p.is_file():shutil.copyfile(p,a.output/p.name)
    x=[np.load(a.trained/f"{w['name']}.npy") for w in c['train']]
    active=[joblib.load(prepare(w,hosts,c['feature_dir']))['events']>=300 for w in c['train']]
    for mode in ['no_graph','graph']:
        model=checkpoint(a.trained/f'{mode}.pt');model,history=train_full_gallery(model,x,active,a.steps,c.get('seed',17));name=mode+'_global'
        torch.save({'state':model.state_dict(),'mode':model.mode,'dim':model.dim,'hidden':model.hidden},a.output/f'{name}.pt')
        (a.output/f'{name}_history.json').write_text(json.dumps(history,indent=2)+'\n')
        for strategy in ['single','recent']:
            label=name if strategy=='single' else name+'_recent'
            if label!=name:shutil.copyfile(a.output/f'{name}.pt',a.output/f'{label}.pt')
            values=encode(model,x[0]) if strategy=='single' else pooled_source(model,x,[w['day'] for w in c['train']],presence=active)
            frame(values,hosts).to_csv(a.output/f'{label}_source.csv',float_format='%.17g')
            f['candidates'][label]={'mode':mode,'source_strategy':'history_pool' if strategy=='recent' else 'source02','dim':512,'source_sha256':sha(a.output/f'{label}_source.csv'),'checkpoint_sha256':sha(a.output/f'{label}.pt')}
    f['global_gallery_steps']=a.steps;f['code_sha256']=code_shas();f['parent_training_sha256']=sha(a.trained/'frozen_training.json')
    (a.output/'frozen_training.json').write_text(json.dumps(f,indent=2)+'\n')


if __name__=='__main__':main()
