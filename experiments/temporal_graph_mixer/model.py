"""Общий временной Mixer с фиксированной поведенческой частью."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .features import unit


class MixerLayer(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.norm=nn.LayerNorm(width)
        self.time=nn.Sequential(nn.Linear(4,16),nn.GELU(),nn.Linear(16,4))
        self.channel=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,2*width),nn.GELU(),nn.Linear(2*width,width))
    def forward(self,x):
        x=x+self.time(self.norm(x).transpose(1,2)).transpose(1,2)
        return x+self.channel(x)


class TemporalMixer(nn.Module):
    def __init__(self,dim=128,mode='graph',hidden=256):
        super().__init__();self.dim,self.mode,self.hidden=dim,mode,hidden
        self.input=nn.Linear(4*dim,hidden)
        self.layers=nn.Sequential(MixerLayer(hidden),MixerLayer(hidden))
        self.stable=nn.Linear(hidden,256);self.dynamic=nn.Linear(hidden,128)
        self.reconstruct=nn.Sequential(nn.Linear(384,hidden),nn.GELU(),nn.Linear(hidden,4*dim))
        # Причинный прогноз получает один исходный quarter до временного Mixer.
        self.next_patch=nn.Sequential(nn.Linear(384,hidden),nn.GELU(),nn.Linear(hidden,4*dim))

    def inputs(self,x):
        if self.mode=='no_graph':
            x=x.clone();x[:,:,3,:]=0
        return x

    def forward(self,x):
        x=self.inputs(x);h=self.layers(self.input(x.flatten(2)))
        stable=F.normalize(self.stable(h.mean(1)),dim=1)
        dynamic=F.normalize(self.dynamic(h[:,-1]-h.mean(1)),dim=1)
        return stable,dynamic

    def embedding(self,x,weights=(2.,1.,.5)):
        own=F.normalize(x[:,:,0,:].mean(1),dim=1)
        stable,dynamic=self(x)
        return F.normalize(torch.cat([own*weights[0],stable*weights[1],dynamic*weights[2]],1),dim=1)


def vicreg(x):
    x=x-x.mean(0);cov=x.T@x/max(len(x)-1,1)
    variance=F.relu(1-torch.sqrt(x.var(0,unbiased=False)+1e-4)).mean()
    return variance+(cov.square().sum()-cov.diag().square().sum())/x.shape[1]


def pretrain(features,active,mode='graph',steps=600,seed=17,dim=128):
    torch.set_num_threads(2);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=TemporalMixer(dim,mode);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    n=features[0].shape[0];eligible=np.flatnonzero(np.stack(active).sum(0)>=2)
    if len(eligible)<2:raise ValueError('need at least two hosts active in two training windows')
    log=[]
    for step in range(steps):
        # Равномерно выбираются хосты, затем два разных активных дня каждого.
        labels=rng.choice(eligible,min(128,len(eligible)),replace=False)
        pairs=np.array([rng.choice(np.flatnonzero(np.array(active)[:,i]),2,replace=False) for i in labels])
        a=torch.from_numpy(np.stack([features[d][h] for d,h in zip(pairs[:,0],labels)]))
        b=torch.from_numpy(np.stack([features[d][h] for d,h in zip(pairs[:,1],labels)]))
        a=model.inputs(a);b=model.inputs(b)
        # Независимое маскирование блоков; собственного ID во входе нет.
        keep=torch.rand((len(a),4,4,1))>.15;masked=a*keep
        sa,da=model(masked);sb,db=model(b)
        logits=sa@sb.T/.12;truth=torch.arange(len(labels))
        identity=.5*(F.cross_entropy(logits,truth)+F.cross_entropy(logits.T,truth))
        recovered=model.reconstruct(torch.cat([sa,da],1));reconstruct=F.smooth_l1_loss(recovered,a.mean(1).flatten(1))
        # Цель прогноза — фиксированные признаки будущего quarter; fit на train.
        forecast_terms=[]
        for horizon in [1,2,3]:
            patch=model.input(a[:,:-horizon].flatten(2))
            latent=torch.cat([F.normalize(model.stable(patch),dim=2),F.normalize(model.dynamic(patch),dim=2)],2)
            prediction=model.next_patch(latent)
            forecast_terms.append(F.smooth_l1_loss(prediction,a[:,horizon:].flatten(2)))
        forecast=torch.stack(forecast_terms).mean()
        own=F.normalize(a[:,:,0,:].mean(1),dim=1)
        relation=F.mse_loss(sa@sa.T,own@own.T)
        anti=vicreg(torch.cat([sa,da],1)*np.sqrt(384))
        loss=identity+.5*reconstruct+.3*forecast+.5*relation+.02*anti
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite pretraining loss')
        opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
        if (step+1)%50==0 or step+1==steps:
            row={'step':step+1,'loss':float(loss.detach()),'identity':float(identity.detach()),'forecast':float(forecast.detach()),'reconstruct':float(reconstruct.detach())}
            log.append(row);print(mode,'pretrain',row,flush=True)
    model.eval();return model,log


def encode(model,x,weights=(2.,1.,.5)):
    model.eval()
    with torch.no_grad():return np.vstack([model.embedding(torch.from_numpy(x[i:i+256]),weights).numpy() for i in range(0,len(x),256)])


def plain(x,graph=True):
    parts=x.mean(1)
    return unit(np.hstack([unit(parts[:,0])*2,unit(parts[:,1]),unit(parts[:,2]),unit(parts[:,3])*(.5 if graph else 0)]))


def pooled_source(model,history,days,weights=(2.,1.,.5),presence=None):
    """Fixed source profile from allowed history; source behavior always canonical02.

    This is host-wise aggregation of feature-derived codes, not a learned ID table.
    A new host can use the same operation with a singleton observed history.
    """
    cutoff=max(days);mass=np.zeros(len(history[0]));stable=np.zeros((len(mass),256));dynamic=np.zeros((len(mass),128))
    model.eval()
    for values,day in zip(history,days):
        active=(presence[days.index(day)] if presence is not None else np.ones(len(values),bool))
        # Явная маска активности исключает отсутствующие наблюдения из source.
        w=active*2.**(-(cutoff-day)/3.)
        with torch.no_grad():
            for start in range(0,len(values),256):
                s,d=model(torch.from_numpy(values[start:start+256]))
                stable[start:start+256]+=s.numpy()*w[start:start+256,None]
                dynamic[start:start+256]+=d.numpy()*w[start:start+256,None]
        mass+=w
    own=unit(history[0][:,:,0,:].mean(1))
    return unit(np.hstack([own*weights[0],unit(stable)*weights[1],unit(dynamic)*weights[2]]))
