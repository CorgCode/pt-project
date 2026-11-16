"""Two-tower contrastive fingerprints; day-causal training examples.

Own host IDs are used ONLY to group observations and supply positive labels.
Concrete partner IDs are legitimate traffic tokens. No diagonal lookup exists.
"""
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import normalize,StandardScaler
from .collaborative import unit


def sparse_tensor(x):
    x=x.tocoo()
    return torch.sparse_coo_tensor(np.vstack([x.row,x.col]),x.data.astype(np.float32),x.shape).coalesce()


class FingerprintFeatures:
    def __init__(self,peer_width=8192,service_width=2048):
        self.peer_width,self.service_width=peer_width,service_width
        self.peer_hasher=FeatureHasher(n_features=peer_width,alternate_sign=False)
        self.service_hasher=FeatureHasher(n_features=service_width,alternate_sign=False)

    def hashed(self,obs,prefix):
        h=self.peer_hasher if prefix=='p:' else self.service_hasher
        rate=3600/(obs.window.end-obs.window.start)
        return h.transform([{k:np.log1p(v*rate) for k,v in row.items() if k.startswith(prefix)}
                            for row in obs.counts]).tocsr().astype(np.float32)

    def fit(self,observations):
        self.idf={}
        for prefix in ['p:','s:']:
            # DF counts distinct hosts across history, not repeated host-hours.
            seen=sum((self.hashed(o,prefix).astype(bool).astype(np.int32) for o in observations))
            df=np.asarray((seen>0).sum(0)).ravel()
            self.idf[prefix]=(np.log((len(observations[0].counts)+1)/(df+1))+1).astype(np.float32)
        active=np.vstack([o.numeric[o.events>=300] for o in observations])
        self.scaler=StandardScaler().fit(active)
        return self

    def transform(self,obs):
        blocks=[normalize(self.hashed(obs,p).multiply(self.idf[p]),norm='l2') for p in ['p:','s:']]
        numeric=np.clip(self.scaler.transform(obs.numeric),-6,6).astype(np.float32)
        blocks.append(sp.csr_matrix(normalize(numeric)*.15))
        return sp.hstack(blocks,format='csr',dtype=np.float32)

    def service(self,obs):
        return normalize(self.hashed(obs,'s:').multiply(self.idf['s:']),norm='l2').tocsr()


def historical_average(observations,matrices,half_life=3.):
    last=max(o.window.day for o in observations)
    result=sp.csr_matrix(matrices[0].shape,dtype=np.float32);mass=np.zeros(len(observations[0].counts))
    for o,x in zip(observations,matrices):
        w=(o.events>=300)*2.**(-(last-o.window.day)/half_life)
        result+=sp.diags(w)@x;mass+=w
    return (sp.diags(1/np.maximum(mass,1e-12))@result).tocsr().astype(np.float32),mass>0


class Tower(nn.Module):
    def __init__(self,width,dim):
        super().__init__()
        self.first=nn.Parameter(torch.randn(width,128)*.02)
        self.bias=nn.Parameter(torch.zeros(128))
        self.last=nn.Sequential(nn.LayerNorm(128),nn.GELU(),nn.Linear(128,dim,bias=False))

    def forward(self,x):
        return F.normalize(self.last(torch.sparse.mm(x,self.first)+self.bias),dim=1)

    def orthogonal_penalty(self):
        w=F.normalize(self.last[-1].weight,dim=1)
        return (w@w.T-torch.eye(len(w),device=w.device)).square().mean()


def encode(tower,x):
    tower.eval()
    with torch.no_grad():return np.vstack([tower(sparse_tensor(x[i:i+256])).numpy() for i in range(0,x.shape[0],256)])


class ContrastiveFingerprint:
    def __init__(self,dim=32,steps=300,seed=17,temperature=.07,half_life=3.,peer_width=8192,service_width=2048):
        self.dim,self.steps,self.seed,self.temperature,self.half_life=dim,steps,seed,temperature,half_life
        self.features=FingerprintFeatures(peer_width,service_width)

    def fit(self,observations):
        observations=sorted(observations,key=lambda o:o.window.start)
        self.training_end=max(o.window.end for o in observations)
        self.features.fit(observations)
        matrices=[self.features.transform(o) for o in observations]
        service=[self.features.service(o) for o in observations]
        self.query_tower=Tower(matrices[0].shape[1],self.dim)
        self.history_tower=Tower(matrices[0].shape[1],self.dim)
        folds=[]
        for d,obs in enumerate(observations):
            earlier=[i for i,o in enumerate(observations) if o.window.day<obs.window.day and o.window.end<=obs.window.start]
            if not earlier:continue
            history,present=historical_average([observations[i] for i in earlier],[matrices[i] for i in earlier],self.half_life)
            services,_=historical_average([observations[i] for i in earlier],[service[i] for i in earlier],self.half_life)
            labels=np.flatnonzero(present&(obs.events>=300))
            if len(labels)<2:continue
            # Service-hard candidates contain no target-day peer information.
            hard=[]
            for start in range(0,len(labels),256):
                sim=(service[d][labels[start:start+256]]@services[labels].T).toarray()
                for row in range(len(sim)):
                    sim[row,start+row]=-np.inf
                k=min(8,len(labels)-1)
                hard.extend(labels[np.argpartition(-sim,k-1,axis=1)[:,:k]])
            folds.append({'query':matrices[d],'history':history,'labels':labels,'hard':np.array(hard),
                          'query_day':obs.window.day,'history_days':[observations[i].window.day for i in earlier]})
        if not folds:raise ValueError('Identity needs hosts observed on at least two distinct days')
        torch.manual_seed(self.seed);rng=np.random.default_rng(self.seed)
        # Reset construction after seeding, so all initial parameters are reproducible.
        self.query_tower=Tower(matrices[0].shape[1],self.dim)
        self.history_tower=Tower(matrices[0].shape[1],self.dim)
        parameters=[*self.query_tower.parameters(),*self.history_tower.parameters()]
        optimizer=torch.optim.AdamW(parameters,lr=1e-3,weight_decay=1e-4)
        self.log=[]
        for step in range(self.steps):
            fold=folds[int(rng.integers(len(folds)))];labels=fold['labels']
            positions=rng.choice(len(labels),min(96,len(labels)),replace=False);queries=labels[positions]
            candidates=np.unique(np.concatenate([queries,fold['hard'][positions].ravel(),rng.choice(labels,min(128,len(labels)),replace=False)]))
            q=self.query_tower(sparse_tensor(fold['query'][queries]))
            h=self.history_tower(sparse_tensor(fold['history'][candidates]))
            target=torch.from_numpy(np.searchsorted(candidates,queries))
            logits=q@h.T/self.temperature
            ce=F.cross_entropy(logits,target)
            paired=h[target]
            reverse=F.cross_entropy(paired@q.T/self.temperature,torch.arange(len(queries)))
            alignment=(q-paired).square().mean()
            ortho=self.query_tower.orthogonal_penalty()+self.history_tower.orthogonal_penalty()
            loss=.5*(ce+reverse)+.2*alignment+.05*ortho
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite fingerprint loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(parameters,3.);optimizer.step()
            if (step+1)%50==0 or step+1==self.steps:
                row={'step':step+1,'loss':float(loss.detach()),'nce':float(ce.detach()),'orthogonal':float(ortho.detach())}
                self.log.append(row);print('fingerprint',self.seed,row,flush=True)
        bank,_=historical_average(observations,matrices,self.half_life)
        self.bank=unit(encode(self.history_tower,bank))
        self.fold_provenance=[{k:f[k] for k in ['query_day','history_days']} for f in folds]
        return self

    def query(self,observation):
        return unit(encode(self.query_tower,self.features.transform(observation)))

    def assignments(self,observation,k=1,temperature=.02):
        if not 1<=k<=len(self.bank):raise ValueError('invalid retrieval k')
        query=self.query(observation);rr=[];cc=[];vv=[]
        for start in range(0,len(query),256):
            sim=query[start:start+256]@self.bank.T
            top=np.argsort(-sim,axis=1,kind='stable')[:,:k]
            weights=np.take_along_axis(sim,top,1)
            weights=np.exp((weights-weights.max(1,keepdims=True))/temperature);weights/=weights.sum(1,keepdims=True)
            rr.extend(np.repeat(np.arange(start,start+len(top)),k));cc.extend(top.ravel());vv.extend(weights.ravel())
        return sp.csr_matrix((vv,(rr,cc)),shape=(len(query),len(self.bank))),query
