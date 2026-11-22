"""Цель исторического голосования на выходах общего encoder."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.nn import functional as F
from scipy import sparse as sp
from .model import encode


class VotingTask:
    def __init__(self,source_rows,target,hosts):
        self.population=target['events']>=300
        self.vocabulary=sorted({p for i,row in enumerate(source_rows) if self.population[i] for p in row})
        lookup={p:i for i,p in enumerate(self.vocabulary)};rr=[];cc=[];vv=[]
        self.old=[];self.positives=[];self.own=[]
        for i,(old,new) in enumerate(zip(source_rows,target['contacts'])):
            self.old.append(np.array([lookup[p] for p in old if p in lookup],int))
            self.positives.append(np.array([lookup[p] for p in new if p in lookup and p not in old and p!=hosts[i]],int))
            self.own.append(lookup.get(hosts[i],-1))
            if self.population[i]:
                for p,v in old.items():
                    if p in lookup:rr.append(i);cc.append(lookup[p]);vv.append(v)
        self.matrix=sp.csr_matrix((vv,(rr,cc)),shape=(len(hosts),len(lookup)))
        coo=self.matrix.T.tocoo()
        self.tensor=torch.sparse_coo_tensor(np.vstack([coo.row,coo.col]),coo.data.astype(np.float32),coo.shape).coalesce()
        self.eligible=np.array([q for q in np.flatnonzero(self.population) if len(self.positives[q])],int)

    def loss(self,z,queries):
        q=torch.tensor(queries);s=z[q]@z.T
        allowed=torch.tensor(self.population);s=s.masked_fill(~allowed[None,:],-1e9).scatter(1,q[:,None],-1e9)
        k=min(10,int(self.population.sum())-1);val,ix=s.topk(k,dim=1)
        hard=torch.zeros_like(s).scatter(1,ix,val.clamp_min(0))
        soft=F.softmax(s/.08,1)*hard.detach().sum(1,keepdim=True)
        weights=hard+soft-soft.detach()
        votes=torch.sparse.mm(self.tensor,weights.T).T
        valid=np.ones(votes.shape,bool);positive=np.zeros(votes.shape,bool)
        for row,query in enumerate(queries):
            valid[row,self.old[query]]=False
            if self.own[query]>=0:valid[row,self.own[query]]=False
            positive[row,self.positives[query]]=True
        logits=torch.log1p(votes.clamp_min(0))/.2
        valid=torch.tensor(valid);positive=torch.tensor(positive)
        loss=(torch.logsumexp(logits.masked_fill(~valid,-torch.inf),1)-torch.logsumexp(logits.masked_fill(~positive,-torch.inf),1)).mean()
        return loss,votes


def tune(model,x,observations,steps=200,seed=17):
    """All tasks are historical train; source features always canonical hour02."""
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    source=torch.from_numpy(x[0]);teacher=torch.from_numpy(encode(model,x[0]))
    tasks=[VotingTask(observations[0]['contacts'],o,o['hosts']) for o in observations[1:]]
    tasks=[t for t in tasks if len(t.eligible)]
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4);log=[]
    model.train()
    for step in range(steps):
        task=tasks[int(rng.integers(len(tasks)))];queries=rng.choice(task.eligible,min(32,len(task.eligible)),replace=False)
        # Доноры — вся активная когорта; source строит тот же encoder.
        z=model.embedding(source);vote,_=task.loss(z,queries)
        q=torch.tensor(queries);s=z[q]@z.T;old=teacher[q]@teacher.T
        preserve=F.kl_div(F.log_softmax(s/.05,1),F.softmax(old/.05,1),reduction='batchmean')
        d=int(rng.integers(1,len(x)));present=np.flatnonzero((observations[0]['events']>=300)&(observations[d]['events']>=300))
        ids=rng.choice(present,min(128,len(present)),replace=False);a,_=model(source[ids]);b,_=model(torch.from_numpy(x[d][ids]))
        target=torch.arange(len(ids));identity=.5*(F.cross_entropy(a@b.T/.12,target)+F.cross_entropy(b@a.T/.12,target))
        # Меток holdout нет; сохранение геометрии задаётся штрафом, не гарантией.
        loss=.1*vote+3*preserve+.5*identity
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite fine tuning')
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if (step+1)%20==0 or step+1==steps:
            row={'step':step+1,'vote_surrogate':float(vote.detach()),'geometry_kl':float(preserve.detach()),'identity':float(identity.detach())}
            log.append(row);print('finetune',row,flush=True)
    model.eval();return model,log
