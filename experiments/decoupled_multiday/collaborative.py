"""Implicit ALS on decayed host x (peer,service) counts; exact top-ten loss.

ALS implements Hu/Koren/Volinsky's confidence-weighted implicit objective.
The listwise objective is a smooth first-relevant surrogate, NOT exact MRR.
Development always measures the benchmark's actual voting and midranks.
"""
from collections import Counter
import numpy as np
import scipy.sparse as sp
import torch
from torch.nn import functional as F
from experiments.similarity_graph.metrics import contact_matrix, first_rr
from experiments.similarity_graph.data import contacts


def unit(x):
    x=np.asarray(x,np.float32).copy()
    norms=np.linalg.norm(x,axis=1,keepdims=True)
    x/=np.maximum(norms,1e-12)
    # A deterministic common unknown, never an own-ID code.
    x[norms[:,0]<1e-12,0]=1.
    return x


def interaction_matrix(observations, half_life=3., min_hosts=2):
    if not observations: raise ValueError('CF needs at least one historical target day')
    rows=[Counter() for _ in observations[0].counts]
    last=max(o.window.day for o in observations)
    for obs in observations:
        weight=2.**(-(last-obs.window.day)/half_life)
        rate=3600/(obs.window.end-obs.window.start)
        for row,tokens in zip(rows,obs.counts):
            for token,count in tokens.items():
                if token.startswith('c:'):
                    # Remove only direction: peer+protocol+port remain concrete.
                    row[token[4:]]+=weight*np.log1p(count*rate)
    frequency=Counter(k for row in rows for k in row)
    vocabulary=sorted(k for k,count in frequency.items() if count>=min_hosts)
    if not vocabulary:raise ValueError('empty peer/service interaction vocabulary')
    return contact_matrix(rows,vocabulary).astype(np.float64),vocabulary


def als_half(matrix, fixed, regularization, alpha):
    """Exact normal equations for p=1 at observed entries, c=1+alpha*x."""
    matrix=matrix.tocsr();d=fixed.shape[1]
    gram=fixed.T@fixed+regularization*np.eye(d)
    out=np.zeros((matrix.shape[0],d))
    for row in range(matrix.shape[0]):
        start,end=matrix.indptr[row:row+2]
        if start==end:continue
        y=fixed[matrix.indices[start:end]];extra=alpha*matrix.data[start:end]
        a=gram+(y.T*extra)@y
        b=y.T@(1+extra)
        out[row]=np.linalg.solve(a,b)
    return out


class ImplicitALS:
    def __init__(self,dim=32,regularization=10.,alpha=10.,steps=8,seed=17):
        self.dim,self.regularization,self.alpha,self.steps,self.seed=dim,regularization,alpha,steps,seed

    def fit(self,matrix):
        matrix=matrix.tocsr();rng=np.random.default_rng(self.seed)
        self.items=rng.normal(0,.05,(matrix.shape[1],self.dim))
        for step in range(self.steps):
            self.users=als_half(matrix,self.items,self.regularization,self.alpha)
            self.items=als_half(matrix.T,self.users,self.regularization,self.alpha)
        return self


class PartnerTask:
    def __init__(self,source,target,hosts):
        self.population=target.events>=300
        visible=[r if self.population[i] else {} for i,r in enumerate(source)]
        self.vocabulary=sorted(set().union(*(set(r) for r in visible)))
        self.matrix=contact_matrix(visible,self.vocabulary)
        lookup={p:i for i,p in enumerate(self.vocabulary)}
        self.valid=np.ones(self.matrix.shape,bool);self.positive=[]
        for i,(old,new,host) in enumerate(zip(source,contacts(target),hosts)):
            for peer in old:
                if peer in lookup:self.valid[i,lookup[peer]]=False
            if host in lookup:self.valid[i,lookup[host]]=False
            self.positive.append(np.array([lookup[p] for p in new if p in lookup and self.valid[i,lookup[p]]],int))
        self.eligible=np.array([i for i in np.flatnonzero(self.population) if len(self.positive[i])],int)
        coo=self.matrix.T.tocoo()
        self.tensor=torch.sparse_coo_tensor(np.vstack([coo.row,coo.col]),coo.data.astype(np.float32),coo.shape).coalesce()


def listwise_loss(base,identity,graph,partner,queries,task,weights=(1.,.15,.5,.1)):
    """Straight-through top-10 forward; smooth gradient; actual source counts.

    The forward vote has EXACTLY ten donors (or fewer in tiny cohorts), clipped
    nonnegative cosine weights, old/own exclusions and all admissible candidates.
    Backward uses a dense softmax surrogate for changes in membership.
    """
    blocks=[base,identity,partner,graph]
    z=F.normalize(torch.cat([b*w for b,w in zip(blocks,weights)],1),dim=1)
    q=torch.as_tensor(queries);sim=z[q]@z.T
    allowed=torch.from_numpy(task.population)
    sim=sim.masked_fill(~allowed[None,:],-1e9)
    sim=sim.scatter(1,q[:,None],-1e9)
    k=min(10,int(task.population.sum())-1)
    if k<1:raise ValueError('fewer than two active hosts')
    values,indices=sim.topk(k,1)
    hard=torch.zeros_like(sim).scatter(1,indices,values.clamp_min(0))
    soft=F.softmax(sim/.07,1)
    # Common row scale cancels from candidate ranks, retain hard weight magnitude.
    smooth=soft*hard.detach().sum(1,keepdim=True)
    vote_weights=hard+smooth-smooth.detach()
    votes=torch.sparse.mm(task.tensor,vote_weights.T).T
    valid=torch.from_numpy(task.valid[queries])
    positive=np.zeros(valid.shape,bool)
    for row,query in enumerate(queries):positive[row,task.positive[query]]=True
    positive=torch.from_numpy(positive)
    # Maximizing probability of ANY relevant first item is a listwise surrogate.
    # log1p keeps zero-vote positives trainable, unlike log(clamp(0)).
    logits=torch.log1p(votes.clamp_min(0))/.2
    return (torch.logsumexp(logits.masked_fill(~valid,-torch.inf),1)-
            torch.logsumexp(logits.masked_fill(~positive,-torch.inf),1)).mean(),votes


def refine_factors(initial,base,identity,graph,tasks,steps=200,seed=17,on_step=None):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    codes=torch.nn.Parameter(torch.from_numpy(unit(initial)))
    tensors=[torch.from_numpy(unit(x)) for x in [base,identity,graph]]
    teacher=torch.from_numpy(unit(initial))
    optimizer=torch.optim.AdamW([codes],lr=1e-3,weight_decay=0)
    usable=[t for t in tasks if len(t.eligible)]
    if not usable:raise ValueError('no historical new-partner targets')
    log=[]
    for step in range(steps):
        task=usable[int(rng.integers(len(usable)))];q=rng.choice(task.eligible,min(64,len(task.eligible)),replace=False)
        z=F.normalize(codes,dim=1)
        loss,_=listwise_loss(*tensors[:2],tensors[2],z,q,task)
        penalty=.2*(1-(z*teacher).sum(1)).mean()
        total=loss+penalty
        if not torch.isfinite(total):raise FloatingPointError('nonfinite listwise loss')
        optimizer.zero_grad(set_to_none=True);total.backward()
        torch.nn.utils.clip_grad_norm_([codes],3.);optimizer.step()
        if on_step is not None:on_step(step+1,unit(codes.detach().numpy()))
        if (step+1)%50==0 or step+1==steps:
            row={'step':step+1,'listwise':float(loss.detach()),'anchor':float(penalty.detach())}
            log.append(row);print('partner',row,flush=True)
    return unit(codes.detach().numpy()),log


def direct_validation(model,vocabulary,target,source,hosts):
    """Direct CF forecast on a later day; collapse services to actual peers."""
    peers=sorted({v.rsplit('|',1)[0] for v in vocabulary});lookup={p:i for i,p in enumerate(peers)}
    map_to=sp.csr_matrix((np.ones(len(vocabulary)),(np.arange(len(vocabulary)),
        [lookup[v.rsplit('|',1)[0]] for v in vocabulary])),shape=(len(vocabulary),len(peers)))
    answers=contacts(target);rr=[]
    for start in range(0,len(hosts),128):
        scores=np.asarray((model.users[start:start+128]@model.items.T)@map_to)
        for offset,row in enumerate(scores):
            i=start+offset
            if target.events[i]<300:continue
            valid=np.array([p not in source[i] and p!=hosts[i] for p in peers])
            positive=np.array([lookup[p] for p in answers[i] if p in lookup and valid[lookup[p]]],int)
            if len(positive):rr.append(first_rr(row,positive,valid))
    return {'mrr':float(np.mean(rr)) if rr else None,'n_queries':len(rr),
            'scope':'direct CF; its own learned candidate universe, not benchmark MRR'}
