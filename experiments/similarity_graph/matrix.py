"""Symmetric coordinate search: BOTH endpoints, exact top-ten votes and pair AUC deltas."""
from bisect import bisect_left,bisect_right,insort
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from scipy import sparse
from sklearn.preprocessing import normalize


class PairOrder:
    def __init__(self,S,pairs):
        a,b,labels=pairs;self.counts={}
        for i,j,y in zip(a,b,labels):
            key=(min(i,j),max(i,j));counts=self.counts.setdefault(key,[0,0]);counts[y]+=1
        scores=S[a,b]
        self.negative=sorted(scores[labels==0].tolist())
        self.positive=sorted(scores[labels==1].tolist())

    @staticmethod
    def smaller(values,x):return bisect_left(values,x)+.5*(bisect_right(values,x)-bisect_left(values,x))

    def delta(self,i,j,old,new):
        negative,positive=self.counts.get((min(i,j),max(i,j)),[0,0])
        change=positive*(self.smaller(self.negative,new)-self.smaller(self.negative,old))
        change+=negative*(self.smaller(self.positive,old)-self.smaller(self.positive,new))
        return change/max(len(self.negative)*len(self.positive),1)

    def commit(self,i,j,old,new):
        counts=self.counts.get((min(i,j),max(i,j)),[0,0])
        for values,count in zip((self.negative,self.positive),counts):
            for _ in range(count): values.pop(bisect_left(values,old));insort(values,new)


def initialize(behavior,structural,behavior_weight=.8,seed=17):
    a,b=normalize(behavior),normalize(structural)
    second=b@b.T
    if sparse.issparse(second):second=second.toarray()
    S=np.asarray(behavior_weight*(a@a.T)+(1-behavior_weight)*second,dtype=float)
    rng=np.random.default_rng(seed)
    # Symmetric tiny noise breaks degenerate top-k ties before learning, never after factorization.
    noise=rng.uniform(-1e-5,1e-5,S.shape);S+=(noise+noise.T)*.5
    np.fill_diagonal(S,0.)
    return S


def score_proxy(values,task,chance):
    rr,beh,pair=values
    normalized=[(rr-chance)/max(1-chance,1e-12),
                (beh-10/(len(task.population)-1))/max(task.ceiling-10/(len(task.population)-1),1e-12),
                (pair-.5)/.5]
    rr,beh,pair=np.clip(normalized,.01,1.)
    # Eight times log scoreboard with Links/Identity held fixed: MRR weight1, B/Pairs weight2.
    return float(np.log(rr)+2*np.log(beh)+2*np.log(pair))


def optimize(initial,tasks,passes=2,queries_per_pass=96,proposals=8,behavior_weight=2.,seed=17,
             objective='weighted',min_behavior_retention=.95):
    import hashlib
    if objective not in ['weighted','score_proxy']:raise ValueError('unknown matrix objective')
    S=np.asarray(initial,float).copy();rng=np.random.default_rng(seed)
    orders=[PairOrder(S,t.pairs) for t in tasks]
    source_columns={id(t):t.matrix.tocsc() for t in tasks}
    stats=[];chances=[]
    if objective=='score_proxy':
        for t in tasks:
            s=t.summary(S);stats.append([s['new_partners_mrr'],s['neighbors_at_10'],s['pair_auc']])
            chances.append(t.partner_chance())
    stats=np.asarray(stats);initial_behavior=stats[:,1].copy() if len(stats) else None
    history=[]
    for iteration in range(passes):
        accepted=0;total_gain=0.
        for task in tasks:
            choices=rng.permutation(task.population)[:queries_per_pass]
            for position,query in enumerate(choices):
                if position and position%256==0:
                    print('matrix progress',iteration+1,task.name,position,len(choices),'accepted',accepted,flush=True)
                donors,_=task.neighbors(S,query)
                options=set(task.truth[query])|set(task.population[np.argsort(-S[query,task.population])[:20]])
                # Include supporters of ANY historical correct answer, not only the first one.
                csc=source_columns[id(task)]
                for peer in task.positive[query][:8]:
                    support=csc.indices[csc.indptr[peer]:csc.indptr[peer+1]]
                    options.update(support[np.argsort(-S[query,support])[:4]].tolist())
                options.discard(int(query))
                options=sorted(options,key=lambda j:-S[query,j])
                if len(options)>proposals:
                    # Half strongest candidates, half sampled from the remaining supporters/behavior neighbors.
                    keep=proposals//2
                    options=options[:keep]+rng.choice(options[keep:],proposals-keep,replace=False).tolist()
                for other in options:
                    old=float(S[query,other]);baseline={}
                    for k,t in enumerate(tasks):
                        for q,p in ((query,other),(other,query)):
                            if t.mask[q] and t.mask[p]: baseline[k,q]=t.row(S,q)[:2]
                    levels={old*.8,old*1.05}
                    for t in tasks:
                        if t.mask[query] and t.mask[other]:
                            d,w=t.neighbors(S,query)
                            cutoff=float(np.min(S[query,d]));maximum=float(np.max(S[query,d]))
                            levels.update((max(cutoff+1e-4,.001),min(maximum+1e-4,1.05)))
                    code=f'{seed}|{min(query,other)}|{max(query,other)}'.encode()
                    jitter=((int.from_bytes(hashlib.blake2b(code,digest_size=8).digest(),'little')>>11)/2**53-.5)*1e-5
                    levels={value+jitter for value in levels}
                    best=(0.,old,None)
                    for value in sorted(levels):
                        if abs(value-old)<1e-10:continue
                        gain=0.;deltas=np.zeros((len(tasks),3))
                        for (k,q),(rr,beh) in baseline.items():
                            t=tasks[k];p=other if q==query else query
                            new_rr,new_beh,_=t.row(S,q,(p,value))
                            gain+=(new_rr-rr)/max(t.n_queries,1)
                            gain+=behavior_weight*(new_beh-beh)/(len(t.population)*max(t.ceiling,1e-6))
                            deltas[k,0]+=(new_rr-rr)/max(t.n_queries,1)
                            deltas[k,1]+=(new_beh-beh)/len(t.population)
                        for k,order in enumerate(orders):deltas[k,2]=order.delta(query,other,old,value)
                        gain+=2*deltas[:,2].sum()
                        if objective=='score_proxy':
                            proposed=stats+deltas
                            if np.any(proposed[:,1]<min_behavior_retention*initial_behavior-1e-12):continue
                            gain=sum(score_proxy(proposed[k],t,chances[k])-score_proxy(stats[k],t,chances[k]) for k,t in enumerate(tasks))
                        if gain>best[0]+1e-12: best=(gain,float(value),deltas)
                    if best[0]>1e-12:
                        for order in orders:order.commit(query,other,old,best[1])
                        S[query,other]=S[other,query]=best[1]
                        if objective=='score_proxy':stats+=best[2]
                        accepted+=1;total_gain+=best[0]
        row={'pass':iteration+1,'accepted_symmetric_updates':accepted,'objective_gain':total_gain,
             'objective':objective,
             'training_windows':{t.name:t.summary(S) for t in tasks}}
        history.append(row);print('matrix training',row,flush=True)
    return S,history


def exact_vectors(S,epsilon=None):
    S=np.asarray(S,dtype=float)
    if not np.allclose(S,S.T,atol=1e-12) or np.any(np.diag(S)!=0):
        raise ValueError('S must be symmetric with zero diagonal')
    bound=float(np.max(np.abs(S).sum(1)))
    if epsilon is None:epsilon=.9/max(bound,1.)
    if epsilon<=0 or epsilon*bound>=1:raise ValueError('epsilon must obey the certified spectral-norm upper bound')
    K=S*epsilon;np.fill_diagonal(K,1.)
    Z=np.linalg.cholesky(K)
    error=0.
    for start in range(0,len(Z),128):
        error=max(error,float(np.max(np.abs(Z[start:start+128]@Z.T-K[start:start+128]))))
    if error>1e-10:raise FloatingPointError('factorization failed the Gram preservation check')
    return Z,{'epsilon':epsilon,'spectral_norm_upper_bound':bound,
              'certified_min_eigenvalue':1-epsilon*bound,'max_gram_error':error,
              'max_row_norm_error':float(np.max(np.abs((Z**2).sum(1)-1)))}


def psd_edge_vectors(S):
    """Less conservative exact control. The spectral-norm bound is sufficient, not necessary.

    For a symmetric S, only 1+epsilon*lambda_min(S)>0 is necessary for positive definiteness.
    An explicit eigenvalue calculation and successful checked Cholesky certify this control.
    """
    S=np.asarray(S,float)
    if not np.allclose(S,S.T,atol=1e-12) or np.any(np.diag(S)!=0):raise ValueError('invalid S')
    minimum=float(eigh(S,subset_by_index=[0,0],eigvals_only=True,check_finite=True)[0])
    epsilon=.999/(-minimum) if minimum<0 else 1.
    K=epsilon*S;np.fill_diagonal(K,1.)
    Z=np.linalg.cholesky(K)
    error=max(float(np.max(np.abs(Z[i:i+128]@Z.T-K[i:i+128]))) for i in range(0,len(Z),128))
    if error>1e-10:raise FloatingPointError('PSD-edge Gram factorization failed')
    return Z,{'epsilon':epsilon,'epsilon_rule':'explicit minimum eigenvalue, not norm bound',
              'minimum_eigenvalue_S':minimum,'minimum_eigenvalue_K':1+epsilon*minimum,
              'max_gram_error':error,'max_row_norm_error':float(np.max(np.abs((Z**2).sum(1)-1)))}


def compact_vectors(S,dim,steps=80,seed=17,loss_kind='soft'):
    import torch
    from torch.nn import functional as F
    torch.manual_seed(seed);torch.set_num_threads(2)
    rank=min(dim,len(S)-2)
    values,vectors=eigsh(S,k=rank,which='LA',v0=np.random.default_rng(seed).standard_normal(len(S)))
    codes=torch.nn.Parameter(torch.from_numpy(normalize(vectors*np.sqrt(np.maximum(values,1e-6)))).float())
    optimizer=torch.optim.AdamW([codes],lr=.003,weight_decay=0.)
    rng=np.random.default_rng(seed);history=[]
    for step in range(steps):
        queries=rng.choice(len(S),min(64,len(S)),replace=False)
        q=torch.from_numpy(queries);z=F.normalize(codes,dim=1)
        teacher=torch.from_numpy(S[queries]).float();student=z[q]@z.T
        teacher[torch.arange(len(q)),q]=-1e6;student[torch.arange(len(q)),q]=-1e6
        soft=F.kl_div(F.log_softmax(student/.05,1),F.softmax(teacher/.05,1),reduction='batchmean')
        if loss_kind=='top10':
            top_values,top=teacher.topk(min(10,len(S)-1),dim=1)
            target=torch.zeros_like(teacher).scatter(1,top,top_values.clamp_min(0))
            target/=target.sum(1,keepdim=True).clamp_min(1e-12)
            loss=-(target*F.log_softmax(student/.05,1)).sum(1).mean()+.25*soft
        elif loss_kind=='soft':loss=soft
        else:raise ValueError('unknown compact loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if step%20==0:history.append({'step':step,'neighbor_kl':float(loss.detach())})
    return F.normalize(codes,dim=1).detach().numpy().astype(float),history
