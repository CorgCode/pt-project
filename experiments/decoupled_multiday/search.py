"""Uncapped raw-metric development search. Full official evaluation follows shortlist."""
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from experiments.similarity_graph.metrics import WindowTask,first_rr
from .collaborative import unit


class RawWindow:
    def __init__(self,source,target,hosts,reference):
        self.task=WindowTask(source,target,hosts,reference)
        self.population=self.task.population

    def evaluate(self,similarity):
        task=self.task;population=self.population
        if similarity.shape!=(len(population),len(population)):raise ValueError('wrong active similarity shape')
        s=similarity.copy();np.fill_diagonal(s,-np.inf)
        k=min(10,len(population)-1);top=np.argpartition(-s,k-1,axis=1)[:,:k]
        weights=np.take_along_axis(s,top,1).clip(min=0)
        global_top=population[top]
        behavior=np.mean([len(set(row)&task.truth[q])/k for row,q in zip(global_top,population)])
        rrs=[]
        for start in range(0,len(population),256):
            donors=global_top[start:start+256];w=weights[start:start+256]
            W=sp.csr_matrix((w.ravel(),(np.repeat(np.arange(len(w)),k),donors.ravel())),shape=(len(w),len(task.mask)))
            votes=(W@task.matrix).toarray()
            for offset,query in enumerate(population[start:start+256]):
                if not len(task.positive[query]):continue
                valid=np.ones(votes.shape[1],bool);valid[task.old[query]]=False
                if task.own[query]>=0:valid[task.own[query]]=False
                rrs.append(first_rr(votes[offset],task.positive[query],valid))
        inverse=np.full(len(task.mask),-1,int);inverse[population]=np.arange(len(population))
        a,b,y=task.pairs
        if min(inverse[a].min(),inverse[b].min())<0:raise ValueError('pairs outside active population')
        return {'consistency@10_mean':float(behavior),'partner_new_mrr':float(np.mean(rrs)),
                'pair_roc_auc':float(roc_auc_score(y,similarity[inverse[a],inverse[b]]))}


def block_similarity(blocks,weights,population):
    """Cosine of concatenation; supports non-unit frozen bases and empty blocks."""
    grams=[];norms=[]
    for block,weight in zip(blocks,weights):
        x=np.asarray(block[population],np.float64)
        grams.append((x@x.T)*weight**2);norms.append((x*x).sum(1)*weight**2)
    norm=np.sqrt(np.maximum(sum(norms),1e-24))
    return sum(grams)/norm[:,None]/norm[None,:]


def cached_block_grams(blocks,population):
    values=[np.asarray(x[population],np.float64) for x in blocks]
    return [x@x.T for x in values],[(x*x).sum(1) for x in values]


def from_grams(grams,norms,weights):
    norm=np.sqrt(np.maximum(sum(n*w*w for n,w in zip(norms,weights)),1e-24))
    return sum(g*w*w for g,w in zip(grams,weights) if w)/norm[:,None]/norm[None,:]


def raw_utility(results):
    """Geometric mean of RAW metric values, with no score caps/chance floors.

    All candidates use the same metric set and windows. This differs deliberately
    from scoreboard's task weighting and saturated Behavior normalization.
    """
    values=[v for result in results for v in result.values()]
    if not values or any(not np.isfinite(v) or v<0 for v in values):raise ValueError('invalid raw metrics')
    return float(np.exp(np.mean(np.log(np.maximum(values,1e-12)))))
