"""Historical donor replacement rewards; no test-window targets accepted here."""
import numpy as np
import torch
from torch.nn import functional as F
from evaluation.partners import _mid_ranks


def reciprocal_rank(scores, valid, positive):
    """Same best-positive midrank convention as evaluation.partners."""
    scores=np.asarray(scores,dtype=float).copy()
    scores[~valid]=-np.inf
    positives=np.flatnonzero(positive & valid)
    if not len(positives):return None
    return float(1./_mid_ranks(scores,positives).min())


def replacement_pairs(vectors,queries,episode,*,k=10,near=10,positive_cap=10):
    """One-swap oracle proposals for a historical episode, not a test input.

    Rewards are evaluated with current positive cosine weights; later parameter
    updates can invalidate them. A tie at the top-k boundary follows numpy's
    argpartition, as in the evaluator.
    """
    z=np.asarray(vectors,dtype=float)
    z=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
    active=np.asarray(episode.active);M=episode.source
    triples=[]
    valid_t,positive_t=episode.masks(queries)
    valid=valid_t.numpy();positive=positive_t.numpy()
    for row,q in enumerate(queries):
        similarities=z[active]@z[q]
        similarities[active==q]=-np.inf
        top=np.argpartition(-similarities,k-1)[:k]
        donors=active[top]
        weights=np.maximum(similarities[top],0)
        scores=np.asarray(M[donors].T@weights).ravel()
        old=reciprocal_rank(scores,valid[row],positive[row])
        if old is None:continue
        ranked=np.argsort(-similarities,kind='stable')
        near_ids=active[ranked[k:k+near]]
        positives=np.flatnonzero(positive[row])
        relevant=np.asarray((M[active][:,positives].getnnz(axis=1)>0)).ravel()
        other=active[np.flatnonzero(relevant)[:positive_cap]]
        candidates=np.unique(np.concatenate((near_ids,other)))
        candidates=candidates[(candidates!=q)&~np.isin(candidates,donors)]
        for a,weight in zip(donors,weights):
            without=scores-weight*M[a].toarray().ravel()
            for b in candidates:
                sim=float(np.dot(z[q],z[b]))
                new=reciprocal_rank(without+max(sim,0)*M[b].toarray().ravel(),valid[row],positive[row])
                if new>old+1e-12:triples.append((int(q),int(a),int(b),new-old))
    return triples


def swap_loss(vectors,triples,temperature=.1):
    if not triples:return vectors.sum()*0
    v=F.normalize(vectors,dim=1)
    idx=torch.tensor([t[:3] for t in triples],dtype=torch.long,device=v.device)
    reward=torch.tensor([t[3] for t in triples],dtype=v.dtype,device=v.device)
    qa=(v[idx[:,0]]*v[idx[:,1]]).sum(1)
    qb=(v[idx[:,0]]*v[idx[:,2]]).sum(1)
    return (reward*F.softplus((qa-qb)/temperature)).mean()
