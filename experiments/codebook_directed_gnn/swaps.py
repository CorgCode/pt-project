"""Награды исторических замен; ответы benchmark не используются."""
import numpy as np
import torch
from torch.nn import functional as F
from evaluation.partners import _mid_ranks


def reciprocal_rank(scores,valid,positive):
    scores=np.asarray(scores,dtype=float).copy();scores[~valid]=-np.inf
    positives=np.flatnonzero(positive&valid)
    return None if not len(positives) else float(1./_mid_ranks(scores,positives).min())


def replacement_pairs(vectors,queries,episode,*,k=10,near=5,positive_cap=5):
    z=np.asarray(vectors,dtype=float);z=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
    active=np.asarray(episode.active);matrix=episode.source;triples=[]
    valid,positive=episode.masks(queries);valid=valid.numpy();positive=positive.numpy()
    for row,q in enumerate(queries):
        sim=z[active]@z[q];sim[active==q]=-np.inf
        top=np.argpartition(-sim,k-1)[:k];donors=active[top];weights=sim[top].clip(min=0)
        scores=np.asarray(matrix[donors].T@weights).ravel();old=reciprocal_rank(scores,valid[row],positive[row])
        if old is None:continue
        ranked=np.argsort(-sim,kind='stable');near_ids=active[ranked[k:k+near]]
        positives=np.flatnonzero(positive[row]);relevant=matrix[active][:,positives].getnnz(axis=1)>0
        other=active[np.flatnonzero(relevant)[:positive_cap]]
        candidates=np.unique(np.r_[near_ids,other]);candidates=candidates[(candidates!=q)&~np.isin(candidates,donors)]
        for a,weight in zip(donors,weights):
            without=scores-weight*matrix[a].toarray().ravel()
            for b in candidates:
                new=reciprocal_rank(without+max(float(z[q]@z[b]),0)*matrix[b].toarray().ravel(),
                                    valid[row],positive[row])
                if new>old+1e-12:triples.append((int(q),int(a),int(b),new-old))
    return triples


def swap_loss(vectors,triples,temperature=.1):
    if not triples:return vectors.sum()*0
    z=F.normalize(vectors,dim=1)
    idx=torch.tensor([t[:3] for t in triples],dtype=torch.long,device=z.device)
    reward=torch.tensor([t[3] for t in triples],dtype=z.dtype,device=z.device)
    a=(z[idx[:,0]]*z[idx[:,1]]).sum(1);b=(z[idx[:,0]]*z[idx[:,2]]).sum(1)
    return (reward*F.softplus((a-b)/temperature)).mean()
