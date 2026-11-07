"""Orthogonal block rotation trained ONLY on historical Links labels, never on development/test."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit_rotation(source,hosts,edges,steps=150,width=64):
    import torch
    from torch.nn import functional as F
    torch.manual_seed(17);torch.set_num_threads(2)
    d=source.shape[1];width=min(width,d)
    # Choose high-variance coordinates from training source; no future edges used to select axes.
    indices=np.argsort(-np.var(source,axis=0))[:width]
    raw=torch.tensor(np.asarray(source),dtype=torch.float32)
    base=torch.tensor(StandardScaler().fit_transform(source),dtype=torch.float32)
    block=raw[:,indices]
    skew=torch.nn.Parameter(torch.zeros(width,width))
    weights=torch.nn.Parameter(torch.zeros(d*2+1))
    optimizer=torch.optim.AdamW([skew,weights],lr=.01,weight_decay=0.)
    index=pd.Index(hosts);a=index.get_indexer(edges.host_a);b=index.get_indexer(edges.host_b)
    y=edges.label.to_numpy(dtype=np.float32)
    if (a<0).any() or (b<0).any():raise ValueError('historical edges outside source cohort')
    rng=np.random.default_rng(17);eye=torch.eye(width);log=[]
    for step in range(steps):
        chosen=rng.choice(len(edges),min(512,len(edges)),replace=False)
        antisymmetric=(skew-skew.T)*.5
        R=torch.linalg.solve(eye-antisymmetric,eye+antisymmetric)
        rotated=block@R;rotated=(rotated-rotated.mean(0))/rotated.std(0,correction=0).clamp_min(1e-8)
        left,right=base[a[chosen]].clone(),base[b[chosen]].clone()
        left[:,indices]=rotated[a[chosen]];right[:,indices]=rotated[b[chosen]]
        logits=torch.abs(left-right)@weights[:d]+(left*right)@weights[d:2*d]+weights[-1]
        loss=F.binary_cross_entropy_with_logits(logits,torch.from_numpy(y[chosen]))
        loss+=1e-4*(weights[:-1]**2).sum()+1e-3*(skew**2).mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([skew,weights],3);optimizer.step()
        if step%50==0:log.append({'step':step,'historical_bce':float(loss.detach())})
    A=(skew-skew.T).detach().numpy()*.5
    R=np.linalg.solve(np.eye(width)-A,np.eye(width)+A)
    if np.max(np.abs(R@R.T-np.eye(width)))>1e-10:raise FloatingPointError('rotation lost orthogonality')
    return {'indices':indices,'matrix':R,'history':log}


def apply_rotation(source,state):
    result=np.asarray(source).copy()
    result[:,state['indices']]=result[:,state['indices']]@state['matrix']
    return result
