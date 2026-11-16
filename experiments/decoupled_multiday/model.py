"""Portable frozen-base contract and inference-only assembled checkpoints."""
from pathlib import Path
import numpy as np
import pandas as pd
from .collaborative import unit


def frame(values,hosts):
    return pd.DataFrame(values,index=pd.Index(hosts,name='host_id')).add_prefix('emb_')


def assemble(base,identity,partner,graph,weights):
    blocks=[np.asarray(base),unit(identity),unit(partner),unit(graph)]
    if blocks[0].shape[1]!=896:raise ValueError('frozen base must have exactly 896 coordinates')
    if any(len(b)!=len(blocks[0]) for b in blocks):raise ValueError('block populations differ')
    result=np.concatenate([b*float(w) for b,w in zip(blocks,weights) if w!=0],axis=1)
    if result.shape[1]>1024:raise ValueError('assembled dimension exceeds 1024')
    if not np.isfinite(result).all() or not np.all(np.linalg.norm(result,axis=1)>0):raise ValueError('invalid embedding')
    return result


class FrozenCodebook:
    """CSV contract also accepts the real leader without its Python model classes.

    Optional traffic model is used only for unchanged Ridge/plus controls. A learned
    fingerprint checkpoint can infer arbitrary late traffic without that model.
    """
    def __init__(self,hosts,codes,training_end,traffic_model=None,provenance='external codebook'):
        # Preserve externally supplied coordinates, including float64 precision.
        self.hosts=list(hosts);self.codes=np.asarray(codes).copy()
        self.training_end=training_end;self.traffic_model=traffic_model;self.provenance=provenance
        if self.codes.shape!=(len(hosts),896):raise ValueError('expected known-host codebook N x 896')
        if len(set(hosts))!=len(hosts) or not np.isfinite(self.codes).all():raise ValueError('invalid base')

    def late(self,obs):
        if self.traffic_model is not None:
            m=self.traffic_model;t=m.traffic(obs)
            return unit(.5*t+.5*unit(m.ridge.predict(t)))
        if not obs.window.baseline:raise ValueError('external frozen base requires a late baseline CSV for Ridge control')
        data=pd.read_csv(obs.window.baseline,index_col=0)
        if set(data.index)!=set(self.hosts) or data.shape[1]!=896:raise ValueError('late base differs from frozen cohort/dimension')
        return data.loc[self.hosts].to_numpy(np.float32)


class DecoupledModel:
    def __init__(self,base,fingerprint,partner,graph,weights,k=1,temperature=.02,late_mode='fingerprint'):
        self.base,self.fingerprint=base,fingerprint
        self.partner,self.graph=unit(partner),unit(graph)
        self.weights=tuple(weights);self.k,self.temperature,self.late_mode=k,temperature,late_mode
        self.hosts=base.hosts;self.training_end=max(base.training_end,fingerprint.training_end)
        self.source=assemble(base.codes,fingerprint.bank,self.partner,self.graph,self.weights)

    def predict(self,observation):
        p,query=self.fingerprint.assignments(observation,self.k,self.temperature)
        if self.late_mode=='ridge':late_base=self.base.late(observation)
        else:late_base=unit(p@self.base.codes)
        return assemble(late_base,query,p@self.partner,p@self.graph,self.weights)

    def set_cache(self,cache):
        m=self.base.traffic_model
        if m is not None:
            m.features.cache=str(Path(cache))
