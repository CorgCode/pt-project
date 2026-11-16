"""Benchmark-exact top-ten voting from a symmetric similarity, before vectorization."""
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from .data import contacts


def partner_frame(source_rows,target_rows,hosts,population):
    return pd.DataFrame([(hosts[i],p,w,c) for w,rows in [('source',source_rows),('target',target_rows)]
                         for i in population for p,c in rows[i].items()],
                        columns=['host_id','partner','window','count'])


def contact_matrix(rows,vocabulary):
    lookup={v:i for i,v in enumerate(vocabulary)}
    rr,cc,vv=[],[],[]
    for i,row in enumerate(rows):
        for peer,count in row.items():
            if peer in lookup: rr.append(i);cc.append(lookup[peer]);vv.append(count)
    return sp.csr_matrix((vv,(rr,cc)),shape=(len(rows),len(vocabulary)),dtype=float)


def first_rr(scores,positive,valid):
    if not len(positive): return None
    best=np.max(scores[positive]); greater=np.count_nonzero((scores>best)&valid)
    ties=np.count_nonzero((scores==best)&valid)
    return 1./(1+greater+(ties-1)/2)


class WindowTask:
    def __init__(self,source_rows,target,hosts,reference):
        self.name=target.window.name;self.population=np.flatnonzero(target.events>=300)
        self.mask=np.zeros(len(hosts),bool);self.mask[self.population]=True
        rows=[r if self.mask[i] else {} for i,r in enumerate(source_rows)]
        self.vocabulary=sorted(set().union(*(set(r) for r in rows)))
        self.matrix=contact_matrix(rows,self.vocabulary)
        own=pd.Index(self.vocabulary).get_indexer(hosts)
        future=contacts(target);lookup={p:i for i,p in enumerate(self.vocabulary)}
        self.positive=[];self.old=[];self.own=own
        for i in range(len(hosts)):
            self.old.append(self.matrix[i].indices)
            self.positive.append(np.array([lookup[p] for p in future[i]
                if p in lookup and p not in source_rows[i] and p!=hosts[i]],dtype=int))
        self.n_queries=sum(bool(len(self.positive[i])) for i in self.population)
        nearest=pd.read_csv(Path(reference)/'neighbors.csv',dtype={'host_id':str,'neighbor_id':str})
        index=pd.Index(hosts)
        self.truth={index.get_loc(h):set(index.get_indexer(g.sort_values('rank').neighbor_id.iloc[:10]))
                    for h,g in nearest.groupby('host_id',sort=False)}
        pairs=pd.read_csv(Path(reference)/'pairs.csv',dtype={'host_a':str,'host_b':str})
        self.pairs=(index.get_indexer(pairs.host_a),index.get_indexer(pairs.host_b),pairs.label.to_numpy(int))
        import json
        self.ceiling=json.loads(Path(reference,'manifest.json').read_text())['neighbor_ceiling']

    def neighbors(self,S,query,changed=None):
        eligible=self.population[self.population!=query]
        scores=S[query,eligible].copy()
        if changed is not None:
            other,value=changed;positions=np.flatnonzero(eligible==other)
            if len(positions): scores[positions[0]]=value
        k=min(10,len(eligible));top=np.argpartition(-scores,k-1)[:k]
        return eligible[top],np.maximum(scores[top],0)

    def row(self,S,query,changed=None):
        donors,weights=self.neighbors(S,query,changed)
        behavior=len(set(donors)&self.truth[query])/len(donors)
        if not len(self.positive[query]): return 0.,behavior,donors
        # The official CSR W sorts donor columns before accumulating weighted votes.
        order=np.argsort(donors);donors,weights=donors[order],weights[order]
        values=np.asarray(self.matrix[donors].T@weights).ravel()
        valid=np.ones(len(values),bool);valid[self.old[query]]=False
        if self.own[query]>=0: valid[self.own[query]]=False
        return first_rr(values,self.positive[query],valid),behavior,donors

    def partner_chance(self):
        pop=np.asarray(self.matrix.sum(0)).ravel();popular=[];history=[]
        for query in self.population:
            if not len(self.positive[query]):continue
            valid=np.ones(len(pop),bool);valid[self.old[query]]=False
            if self.own[query]>=0:valid[self.own[query]]=False
            popular.append(first_rr(pop,self.positive[query],valid))
            history.append(2/(np.count_nonzero(valid)+1))
        return max(float(np.mean(popular)),float(np.mean(history)))

    def summary(self,S):
        rows=[self.row(S,q)[:2] for q in self.population]
        a,b,y=self.pairs
        return {'new_partners_mrr':sum(r[0] for r in rows)/max(self.n_queries,1),
                'neighbors_at_10':float(np.mean([r[1] for r in rows])),
                'pair_auc':float(roc_auc_score(y,S[a,b])),
                'n_hosts':len(self.population),'n_new_partner_queries':self.n_queries}
