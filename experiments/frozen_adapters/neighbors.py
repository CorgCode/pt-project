"""Штатное голосование, причинный прогноз контактов и teacher полезных замен top-10."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import LogisticRegression
from evaluation.uncertainty import bootstrap_ci

from .data import contacts


def contact_matrix(rows, vocabulary):
    index={v:i for i,v in enumerate(vocabulary)}
    rr,cc,vv=[],[],[]
    for i,row in enumerate(rows):
        for peer,count in row.items():
            if peer in index:
                rr.append(i);cc.append(index[peer]);vv.append(count)
    return sp.csr_matrix((vv,(rr,cc)),shape=(len(rows),len(vocabulary)),dtype=float)


def first_rr(scores, positive, valid):
    positive=np.asarray(positive,dtype=int)
    positive=positive[valid[positive]]
    if not len(positive): return None
    best=float(np.max(scores[positive]))
    greater=int(((scores > best)&valid).sum())
    ties=int(((scores == best)&valid).sum())
    return 1.0/(1+greater+(ties-1)/2)


def selected_neighbors(source, query, population, k=10):
    population=np.asarray(population,dtype=int)
    eligible=population[population!=query]
    if not len(eligible): raise ValueError('нет соседей')
    sim=normalize(source[[query]]) @ normalize(source[eligible]).T
    sim=sim.ravel()
    n=min(k,len(eligible))
    top=np.argpartition(-sim,n-1)[:n]
    return eligible[top], np.maximum(sim[top],0)


def votes(matrix, donors, weights):
    return np.asarray(matrix[donors].T @ np.asarray(weights)).ravel()


class ContactHistory:
    def __init__(self, observations):
        self.end=max(o.window.end for o in observations)
        n=len(observations[0].counts)
        self.hosts=[{} for _ in range(n)]
        self.popularity=Counter()
        self.host_frequency=Counter()
        for obs in observations:
            hour=int(obs.window.start%86400//3600)
            for host,row in enumerate(contacts(obs)):
                for peer,count in row.items():
                    state=self.hosts[host].setdefault(peer,{'count':0,'days':set(),'last':0,'hours':set()})
                    state['count']+=count;state['days'].add(obs.window.day)
                    state['last']=max(state['last'],obs.window.end);state['hours'].add(hour)
                    self.popularity[peer]+=count
        for row in self.hosts:
            self.host_frequency.update(row.keys())

    def features(self, host, candidates, target):
        if self.end > target.start: raise ValueError('история пересекает прогнозируемое окно')
        row=self.hosts[host]
        hour=int(target.start%86400//3600)
        features=np.zeros((len(candidates),8))
        for i,peer in enumerate(candidates):
            state=row.get(peer)
            if state:
                features[i,:4]=[np.log1p(state['count']),len(state['days']),
                    np.exp(-(target.start-state['last'])/(3*86400)),hour in state['hours']]
            features[i,4:]=[np.log1p(self.popularity[peer]),np.log1p(len(row)),state is None,
                            np.log1p(self.host_frequency[peer])]
        return features


class ContactForecaster:
    def __init__(self, source_rows, vocabulary, max_queries=128, seed=17):
        self.source_rows=source_rows
        self.vocabulary=np.asarray(vocabulary)
        self.max_queries=max_queries
        self.seed=seed

    @staticmethod
    def score(model, features):
        if model is None:
            # Объявленный cold-start baseline до первого доступного обучающего перехода.
            return features @ np.array([0.2,0.8,1.2,0.5,0.15,0,-0.3,-0.1])
        return model.decision_function(features)

    def fit(self, observations):
        ordered=sorted(observations,key=lambda o:o.window.start)
        rng=np.random.default_rng(self.seed)
        x,y=[],[]
        self.folds=[]
        model=None
        days=sorted({o.window.day for o in ordered})
        for day in days[1:]:
            targets=[o for o in ordered if o.window.day==day]
            past=[o for o in ordered if o.window.day<day]
            history=ContactHistory(past)
            if history.end>min(o.window.start for o in targets):
                raise ValueError('исторические дни пересекаются')
            if model is not None and model.training_end_>min(o.window.start for o in targets):
                raise ValueError('прогноз контакта обучался на будущем')
            for obs in targets:
                self.folds.append((obs,history,model))
                target=contacts(obs)
                queries=np.flatnonzero(obs.events>=300)
                queries=rng.choice(queries,min(self.max_queries,len(queries)),replace=False)
                for query in queries:
                    valid=np.array([p not in self.source_rows[query] for p in self.vocabulary])
                    positive=np.array([p in target[query] for p in self.vocabulary]) & valid
                    pos=np.flatnonzero(positive)
                    neg=np.flatnonzero(valid & ~positive)
                    if not len(pos) or not len(neg): continue
                    neg=rng.choice(neg,min(len(neg),max(8,5*len(pos))),replace=False)
                    chosen=np.r_[pos,neg]
                    x.append(history.features(query,self.vocabulary[chosen],obs.window))
                    y.append(np.r_[np.ones(len(pos)),np.zeros(len(neg))])
            # Обновление параметров только после всех прогнозов полностью отложенного дня.
            if x and len(np.unique(np.concatenate(y)))==2:
                model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=self.seed))
                model.fit(np.vstack(x),np.concatenate(y))
                model.training_end_=max(o.window.end for o in targets)
        if not self.folds: raise ValueError('нужны минимум два обучающих окна')
        self.model=model
        self.history=ContactHistory(ordered)
        return self


def swap_teacher(source, matrix, vocabulary, source_rows, hosts, forecaster, max_queries=128,
                 proposal_k=16, k=10):
    """Исторический oracle при принудительных заменах. Не выдаётся за реализуемый embedding."""
    rng=np.random.default_rng(17)
    vocabulary=np.asarray(vocabulary)
    own={h:i for i,h in enumerate(vocabulary)}
    triples=[];statistics=[]
    source_unit=normalize(source)
    for obs,history,predictor in forecaster.folds:
        population=np.flatnonzero(obs.events>=300)
        available=np.asarray(matrix[population].sum(0)).ravel()>0
        possible=np.zeros(len(source),bool);possible[population]=True
        csc=matrix.tocsc()
        target=contacts(obs)
        records=[]
        for query in rng.permutation(population):
            valid=available.copy()
            valid[matrix[query].indices]=False
            if hosts[query] in own: valid[own[hosts[query]]]=False
            positive=np.flatnonzero(valid & np.array([p in target[query] for p in vocabulary]))
            if not len(positive): continue
            forecast=forecaster.score(predictor,history.features(query,vocabulary,obs.window))
            direct=first_rr(forecast,positive,valid)
            desired=np.argsort(-np.where(valid,forecast,-np.inf),kind='stable')[:8]
            donors,weights=selected_neighbors(source,query,population,k)
            current=votes(matrix,donors,weights)
            base_rr=first_rr(current,positive,valid)
            similarity=source_unit @ source_unit[query]
            similarity[~possible]=-np.inf;similarity[query]=-np.inf
            candidates=list(np.argsort(-similarity,kind='stable')[:proposal_k])
            for peer in desired:
                support=csc.indices[csc.indptr[peer]:csc.indptr[peer+1]]
                candidates.extend(support[possible[support]].tolist())
            candidates=sorted(set(candidates)-set(donors)-{int(query)},key=lambda h:(-similarity[h],h))[:2*proposal_k]
            best=(base_rr,None,None)
            for added in candidates:
                if similarity[added]<=0: continue
                for slot,removed in enumerate(donors):
                    replacement=donors.copy();replacement[slot]=added
                    replaced_weights=weights.copy();replaced_weights[slot]=similarity[added]
                    proposed=votes(matrix,replacement,replaced_weights)
                    rr=first_rr(proposed,positive,valid)
                    if rr>best[0]+1e-12: best=(rr,int(added),int(removed))
            if best[1] is not None:
                triples.append((int(query),best[1],best[2],best[0]-base_rr))
            records.append((base_rr,best[0],direct))
            if len(records)>=max_queries: break
        values=np.asarray(records)
        statistics.append({'window':obs.window.name,'history_end':history.end,'target_start':obs.window.start,
                           'forecast_training_end_before_target':True,'n_queries':len(records),
                           'baseline_mrr':float(values[:,0].mean()) if len(values) else None,
                           'forced_swap_oracle_mrr':float(values[:,1].mean()) if len(values) else None,
                           'direct_forecast_mrr':float(values[:,2].mean()) if len(values) else None})
    return np.asarray(triples,dtype=float).reshape(-1,4),statistics


def fit_residual(source, triples, steps=150, rank=64, strength=0.2, seed=17):
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(2);torch.manual_seed(seed)
    if not len(triples): raise ValueError('teacher не нашёл полезных замен; обучать голову не на чем')
    base=torch.tensor(np.asarray(source),dtype=torch.float32)
    lengths=base.norm(dim=1,keepdim=True)
    projection=torch.linalg.qr(torch.randn(base.shape[1],min(rank,base.shape[1]))).Q.T
    codes=torch.nn.Parameter(torch.zeros(len(base),projection.shape[0]))
    optimizer=torch.optim.AdamW([codes],lr=0.03,weight_decay=0.01)
    truth=F.normalize(base,dim=1)
    rng=np.random.default_rng(seed)
    history=[]
    for step in range(steps):
        selected=triples[rng.choice(len(triples),min(128,len(triples)),replace=False)]
        q,good,bad=[torch.tensor(selected[:,i],dtype=torch.long) for i in range(3)]
        gain=torch.tensor(selected[:,3],dtype=torch.float32)
        z=F.normalize(base+strength*lengths*(codes@projection),dim=1)
        ranking=(gain*F.softplus(((z[q]*z[bad]).sum(1)-(z[q]*z[good]).sum(1)+0.05)/0.1)).sum()/gain.sum()
        preserve=F.mse_loss(z[q]@z.T,truth[q]@truth.T)
        loss=ranking+3*preserve+0.01*(codes**2).mean()
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([codes],3);optimizer.step()
        if step%25==0: history.append({'step':step,'loss':float(loss.detach()),'geometry':float(preserve.detach())})
    return {'codes':codes.detach().numpy(),'projection':projection.numpy(),'strength':strength,'history':history}


def apply_residual(source, state, strength=None):
    strength=state['strength'] if strength is None else strength
    return np.asarray(source)+strength*np.linalg.norm(source,axis=1,keepdims=True)*(state['codes']@state['projection'])


def assess_forecast(source,matrix,vocabulary,source_rows,hosts,forecaster,observation,max_queries=128):
    """Прямой прогноз и обычное голосование на одном фиксированном подмножестве запросов."""
    population=np.flatnonzero(observation.events>=300)
    available=np.asarray(matrix[population].sum(0)).ravel()>0
    target=contacts(observation)
    own={h:i for i,h in enumerate(vocabulary)}
    rng=np.random.default_rng(31)
    rows=[]
    for q in rng.permutation(population):
        valid=available.copy();valid[matrix[q].indices]=False
        if hosts[q] in own: valid[own[hosts[q]]]=False
        positives=np.flatnonzero(valid & np.array([p in target[q] for p in vocabulary]))
        if not len(positives): continue
        features=forecaster.history.features(q,vocabulary,observation.window)
        if forecaster.model is not None and forecaster.model.training_end_>observation.window.start:
            raise ValueError('forecast fit пересекает оценку')
        direct=first_rr(forecaster.score(forecaster.model,features),positives,valid)
        heuristic=first_rr(forecaster.score(None,features),positives,valid)
        popularity=first_rr(np.array([forecaster.history.popularity[p] for p in vocabulary]),positives,valid)
        own_history=forecaster.history.hosts[q]
        recurrence=first_rr(np.array([own_history.get(p,{}).get('count',0) for p in vocabulary]),positives,valid)
        donors,weights=selected_neighbors(source,q,population)
        ordinary=first_rr(votes(matrix,donors,weights),positives,valid)
        rows.append((direct,ordinary,popularity,recurrence,heuristic))
        if len(rows)>=max_queries: break
    values=np.asarray(rows)
    return {'n_queries':len(rows),'history_end':forecaster.history.end,
            'direct_forecast_mrr':float(values[:,0].mean()) if len(values) else None,
            'source_embedding_mrr_same_queries':float(values[:,1].mean()) if len(values) else None,
            'popularity_mrr_same_queries':float(values[:,2].mean()) if len(values) else None,
            'history_mrr_same_queries':float(values[:,3].mean()) if len(values) else None,
            'fixed_temporal_heuristic_mrr_same_queries':float(values[:,4].mean()) if len(values) else None,
            'direct_minus_embedding_paired_ci':bootstrap_ci(lambda i:float((values[i,0]-values[i,1]).mean()),
                                                          len(values),n_boot=500,seed=31) if len(values) else None,
            'sampling_seed':31,'not_comparable_to_full_population_mrr':True}
