"""Sparse partial assignment with graph messages; no own-ID candidate boosts or diagonal lookup.

This is a transparent iterative baseline inspired by partial graph matching, not an
implementation of Ratnayaka et al.'s trained ICLR2025 optimal-partial-transport model.
"""
import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize,StandardScaler
from .data import contacts


def network(observation,host_order):
    index={h:i for i,h in enumerate(host_order)};rr,cc,vv=[],[],[]
    for row,peers in enumerate(contacts(observation)):
        for peer,count in peers.items():
            if peer in index and index[peer]!=row:
                rr.append(row);cc.append(index[peer]);vv.append(np.log1p(count))
    return sp.csr_matrix((vv,(rr,cc)),shape=(len(host_order),len(host_order)))


def norm_graph(matrix):
    graph=(matrix+matrix.T)*.5
    # Downweight hubs; preserve symmetry and keep one-hop messages comparable.
    inv=1/np.sqrt(np.maximum(np.asarray(graph.sum(1)).ravel(),1e-12))
    return (sp.diags(inv)@graph@sp.diags(inv)).tocsr()


def partial_assignment(logits,candidates,n_sources,unknown_logit,temperature=.03,iterations=60):
    """KL projection: row mass=1 including unknown, each REAL column mass <=1."""
    offset=np.maximum(np.max(logits,1),unknown_logit)
    kernel=np.exp((logits-offset[:,None])/temperature)
    dust=np.exp((unknown_logit-offset)/temperature)
    v=np.ones(n_sources)
    for _ in range(iterations):
        u=1/np.maximum((kernel*v[candidates]).sum(1)+dust,1e-300)
        demand=np.bincount(candidates.ravel(),weights=(kernel*u[:,None]).ravel(),minlength=n_sources)
        new_v=np.minimum(1.,1/np.maximum(demand,1e-300))
        if np.max(np.abs(v-new_v))<1e-9:v=new_v;break
        v=new_v
    u=1/np.maximum((kernel*v[candidates]).sum(1)+dust,1e-300)
    real=kernel*v[candidates]*u[:,None]
    # Final feasibility projection, including finite-iteration/numerical residuals.
    sums=np.bincount(candidates.ravel(),weights=real.ravel(),minlength=n_sources)
    real*=np.minimum(1.,1/np.maximum(sums,1e-300))[candidates]
    unknown=1-real.sum(1)
    return real,unknown


class GraphMatcher:
    def fit(self,observations,hosts,joint_context=False,cache=None):
        self.hosts=list(hosts);self.training_end=max(o.window.end for o in observations)
        self.joint_context=joint_context;self.context_cache=str(cache) if cache else None
        if joint_context:
            if cache is None:raise ValueError('joint context needs a cache directory')
            from .features import with_context
            observations=[with_context(o,hosts,cache) for o in observations]
        self.vectorizer=DictVectorizer(sparse=True,sort=True)
        def rows(obs):
            rate=3600/(obs.window.end-obs.window.start)
            return [{k:np.log1p(v*rate) for k,v in row.items()} for row in obs.counts]
        self.vectorizer.fit([r for o in observations for r in rows(o)])
        raw=[self.vectorizer.transform(rows(o)).tocsr() for o in observations]
        aggregate=sum(raw,sp.csr_matrix(raw[0].shape))
        df=np.asarray((aggregate>0).sum(0)).ravel()
        self.idf=np.log((len(hosts)+1)/(df+1))+1
        active=[o.events>=300 for o in observations]
        # Distinct-day persistence, not an advantage from duplicating the same day's hours.
        days=sorted(set(o.window.day for o in observations))
        presence=[]
        for day in days:
            total=sum([x.multiply(a[:,None]) for x,a,o in zip(raw,active,observations) if o.window.day==day],
                      sp.csr_matrix(raw[0].shape))
            presence.append((total>0).astype(float))
        stability=sum(presence,sp.csr_matrix(raw[0].shape))/len(days)
        self.profiles=sp.vstack([normalize(x.multiply(self.idf).multiply(stability).multiply(a[:,None]))
                                for x,a in zip(raw,active)],format='csr')
        self.n_history=len(raw)
        numeric=np.vstack([o.numeric[a] for o,a in zip(observations,active)])
        self.scaler=StandardScaler().fit(numeric)
        self.numeric=np.zeros((len(hosts),numeric.shape[1]));counts=np.zeros(len(hosts))
        self.graph=sp.csr_matrix((len(hosts),len(hosts)))
        latest=max(o.window.day for o in observations)
        for o,a in zip(observations,active):
            w=2**(-(latest-o.window.day)/3.)
            self.numeric+=self.scaler.transform(o.numeric)*a[:,None]*w;counts+=a*w
            self.graph+=network(o,hosts)*w
        self.numeric/=np.maximum(counts[:,None],1)
        self.graph=norm_graph(self.graph)
        return self

    def unary(self,observation,k=30,target_hosts=None):
        if getattr(self,'joint_context',False):
            from .features import with_context
            observation=with_context(observation,target_hosts or self.hosts,self.context_cache)
        rate=3600/(observation.window.end-observation.window.start)
        raw=self.vectorizer.transform([{p:np.log1p(v*rate) for p,v in row.items()} for row in observation.counts])
        query=normalize(raw.multiply(self.idf).tocsr())
        numeric=self.scaler.transform(observation.numeric)
        candidate=[];score=[]
        n=len(self.hosts);k=min(k,n)
        for start in range(0,query.shape[0],128):
            similarity=(query[start:start+128]@self.profiles.T).toarray().reshape(-1,self.n_history,n).max(1)
            # Small activity likelihood breaks traffic-count ties; no query row index enters its value.
            distance=np.mean((numeric[start:start+128,None,:]-self.numeric[None,:,:])**2,axis=2)
            similarity+=.03*np.exp(-distance/2)
            top=np.argsort(-similarity,axis=1,kind='stable')[:,:k]
            candidate.append(top);score.append(np.take_along_axis(similarity,top,axis=1))
        return np.vstack(candidate),np.vstack(score)

    def assign(self,observation,target_hosts=None,graph_weight=.1,iterations=5,k=30,unknown_logit=.12):
        if observation.window.start<self.training_end:raise ValueError('query window precedes training cutoff')
        target_hosts=list(target_hosts or self.hosts)
        if len(target_hosts)!=len(observation.counts):raise ValueError('target order mismatch')
        active=np.flatnonzero(observation.events>=300)
        candidates,unary=self.unary(observation,k,target_hosts)
        k=candidates.shape[1]
        candidates,unary=candidates[active],unary[active]
        values,unknown=partial_assignment(unary,candidates,len(self.hosts),unknown_logit)
        target=norm_graph(network(observation,target_hosts)[active][:,active])
        for _ in range(iterations if graph_weight else 0):
            P=sp.csr_matrix((values.ravel(),(np.repeat(np.arange(len(active)),k),candidates.ravel())),
                            shape=(len(active),len(self.hosts)))
            support=target@P
            compatibility=np.empty_like(values)
            for start in range(0,len(active),128):
                stop=min(start+128,len(active));repeat=np.repeat(np.arange(start,stop),k)
                historical=self.graph[candidates[start:stop].ravel()]
                compatibility[start:stop]=np.asarray(support[repeat].multiply(historical).sum(1)).reshape(-1,k)
            relative=compatibility/np.maximum(compatibility.max(1,keepdims=True),1e-12)
            proposed,unknown=partial_assignment(unary+graph_weight*relative,candidates,len(self.hosts),unknown_logit)
            values=.5*values+.5*proposed
        P=sp.csr_matrix((values.ravel(),(np.repeat(active,k),candidates.ravel())),shape=(len(target_hosts),len(self.hosts)))
        dust=np.ones(len(target_hosts));dust[active]=1-values.sum(1)
        info={'n_active':len(active),'mean_unknown_mass':float(dust[active].mean()),
              'maximum_real_column_mass':float(np.asarray(P.sum(0)).max()),
              'graph_weight':graph_weight,'iterations':iterations if graph_weight else 0,
              'own_id_used_as_candidate_feature':False}
        return P,dust,info


def late_vectors(assignment,unknown,source):
    center=np.mean(source,axis=0)
    if np.linalg.norm(center)<1e-12:center=np.eye(1,source.shape[1])[0]
    center=center/np.linalg.norm(center)
    result=np.asarray(assignment@source)+unknown[:,None]*center
    return normalize(result)


def harden_assignment(assignment,unknown):
    """Exact sparse linear partial assignment on the FINAL graph-refined posterior.

    Each query has its own dummy column, so rejection is always feasible. This does
    not solve the quadratic graph-matching optimum; it exactly solves its final linear subproblem.
    """
    from scipy.sparse.csgraph import min_weight_full_bipartite_matching
    active=np.flatnonzero(np.diff(assignment.indptr)>0)
    if not len(active):return assignment.copy(),unknown.copy()
    real=assignment[active].tocoo();n_sources=assignment.shape[1]
    rows=np.r_[real.row,np.arange(len(active))]
    cols=np.r_[real.col,n_sources+np.arange(len(active))]
    costs=np.r_[-np.log(np.maximum(real.data,1e-300))+1,
                -np.log(np.maximum(unknown[active],1e-300))+1]
    matrix=sp.csr_matrix((costs,(rows,cols)),shape=(len(active),n_sources+len(active)))
    query,candidate=min_weight_full_bipartite_matching(matrix)
    matched=candidate<n_sources
    out=sp.csr_matrix((np.ones(matched.sum()),(active[query[matched]],candidate[matched])),shape=assignment.shape)
    dust=np.ones(len(unknown));dust[active[query[matched]]]=0
    return out,dust
