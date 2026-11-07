import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

from .matrix import exact_vectors,PairOrder,optimize,psd_edge_vectors
from .metrics import WindowTask,first_rr,contact_matrix
from evaluation.partners import partner_ranking


def test_exact_vectors_preserve_gram_and_official_positive_votes():
    rng=np.random.default_rng(17);n=24
    raw=rng.normal(size=(n,n));S=(raw+raw.T)/2;np.fill_diagonal(S,0)
    Z,info=exact_vectors(S)
    np.testing.assert_allclose(Z@Z.T,np.eye(n)+info['epsilon']*S,atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(Z,axis=1),1,atol=1e-12)
    hosts=[f'H{i}' for i in range(n)];vocabulary=[f'P{i}' for i in range(30)]
    source=[{p:int(rng.integers(1,100)) for p in vocabulary if rng.random()<.25} for _ in hosts]
    target=[{p:1 for p in vocabulary if p not in row and rng.random()<.2} for row in source]
    records=[(h,p,w,c) for w,rows in [('source',source),('target',target)] for h,row in zip(hosts,rows) for p,c in row.items()]
    actual=partner_ranking(pd.DataFrame(Z,index=hosts),pd.DataFrame(records,columns=['host_id','partner','window','count']))
    M=contact_matrix(source,vocabulary);expected=[]
    for q in range(n):
        row=S[q].copy();row[q]=-np.inf;top=np.argpartition(-row,9)[:10]
        votes=np.asarray(M[top].T@np.maximum(row[top],0)).ravel()
        valid=np.array([p not in source[q] for p in vocabulary]);positive=np.flatnonzero([p in target[q] for p in vocabulary])
        if len(positive):expected.append(first_rr(votes,positive,valid))
    assert actual['new']['embedding']['mrr']==pytest.approx(np.mean(expected),abs=1e-12)
    with pytest.raises(ValueError):exact_vectors(S,epsilon=1.)


def test_psd_edge_control_keeps_same_offdiagonal_order_and_unit_lengths():
    rng=np.random.default_rng(1);x=rng.normal(size=(30,5));x/=np.linalg.norm(x,axis=1,keepdims=True)
    S=x@x.T;np.fill_diagonal(S,0)
    z,report=psd_edge_vectors(S)
    assert report['minimum_eigenvalue_K']>0
    np.testing.assert_allclose(z@z.T,np.eye(30)+report['epsilon']*S,atol=1e-12)


def test_pair_order_delta_matches_global_auc_with_shared_endpoints():
    S=np.array([[0,.3,.2,.9],[.3,0,.4,.1],[.2,.4,0,.8],[.9,.1,.8,0.]])
    a=np.array([0,0,0,1,1,2]);b=np.array([1,2,3,2,3,3]);y=np.array([1,0,0,1,0,1])
    order=PairOrder(S,(a,b,y));before=roc_auc_score(y,S[a,b])
    delta=order.delta(0,1,.3,.95);order.commit(0,1,.3,.95)
    S[0,1]=S[1,0]=.95
    assert delta==pytest.approx(roc_auc_score(y,S[a,b])-before)
    assert order.positive==sorted(S[a[y==1],b[y==1]].tolist())


@pytest.mark.parametrize('objective',['weighted','score_proxy'])
def test_coordinate_updates_improve_exact_joint_training_objective(objective):
    rng=np.random.default_rng(7);n=24
    task=WindowTask.__new__(WindowTask);task.population=np.arange(n);task.mask=np.ones(n,bool)
    task.name='history';task.vocabulary=[f'P{i}' for i in range(n)];task.matrix=sp.eye(n,format='csr')
    task.positive=[np.array([(i+1)%n]) for i in range(n)];task.old=[np.array([i]) for i in range(n)]
    task.own=np.full(n,-1);task.n_queries=n;task.ceiling=1.
    task.truth={i:set((i+np.arange(1,11))%n) for i in range(n)}
    a,b=np.triu_indices(n,1);task.pairs=(a,b,((b-a)%4==0).astype(int))
    S=rng.uniform(.1,.8,(n,n));S=(S+S.T)/2;np.fill_diagonal(S,0);original=S.copy()
    trained,history=optimize(S,[task],passes=1,queries_per_pass=8,proposals=4,objective=objective)
    def value(x):
        s=task.summary(x)
        if objective=='weighted':return s['new_partners_mrr']+2*s['neighbors_at_10']+2*s['pair_auc']
        from .matrix import score_proxy
        return score_proxy([s['new_partners_mrr'],s['neighbors_at_10'],s['pair_auc']],task,task.partner_chance())
    assert value(trained)>=value(original)-1e-12
    assert value(trained)-value(original)==pytest.approx(history[0]['objective_gain'],abs=1e-10)
    if objective=='score_proxy':assert task.summary(trained)['neighbors_at_10']>=.95*task.summary(original)['neighbors_at_10']-1e-12
    np.testing.assert_array_equal(S,original)
    np.testing.assert_array_equal(trained,trained.T);assert not np.diag(trained).any()
