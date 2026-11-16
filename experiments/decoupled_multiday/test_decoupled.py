"""Tests of the actual objective, frozen-base/dimension contract and no-ID inference."""
import copy
import joblib
import numpy as np
import pytest
import scipy.sparse as sp
import torch
from sklearn.preprocessing import StandardScaler
from experiments.similarity_graph.data import Window,Observation
from .collaborative import als_half,ImplicitALS,PartnerTask,listwise_loss,unit,interaction_matrix
from .fingerprint import ContrastiveFingerprint,FingerprintFeatures
from .model import assemble,FrozenCodebook,DecoupledModel,frame
from .search import from_grams,cached_block_grams,raw_utility


def observations():
    rng=np.random.default_rng(4);result=[];n=24
    for day in [2,3,4]:
        counts=[{f'p:o:Server{i%6}':1000+i,f's:o:6/{80+i%4}':800,
                 f'c:o:Server{i%6}|6/{80+i%4}':1000+i,f'p:i:Other{i%3}':300} for i in range(n)]
        result.append(Observation(Window(str(day),day,day*100,day*100+60,'unused'),
                                  counts,rng.random((n,20)).astype(np.float32),np.full(n,1500)))
    return result


def test_als_normal_equation_matches_dense_objective():
    rng=np.random.default_rng(3);x=sp.csr_matrix([[1.,0,2.],[0,3.,0]])
    y=rng.normal(size=(3,4));u=als_half(x,y,2.,5.)
    for i in range(2):
        row=x.toarray()[i];c=1+5*row;p=(row>0).astype(float)
        expected=np.linalg.solve(y.T@np.diag(c)@y+2*np.eye(4),y.T@(c*p))
        np.testing.assert_allclose(u[i],expected,atol=1e-12)
    assert np.all(als_half(sp.csr_matrix((1,3)),y,2,5)==0)


def test_ials_is_seeded_and_handles_empty_users():
    x=sp.csr_matrix([[1.,0,2],[0,3,0],[0,0,0]])
    a=ImplicitALS(dim=3,steps=2).fit(x);b=ImplicitALS(dim=3,steps=2).fit(x)
    np.testing.assert_array_equal(a.users,b.users)
    assert np.all(a.users[2]==0)


def test_interactions_keep_joint_service_and_decay():
    obs=observations();x,v=interaction_matrix(obs)
    assert 'Server0|6/80' in v and all(not k.startswith('p:') for k in v)
    expected=sum(2**(-(4-o.window.day)/3)*np.log1p(1000*60) for o in obs)
    np.testing.assert_allclose(x[0,v.index('Server0|6/80')],expected)


def test_straight_through_votes_are_actual_ten_donor_votes():
    from experiments.similarity_graph.data import contacts
    obs=observations();hosts=[f'h{i}' for i in range(24)]
    # Include a genuine NEW peer that is in the SOURCE vocabulary.
    obs[1].counts[0]['p:o:Server1']=30
    task=PartnerTask(contacts(obs[0]),obs[1],hosts)
    rng=np.random.default_rng(5)
    blocks=[torch.tensor(unit(rng.normal(size=(24,d))),requires_grad=True) for d in [9,4,5,3]]
    q=np.array([0]);loss,votes=listwise_loss(blocks[0],blocks[1],blocks[3],blocks[2],q,task)
    z=torch.nn.functional.normalize(torch.cat([blocks[0],blocks[1]*.15,blocks[2]*.5,blocks[3]*.1],1),dim=1).detach().numpy()
    sim=z[q]@z.T;sim[0,0]=-1e9;top=np.argsort(-sim[0])[:10]
    expected=np.asarray(task.matrix[top].T@sim[0,top].clip(min=0)).ravel()
    np.testing.assert_allclose(votes.detach().numpy()[0],expected,rtol=2e-6,atol=1e-4)
    loss.backward();assert blocks[2].grad is not None and torch.isfinite(blocks[2].grad).all()


def test_fixed_dimension_preserves_base_and_scaler_erases_block_weight():
    rng=np.random.default_rng(6);blocks=[unit(rng.normal(size=(24,d))) for d in [896,32,32,64]]
    a=assemble(*blocks,(1.,.1,.3,.01));b=assemble(*blocks,(1.,.4,.8,.1))
    np.testing.assert_array_equal(a[:,:896],blocks[0]);assert a.shape==(24,1024)
    np.testing.assert_allclose(StandardScaler().fit_transform(a),StandardScaler().fit_transform(b),rtol=2e-5,atol=2e-5)
    assert assemble(*blocks,(1.,0,0,0)).shape==(24,896)
    with pytest.raises(ValueError,match='exceeds'):assemble(blocks[0],np.zeros((24,64)),np.zeros((24,64)),blocks[3],(1,1,1,1))


def test_block_cosines_match_materialized_vectors():
    rng=np.random.default_rng(7);blocks=[rng.normal(size=(24,d)) for d in [896,32,32,64]]
    blocks[1:]=[unit(b) for b in blocks[1:]];pop=np.arange(24);weights=(1.,.2,.5,.1)
    grams,norms=cached_block_grams(blocks,pop);s=from_grams(grams,norms,weights)
    z=assemble(*blocks,weights).astype(float);z/=np.linalg.norm(z,axis=1,keepdims=True)
    np.testing.assert_allclose(s,z@z.T,atol=1e-8)
    assert raw_utility([{'x':.2,'y':.8}])==pytest.approx(.4)


def test_fingerprint_permutation_no_id_feature_and_checkpoint(tmp_path):
    obs=observations();model=ContrastiveFingerprint(dim=4,steps=2,peer_width=64,service_width=32).fit(obs[:2])
    assert model.fold_provenance==[{'query_day':3,'history_days':[2]}]
    permutation=np.arange(24)[::-1]
    other=copy.deepcopy(obs[2]);other.counts=[other.counts[i] for i in permutation]
    other.numeric=other.numeric[permutation];other.events=other.events[permutation]
    # Window name/path/day and host labels cannot change transformed traffic.
    np.testing.assert_allclose(model.query(other),model.query(obs[2])[permutation],atol=1e-6)
    p,q=model.assignments(obs[2],1);assert np.all(np.diff(p.indptr)==1)
    np.testing.assert_allclose(np.asarray(p.sum(1)).ravel(),1)
    hosts=[f'h{i}' for i in range(24)];rng=np.random.default_rng(8)
    base=FrozenCodebook(hosts,unit(rng.normal(size=(24,896))),360)
    delivery=DecoupledModel(base,model,rng.normal(size=(24,4)),rng.normal(size=(24,8)),(1,.2,.3,.1))
    path=tmp_path/'model.joblib';joblib.dump(delivery,path)
    np.testing.assert_array_equal(delivery.predict(obs[2]),joblib.load(path).predict(obs[2]))


def test_frozen_base_rejects_future_cutoff(tmp_path):
    from .train import load_base
    from experiments.similarity_graph.data import signature
    hosts=['a','b'];base=FrozenCodebook(hosts,np.ones((2,896)),900)
    path=tmp_path/'base.joblib';joblib.dump(base,path)
    config={'base':{'checkpoint':str(path),'sha256':signature(path)}}
    with pytest.raises(ValueError,match='later training cutoff'):
        load_base(config,[Window('old',2,200,300,'unused')],hosts)


def test_external_base_precision_is_preserved():
    rng=np.random.default_rng(71);codes=rng.normal(size=(3,896))
    base=FrozenCodebook(['a','b','c'],codes,300)
    assert base.codes.dtype==np.float64
    x=assemble(base.codes,np.ones((3,32)),np.ones((3,32)),np.ones((3,64)),(1,.1,.3,.1))
    np.testing.assert_array_equal(x[:,:896],codes)


def test_raw_window_votes_and_neighbors_match_project(tmp_path):
    from evaluation.frozen_neighbors import save_frozen_neighbors,evaluate_frozen_neighbors
    from evaluation.partners import partner_ranking
    from experiments.similarity_graph.metrics import partner_frame
    from experiments.similarity_graph.data import contacts
    from .search import RawWindow
    import json
    obs=observations();hosts=[f'h{i}' for i in range(24)]
    obs[1].counts[0]['p:o:Server1']=200
    rng=np.random.default_rng(21);z=unit(rng.normal(size=(24,8)))
    save_frozen_neighbors(frame(rng.normal(size=(24,5)),hosts),tmp_path/'neighbors.csv',reference_window='test',k_max=10)
    # Independent pair labels with both classes.
    import pandas as pd
    pd.DataFrame({'host_a':hosts[:12],'host_b':hosts[12:],'label':[0,1]*6}).to_csv(tmp_path/'pairs.csv',index=False)
    (tmp_path/'manifest.json').write_text(json.dumps({'neighbor_ceiling':.5}))
    raw=RawWindow(contacts(obs[0]),obs[1],hosts,tmp_path)
    result=raw.evaluate(z.astype(float)@z.astype(float).T)
    official=partner_ranking(frame(z,hosts),partner_frame(contacts(obs[0]),contacts(obs[1]),hosts,np.arange(24)))
    assert result['partner_new_mrr']==pytest.approx(official['new']['embedding']['mrr'],abs=1e-12)
    neighbors=evaluate_frozen_neighbors(frame(z,hosts),tmp_path/'neighbors.csv',ks=(10,)).loc[(10,'consistency'),'mean']
    assert result['consistency@10_mean']==pytest.approx(neighbors,abs=1e-12)
