import numpy as np
from .data import Observation,Window
from .matching import GraphMatcher,partial_assignment,harden_assignment
import scipy.sparse as sp


def observation(day,rows):
    n=len(rows)
    return Observation(Window(str(day),day,day*86400,day*86400+3600,'unused'),rows,
                       np.tile(np.arange(20),(n,1)).astype(float),np.full(n,300))


def test_partial_matching_can_reject_and_never_overfills_real_columns():
    candidates=np.array([[0,1],[0,1],[0,1]])
    logits=np.array([[.9,.1],[.9,.2],[-2.,-2.]])
    P,unknown=partial_assignment(logits,candidates,2,unknown_logit=.12)
    np.testing.assert_allclose(P.sum(1)+unknown,1,atol=1e-12)
    assert np.all(np.bincount(candidates.ravel(),weights=P.ravel())<=1+1e-12)
    assert unknown[-1]>.99


def test_hard_matching_resolves_competing_rows_and_can_reject():
    P=sp.csr_matrix([[.6,.4],[.59,.41],[1e-4,1e-4]])
    hard,unknown=harden_assignment(P,np.array([0.,0.,.9998]))
    np.testing.assert_array_equal(hard.toarray(),[[1,0],[0,1],[0,0]])
    np.testing.assert_array_equal(unknown,[0,0,1])


def test_identical_query_features_do_not_receive_their_own_id():
    hosts=[f'H{i}' for i in range(24)]
    train=[observation(day,[{f'p:o:P{i}':300,'s:o:6/80':300} for i in range(24)]) for day in (2,3)]
    model=GraphMatcher().fit(train,hosts)
    query=observation(4,[{'s:o:6/80':300} for _ in hosts])
    P,dust,info=model.assign(query,graph_weight=0)
    np.testing.assert_allclose(P.toarray(),np.tile(P.toarray()[0],(24,1)),atol=1e-12)
    assert info['own_id_used_as_candidate_feature'] is False


def test_joint_graph_matching_is_equivariant_to_query_row_order():
    hosts=[f'H{i}' for i in range(24)]
    rows=[{f'p:o:{hosts[(i+1)%24]}':300,f's:o:6/{100+i}':300} for i in range(24)]
    model=GraphMatcher().fit([observation(2,rows),observation(3,rows)],hosts)
    query=observation(4,rows);P,dust,_=model.assign(query,graph_weight=.1,iterations=2)
    order=np.random.default_rng(7).permutation(24)
    shuffled=observation(4,[rows[i] for i in order])
    alternate,dust2,_=model.assign(shuffled,[hosts[i] for i in order],graph_weight=.1,iterations=2)
    np.testing.assert_allclose(alternate.toarray(),P.toarray()[order],atol=1e-8)
    np.testing.assert_allclose(dust2,dust[order],atol=1e-8)


def test_saved_model_predicts_identically_in_a_fresh_process(tmp_path):
    import joblib,subprocess,sys
    from .model import SimilarityGraphModel
    hosts=[f'H{i}' for i in range(24)]
    rows=[{f'p:o:P{i}':300,'s:o:6/80':300} for i in range(24)]
    matcher=GraphMatcher().fit([observation(2,rows),observation(3,rows)],hosts)
    model=SimilarityGraphModel(np.eye(24),matcher,0.)
    query=observation(4,rows);expected,_=model.predict(query)
    model_file,obs_file,result=tmp_path/'model.joblib',tmp_path/'observation.joblib',tmp_path/'result.npy'
    joblib.dump(model,model_file);joblib.dump(query,obs_file)
    completed=subprocess.run([sys.executable,'-c',
        'import joblib,numpy as np,sys; m=joblib.load(sys.argv[1]); o=joblib.load(sys.argv[2]); '
        'p,_=m.predict(o); np.save(sys.argv[3],p)',str(model_file),str(obs_file),str(result)],
        capture_output=True,text=True)
    assert completed.returncode==0,completed.stderr
    np.testing.assert_allclose(np.load(result),expected,atol=1e-12)


def test_joint_context_counts_direction_service_and_excludes_self(tmp_path):
    import pandas as pd
    from .features import with_context
    file=tmp_path/'flows.csv'
    pd.DataFrame({'src':['A','A','A','A'],'dst':['B','B','A','B'],'proto':['6']*4,
                  'dst_port':['80','443','80','80'],'ts':[1,2,3,9],'duration':[0,0,0,2]}).to_csv(file,index=False)
    obs=Observation(Window('w',2,0,10,str(file)),[{'p:o:B':2},{'p:i:A':2}],np.zeros((2,20)),np.full(2,300))
    result=with_context(obs,['A','B'],tmp_path/'cache')
    assert result.counts[0]['c:o:B|6/80']==1
    assert result.counts[0]['c:o:B|6/443']==1
    assert result.counts[1]['c:i:A|6/443']==1
    assert not any('c:o:A|' in k for k in result.counts[0])
    assert result.counts==with_context(obs,['A','B'],tmp_path/'cache').counts
    assert not any(k.startswith('c:') for k in obs.counts[0])
