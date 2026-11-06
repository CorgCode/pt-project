import numpy as np
import pandas as pd
import pytest

torch=pytest.importorskip('torch')

from .codebook import PartnerTask, train_codes, graph_block
from .data import Window, Observation
from .plus_experiment import select_identity, PlusDelivery


def obs(day, rows):
    return Observation(Window(f'd{day}',day,day*86400,day*86400+3600,'unused'),
                       rows,np.zeros((len(rows),20)),np.full(len(rows),300))


def test_partner_targets_match_population_and_any_new_answer():
    hosts=['A','B','C','D']
    source=[{'X':29,'Y':144},{'Z':5},{'W':7},{'hidden':1}]
    target=obs(3,[{'p:o:Z':1,'p:o:W':1,'p:o:Y':1,'p:o:A':1},{},{},{}])
    target.events[-1]=0
    task=PartnerTask(source,target,hosts)
    q=task.vocabulary.index('Z'); w=task.vocabulary.index('W')
    assert task.positive[0,q] and task.positive[0,w]
    assert task.positive[0].nnz==2
    assert 'hidden' not in task.vocabulary
    assert not task.valid[0,task.vocabulary.index('Y')]
    assert task.eligible.tolist()==[0]


def test_source_training_is_finite_and_does_not_mutate_teacher():
    source=[{'X':1},{'Y':1},{'Z':1},{'W':1}]
    target=obs(3,[{'p:o:Y':1},{'p:o:Z':1},{'p:o:W':1},{'p:o:X':1}])
    task=PartnerTask(source,target,['A','B','C','D'])
    anchor=np.random.default_rng(0).normal(size=(4,3)).astype('float32')
    before=anchor.copy()
    torch.set_num_threads(1)
    trained,_=train_codes(anchor,[task],steps=3)
    np.testing.assert_array_equal(anchor,before)
    assert np.isfinite(trained).all()
    np.testing.assert_allclose(np.linalg.norm(trained,axis=1),1,atol=1e-6)


def test_graph_is_independent_of_input_directions():
    first=obs(2,[{'p:o:B':4},{'p:i:A':4,'p:o:C':1},{'p:i:B':1,'p:o:D':3},{'p:i:C':3}])
    second=obs(2,[{'p:i:B':4},{'p:o:A':4,'p:i:C':1},{'p:o:B':1,'p:i:D':3},{'p:o:C':3}])
    a=graph_block([first],['A','B','C','D'],dim=2)
    b=graph_block([second],['A','B','C','D'],dim=2)
    np.testing.assert_allclose(a@a.T,b@b.T,atol=1e-6)


def test_identity_selection_can_keep_original_adapter():
    rng=np.random.default_rng(5)
    values=rng.normal(size=(40,5))
    observation=obs(4,[{'p:o:common':300} for _ in range(40)])
    class BadAdapter:
        def predict(self,obs,variant,temperature):
            return values[np.roll(np.arange(40),1)]
    selection,rows=select_identity(pd.DataFrame(values),BadAdapter(),[values],[observation])
    assert selection['variant']=='baseline'
    assert len(rows)==9


def test_delivery_changes_with_traffic_not_row_identity():
    class Base:
        hosts=['A','B','C']
        def late(self,o,blend): return o.numeric[:,:3]
    model=PlusDelivery(Base(),None,{'variant':'baseline','temperature':.02},.5,np.eye(3))
    original=obs(4,[{}, {}, {}]);original.numeric[:,:3]=np.eye(3)
    permuted=obs(4,[{}, {}, {}]);permuted.numeric=original.numeric[[2,0,1]]
    np.testing.assert_array_equal(model.predict(permuted),model.predict(original)[[2,0,1]])


def test_reference_keeps_full_source_interval_for_links_probe(tmp_path,monkeypatch):
    from . import plus_reference
    n=30
    hosts=[f'H{i}' for i in range(n)]
    first=obs(2,[{f'p:o:{hosts[(i+1)%n]}':300} for i in range(n)])
    later=obs(4,[{f'p:o:{hosts[(i+2)%n]}':300} for i in range(n)])
    for observation in (first,later):
        file=tmp_path/(observation.window.name+'.csv');file.write_text('source fixture')
        observation.window.flows=str(file)
    values=np.random.default_rng(0).normal(size=(n,6))
    monkeypatch.setattr(plus_reference,'behavior',lambda *a,**kw:(values,None))
    path=tmp_path/'ref'
    metadata=plus_reference.build(first,later,hosts,path,tmp_path,source_behavior=values)
    edges=pd.read_csv(path/'edges.csv')
    start=pd.Timestamp(first.window.start,unit='s').isoformat()+'Z'
    end=pd.Timestamp(first.window.end-1,unit='s').isoformat()+'Z'
    assert edges.source_window.unique().tolist()==[start+'/'+end]
    assert edges.target_window.str.contains('00:59:59').all()
    assert metadata['n_evaluable']==n
    assert plus_reference.build(first,later,hosts,path,tmp_path,source_behavior=values)==metadata
    (path/'pairs.csv').write_text('changed')
    with pytest.raises(ValueError,match='reference CSV changed'):
        plus_reference.build(first,later,hosts,path,tmp_path,source_behavior=values)


def test_explicit_source_can_follow_earlier_training_hours(monkeypatch):
    from . import codebook
    n=6;hosts=[f'H{i}' for i in range(n)]
    early=obs(2,[{f'p:o:{hosts[(i+1)%n]}':300} for i in range(n)])
    source=obs(2,[{f'p:o:{hosts[(i+2)%n]}':300} for i in range(n)])
    source.window.start+=3600;source.window.end+=3600
    later=obs(3,[{f'p:o:{hosts[(i+3)%n]}':300} for i in range(n)])
    rng=np.random.default_rng(3)
    for observation in (early,source,later): observation.numeric=rng.normal(size=(n,20)).astype('float32')
    captured=[]
    class Features:
        def __init__(self,*a): pass
        def fit(self,history,source_observation):
            captured.append(source_observation);return self
        def transform(self,o): return o.numeric[:,:8]
    monkeypatch.setattr(codebook,'TrafficFeatures',Features)
    def head(history,active,teacher,dim,**kw):
        np.testing.assert_array_equal(teacher,source.numeric[:,:8])
        net=torch.nn.Sequential(torch.nn.Linear(8,dim),torch.nn.Tanh())
        net.eval();return net,[]
    monkeypatch.setattr(codebook,'train_traffic_head',head)
    def codes(anchor,tasks,steps,dim=None,**kw):
        if dim is None: return codebook.unit(anchor),[]
        return codebook.unit(rng.normal(size=(n,dim))),[]
    monkeypatch.setattr(codebook,'train_codes',codes)
    model=codebook.RebuiltPlus().fit([early,source,later],hosts,'unused',source_observation=source)
    assert captured==[source]
    np.testing.assert_allclose(model.codes,model.traffic(source),atol=1e-6)
    assert not np.allclose(model.codes,model.traffic(early))


def test_delivery_checkpoint_reloads_in_a_fresh_process(tmp_path):
    import joblib
    import subprocess
    import sys
    from types import SimpleNamespace
    model=PlusDelivery(SimpleNamespace(hosts=['A']),None,{'variant':'baseline','temperature':.02},.5,np.eye(1))
    path=tmp_path/'model.joblib';joblib.dump(model,path)
    result=subprocess.run([sys.executable,'-c',
        "import joblib,sys; m=joblib.load(sys.argv[1]); assert m.hosts==['A']; "
        "assert m.__class__.__module__=='experiments.frozen_adapters.codebook'",str(path)],
        capture_output=True,text=True)
    assert result.returncode==0,result.stderr
