from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize

from experiments.multiday.codebook import CodebookEncoder,TrafficAdapter,neighbor_loss,verify,fit_adapter


def test_neighbor_loss_preserves_teacher_and_updates_changed_codes():
    torch.set_num_threads(1);torch.manual_seed(3)
    teacher=torch.nn.functional.normalize(torch.randn(12,8),dim=1)
    assert abs(float(neighbor_loss(teacher,teacher,[0,2,4])))<1e-6
    codes=torch.nn.Parameter(teacher+torch.randn_like(teacher)*0.1)
    loss=neighbor_loss(torch.nn.functional.normalize(codes,dim=1),teacher,[0,2,4])
    loss.backward()
    assert loss>0 and torch.isfinite(codes.grad).all() and codes.grad.abs().sum()>0


def test_traffic_adapter_is_independent_of_query_positions_and_duplicate_ids():
    rng=np.random.default_rng(2);x=normalize(rng.normal(size=(30,8)))
    ridge=Ridge(alpha=0.1).fit(x,x@rng.normal(size=(8,8)))
    adapter=TrafficAdapter(ridge,0.5)
    expected=adapter.transform(x)
    np.testing.assert_allclose(adapter.transform(x[[5,0,8]]),expected[[5,0,8]],atol=1e-6)
    np.testing.assert_allclose(adapter.transform(x[[2,2]])[0],adapter.transform(x[[2,2]])[1])
    np.testing.assert_allclose(np.linalg.norm(expected,axis=1),1,atol=1e-6)


def test_encoder_only_looks_up_training_templates():
    class Base:
        def embed(self,data,window,hosts):
            return pd.DataFrame(np.tile([0.,1.],(len(hosts),1)),index=hosts)
    x=np.eye(2,dtype=np.float32)
    adapter=TrafficAdapter(Ridge().fit(x,x),0.)
    source=pd.DataFrame(x,index=['a','b'])
    model=CodebookEncoder(Base(),source,adapter)
    pd.testing.assert_frame_equal(model.embed(None,'train',['b','a']),source.loc[['b','a']])
    actual=model.embed(None,'test',['a','b'])
    np.testing.assert_allclose(actual.to_numpy(),[[0,1],[0,1]])


def test_revision_guard_rejects_missing_sources(tmp_path):
    import json,pytest
    (tmp_path/'training_revision.json').write_text(json.dumps({'sources':{},'torch':str(torch.__version__)}))
    with pytest.raises(RuntimeError,match='изменился'):verify(tmp_path)


def test_adapter_uses_chronological_holdout_and_then_all_history(monkeypatch):
    import experiments.multiday.codebook as module
    rng=np.random.default_rng(4);source=normalize(rng.normal(size=(30,8))).astype(np.float32)
    history=[source+rng.normal(size=source.shape)*0.01 for _ in range(4)]
    populations=[np.ones(30,dtype=bool) for _ in history]
    real=module.Ridge;calls=[]
    class RecordingRidge(real):
        def fit(self,x,y,**kwargs):
            calls.append((len(x),kwargs['sample_weight']))
            return super().fit(x,y,**kwargs)
    monkeypatch.setattr(module,'Ridge',RecordingRidge)
    adapter,records=fit_adapter(history,populations,source,[3,4,10,12],half_life=3.)
    assert [c[0] for c in calls]==[60,120]
    assert all(r['historical_holdout_days']==[10,12] for r in records)
    assert np.isclose(calls[-1][1].mean(),1.)
    assert calls[-1][1][-1]>calls[-1][1][0]


def test_training_rejects_changed_history_configuration(tmp_path,monkeypatch):
    import json,pytest
    import experiments.multiday.codebook as module
    monkeypatch.setattr(module,'verify',lambda work:None)
    (tmp_path/'training_configuration.json').write_text(json.dumps({'history_root':'wrong'}))
    with pytest.raises(RuntimeError,match='Конфигурация'):
        module.train(tmp_path,1.,history_root=tmp_path,target_days=(3,4,12),half_life=3.)
