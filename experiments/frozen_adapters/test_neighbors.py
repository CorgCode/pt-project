import numpy as np
import pandas as pd
import pytest
from evaluation.partners import partner_ranking
from .neighbors import first_rr, contact_matrix, selected_neighbors, votes, ContactForecaster, fit_residual, apply_residual
from .run_neighbors import partner_frame,transfer_late
from .test_adapters import observations
from .data import contacts


def test_dominating_contact_and_midrank():
    valid=np.ones(2,bool)
    assert first_rr(np.array([29.,144.]),[0],valid)==0.5
    assert first_rr(np.zeros(2),[0],valid)==pytest.approx(2/3)
    assert first_rr(np.array([144.,29.]),[0,1],valid)==1


def test_votes_match_official_partner_evaluator():
    rng=np.random.default_rng(17)
    n=16;hosts=[f'h{i}' for i in range(n)];vocab=[f'p{i}' for i in range(18)]
    rows=[{p:int(rng.integers(1,10)) for p in vocab if rng.random()<0.3} for _ in hosts]
    matrix=contact_matrix(rows,vocab)
    frame=pd.DataFrame(rng.normal(size=(n,8)),index=hosts)
    valid=np.asarray(matrix.sum(0)).ravel()>0
    valid[matrix[0].indices]=False
    positive=np.flatnonzero(valid)[:2]
    assert len(positive)==2
    target=[{} for _ in hosts];target[0]={vocab[p]:1 for p in positive}
    donors,weights=selected_neighbors(frame.to_numpy(),0,np.arange(n))
    rr=first_rr(votes(matrix,donors,weights),positive,valid)
    reference=partner_frame(rows,target,hosts,np.arange(n))
    official=partner_ranking(frame,reference)
    assert rr==pytest.approx(official['new']['embedding']['mrr'])


def test_forecast_has_no_later_training_labels():
    obs=observations()
    rows=contacts(obs[0]);vocab=sorted({p for row in rows for p in row})
    model=ContactForecaster(rows,vocab,max_queries=10).fit(obs)
    for target,history,predictor in model.folds:
        assert history.end<=target.window.start
        assert predictor is None or predictor.training_end_<=target.window.start


def test_residual_zero_and_late_transfer_identity():
    pytest.importorskip('torch')
    source=np.random.default_rng(17).normal(size=(32,8))
    old=source.copy()
    state=fit_residual(source,np.array([[0,1,2,0.3],[3,4,5,0.1]]),steps=2,rank=4)
    np.testing.assert_array_equal(apply_residual(source,state,0),source)
    np.testing.assert_allclose(transfer_late(source,source,source),source,atol=1e-12)
    np.testing.assert_array_equal(source,old)


def test_full_four_way_evaluation_has_all_required_metrics(processed,reference,tmp_path):
    from pipeline.data import ProcessedData
    from .data import Window,read_window,contacts
    from .run_neighbors import evaluate_candidate
    data=ProcessedData(processed)
    hosts=data.cohort
    frame=pd.DataFrame(np.random.default_rng(17).normal(size=(len(hosts),8)),index=hosts).add_prefix('emb_')
    train=data.manifest['windows']['train'];test=data.manifest['windows']['test']
    source_obs=read_window(Window('source',2,train['start'],train['end'],str(processed/'flows/train.csv')),hosts,tmp_path/'cache')
    target=read_window(Window('held',4,test['start'],test['end'],str(processed/'flows/test.csv')),hosts,tmp_path/'cache')
    target.events[:]=[400 if h in data.evaluable('test') else 0 for h in hosts]
    class Adapter:
        def predict(self,*args): return frame.to_numpy().copy()
    config={'references':{'held':str(reference/'test')},'probe_reference':str(reference/'validation'),
            'neighbor_ceilings':{'held':0.5}}
    result=evaluate_candidate(frame,frame.to_numpy(),frame.to_numpy(),contacts(source_obs),target,
        config,tmp_path/'results',Adapter(),{'variant':'ridge','temperature':0.05})
    assert result['full_score_available']
    values=result['four_configurations']
    assert set(values)=={'baseline','identity_only','neighbors_only','combined'}
    assert len({round(v['score'],10) for v in values.values()})==1
    for row in values.values():
        assert row['result']['metrics']['identity_top1']==pytest.approx(1)


def test_forecaster_does_not_train_on_other_hours_of_held_day():
    from dataclasses import replace
    obs=observations()
    extra=replace(obs[1],window=replace(obs[1].window,name='second_day3',start=360,end=380))
    rows=contacts(obs[0]);vocab=sorted({p for row in rows for p in row})
    model=ContactForecaster(rows,vocab,max_queries=8).fit([obs[0],obs[1],extra,obs[2]])
    held=[(history,predictor) for target,history,predictor in model.folds if target.window.day==3]
    assert len(held)==2
    assert all(history.end==obs[0].window.end and predictor is None for history,predictor in held)
