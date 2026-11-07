import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .streaming_links import fit_logistic,predict,score_learned
from .rotation import apply_rotation


def test_streamed_logistic_matches_sklearn_c1_objective():
    rng=np.random.default_rng(17);z=rng.normal(size=(40,8));a=rng.integers(40,size=300);b=rng.integers(40,size=300)
    y=(rng.random(300)<.5).astype(int)
    x=np.hstack([np.abs(z[a]-z[b]),z[a]*z[b]])
    actual=LogisticRegression(C=1,max_iter=2000,random_state=0).fit(x,y).predict_proba(x)[:,1]
    w,info=fit_logistic(z,a,b,y,batch=37)
    np.testing.assert_allclose(predict(z,a,b,w,batch=43),actual,atol=1e-8,rtol=1e-7)
    assert info['converged']
    mapped,_=fit_logistic(z,a,b,y,materialize_threshold=0)
    np.testing.assert_allclose(predict(z,a,b,mapped),actual,atol=1e-10)


def test_streamed_links_agree_with_project_full_implementation(reference,processed,tmp_path):
    from pipeline.data import ProcessedData
    from evaluation.future_links import score_learned as official,load_candidate_edges
    hosts=ProcessedData(processed).cohort
    frame=pd.DataFrame(np.random.default_rng(5).normal(size=(len(hosts),8)),index=hosts)
    fit=load_candidate_edges(reference/'validation/edges.csv');ev=load_candidate_edges(reference/'test/edges.csv')
    a=official(frame,fit,ev);b=score_learned(frame,fit,ev,cache=tmp_path)
    assert a['roc_auc']==pytest.approx(b['roc_auc'],abs=1e-8)
    assert a['random_roc_auc']==pytest.approx(b['random_roc_auc'],abs=1e-8)


def test_orthogonal_block_rotation_preserves_all_cosines():
    rng=np.random.default_rng(3);x=rng.normal(size=(30,20));q,_=np.linalg.qr(rng.normal(size=(8,8)))
    y=apply_rotation(x,{'indices':np.arange(8),'matrix':q})
    np.testing.assert_allclose(x@x.T,y@y.T,atol=1e-12)


def test_all_five_metrics_and_score_match_project_evaluator(reference,processed,tmp_path):
    import json,shutil
    from pipeline.data import ProcessedData
    from evaluation.evaluate import run as official
    from evaluation import submit,scoreboard
    from .evaluate import geometry,with_identity,REQUIRED
    hosts=ProcessedData(processed).cohort
    rng=np.random.default_rng(13)
    source=pd.DataFrame(rng.normal(size=(len(hosts),8)),index=pd.Index(hosts,name='host_id')).add_prefix('emb_')
    later=source+pd.DataFrame(rng.normal(scale=.1,size=source.shape),index=source.index,columns=source.columns)
    ref=tmp_path/'ref';shutil.copytree(reference/'test',ref)
    manifest=json.loads((ref/'manifest.json').read_text());manifest['neighbor_ceiling']=.2
    (ref/'manifest.json').write_text(json.dumps(manifest))
    core=geometry(source,ref,reference/'validation',tmp_path/'cache')
    result=with_identity(source,later,ref,core)
    a,b=tmp_path/'source.csv',tmp_path/'late.csv';source.to_csv(a);later.to_csv(b)
    expected=official(a,ref,'test',fit_reference_dir=reference/'validation')
    submit.add_identity(expected,a,b,ref,shared=True,n_boot=0)
    for key in REQUIRED:assert result['metrics'][key]==pytest.approx(expected['metrics'][key],abs=1e-8)
    old=scoreboard.NEIGHBORS_IDEAL.copy()
    try:
        scoreboard.NEIGHBORS_IDEAL['unit_sg']=.2
        assert result['score']==pytest.approx(scoreboard._one_score(expected,REQUIRED,window='unit_sg'),abs=1e-8)
    finally:
        scoreboard.NEIGHBORS_IDEAL.clear();scoreboard.NEIGHBORS_IDEAL.update(old)


def test_pipeline_reference_schema_is_supported_without_reusing_day02_ceiling(reference,processed,tmp_path):
    import json
    from pipeline.data import ProcessedData
    from .data import Window,read_window
    from .reference import resolve
    data=ProcessedData(processed);hosts=data.cohort
    def obs(name):
        span=data.manifest['windows'][name]
        return read_window(Window(name,2,span['start'],span['end'],str(processed/'flows'/f'{name}.csv')),hosts,tmp_path/'cache')
    target=obs('test')
    # The small project fixture uses >=5 events, while PT-05 matching uses >=300.
    # Translate activity units only for this manifest-schema test; actual flow counts stay intact.
    target.events=np.where(target.events>=data.manifest['config']['min_events_per_window'],300,0)
    result=resolve(reference/'test',obs('train'),target,hosts,tmp_path/'normalized',{'cache':str(tmp_path/'cache')})
    meta=json.loads((result/'manifest.json').read_text())
    assert meta['calibration']=='local persistence, custom cohort/window'
    assert 0<meta['neighbor_ceiling']<=1
    assert meta['inputs']['source']['start']==data.manifest['windows']['train']['start']
