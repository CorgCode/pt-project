from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from .data import Window, Observation, read_window, check_schedule, contacts
from .identity import IdentityAdapter
from .run import assess


def observations(n=32):
    result=[]
    for day in (2,3,4):
        rows=[{f'p:o:peer{i}':20, 's:o:6/80':200} for i in range(n)]
        numeric=np.column_stack([np.arange(n),np.full(n,day)])
        result.append(Observation(Window(f'd{day}',day,day*100,day*100+50,'unused'),rows,numeric,np.full(n,400)))
    return result


def test_gate_is_time_ordered_and_source_is_immutable():
    source=np.random.default_rng(7).normal(size=(32,8))
    before=source.copy()
    obs=observations()
    m=IdentityAdapter(source,svd_dim=12).fit(obs).fit_gate(obs)
    assert all(r['training_end']<=r['query_start'] for r in m.gate_history)
    np.testing.assert_array_equal(source,before)
    for variant in ('ridge','rare','gate','mix:0.5'):
        y=m.predict(obs[-1],variant)
        assert y.shape==source.shape and np.isfinite(y).all()
        perm=np.random.default_rng(9).permutation(len(source))
        query=replace(obs[-1],counts=[obs[-1].counts[i] for i in perm],
                      numeric=obs[-1].numeric[perm],events=obs[-1].events[perm])
        np.testing.assert_allclose(m.predict(query,variant),y[perm],atol=1e-10)
    dev=replace(obs[-1], counts=[{'p:o:never_seen':10}]*32)
    before_idf=m.idf.copy()
    before_vocab=dict(m.vocabulary.vocabulary_)
    assert np.isfinite(m.predict(dev,'rare')).all()
    np.testing.assert_array_equal(m.idf,before_idf)
    assert m.vocabulary.vocabulary_==before_vocab


def test_schedule_rejects_overlap():
    a=Window('a',2,100,200,'unused')
    with pytest.raises(ValueError,match='пересекает development'):
        check_schedule([a],[replace(a,name='b',start=150)],[])


def test_read_counts_directions_boundaries_and_no_query_id_input(tmp_path):
    path=tmp_path/'flows.csv'
    pd.DataFrame([['a','b','6','80',10,2,10,1,2,1],['b','a','6','80',19,1,4,1,2,1],
                  ['a','a','6','80',12,1,3,1,1,1]],
        columns=['src','dst','proto','dst_port','ts','duration','bytes','packets','resp_bytes','resp_packets']).to_csv(path,index=False)
    w=Window('x',2,10,20,str(path))
    obs=read_window(w,['a','b'],tmp_path/'cache',chunksize=1)
    assert contacts(obs)==[{'b':1},{'a':1}]
    assert obs.counts[0]['p:o:b']==1 and obs.counts[1]['p:i:a']==1
    assert 'p:o:a' not in obs.counts[0]
    assert obs.events.tolist()==[1,1]


def test_identity_only_changes_late_representation():
    source=pd.DataFrame(np.random.default_rng(7).normal(size=(32,8)),index=[f'h{i}' for i in range(32)])
    old=source.copy(deep=True)
    obs=observations()[0]
    result=assess(source,source.to_numpy(),obs,activity_threshold=400)
    assert result['direct_full_gallery']['top1']==1
    assert result['official']['top1']==1
    pd.testing.assert_frame_equal(source,old)
    assert result['activity']['low_activity']['gallery_size']==32
    assert result['late_to_source_full_gallery']['top1']==1


def test_oof_uses_whole_prior_days_not_earlier_hours_of_query_day():
    obs=observations()
    extra=replace(obs[1],window=replace(obs[1].window,name='second_hour_day3',start=360,end=380))
    model=IdentityAdapter(np.random.default_rng(7).normal(size=(32,8)),svd_dim=8).fit([obs[0],obs[1],extra,obs[2]])
    model.fit_gate([obs[0],obs[1],extra,obs[2]])
    rows=[r for r in model.gate_history if r['query_window'] in ('d3','second_hour_day3')]
    assert len(rows)==2 and {r['training_end'] for r in rows}=={obs[0].window.end}
