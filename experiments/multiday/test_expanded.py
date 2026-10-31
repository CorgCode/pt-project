import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from experiments.multiday.memory import fingerprint_counts
from experiments.multiday.expanded_data import (ExpandedBank, count_flows, count_series,
                                                    extract_day, token_columns, WIDTH)
from experiments.multiday.search import write_json


def sample():
    return pd.DataFrame({'src':['a','b','a','a'], 'dst':['peer1','peer2','peer1','a'],
                         'proto':['6']*4, 'dst_port':['443','80','443','443']})


def test_streamed_hashes_equal_counts_and_are_independent_of_chunking():
    flows, hosts = sample(), ['a','b']
    direct, events = count_flows(flows, hosts)
    a, ae = count_flows(flows.iloc[:2], hosts)
    b, be = count_flows(flows.iloc[2:], hosts)
    assert (direct != a+b).nnz == 0
    expected = count_series(fingerprint_counts(flows, hosts), hosts)
    assert (direct != expected).nnz == 0
    np.testing.assert_array_equal(events, ae+be)
    np.testing.assert_array_equal(events, [2,1])
    assert direct[0, token_columns(['o:a'], 'peers')[0]] == 0


def test_query_permutation_changes_rows_not_features():
    a, _ = count_flows(sample(), ['a','b'])
    b, _ = count_flows(sample(), ['b','a'])
    assert (a != b[[1,0]]).nnz == 0
    # Хеширование не зависит от соседей токена в массиве.
    assert token_columns(['new','o:peer1'], 'peers')[1] == token_columns(['o:peer1'], 'peers')[0]


def test_expanded_bank_rejects_future_and_repeated_training_days(tmp_path):
    d=tmp_path/'day02'; d.mkdir()
    write_json(d/'complete.json', {'day':2,'interval':[122400,136800]})
    with pytest.raises(ValueError, match='пересекает'):
        ExpandedBank().fit([d], ['a','b'], 130000)
    with pytest.raises(ValueError, match='повторён'):
        ExpandedBank().fit([d,d], ['a','b'], 900000)


def test_raw_extraction_respects_hours_boundaries_and_fixed_cohort(tmp_path):
    import bz2
    raw=tmp_path/'day02.bz2'
    rows=[
        '122400,0,a,peer1,6,10,443,1,0,50,0',  #10:00, valid
        '125999,2,a,cross,6,10,443,1,0,50,0',  # crosses11:00
        '126000,0,b,peer2,6,10,80,1,0,50,0',  #11:00
        '136800,0,a,future,6,10,443,1,0,50,0', #14:00 excluded
        '126100,0,a,a,6,10,443,1,0,50,0',    #self excluded
        '126200,0,outsider,peer3,6,10,443,1,0,50,0']
    raw.write_bytes(bz2.compress(('\n'.join(rows)+'\n').encode()))
    out=tmp_path/'out'
    extract_day(raw, ['a','b'], out, 2, chunksize=2)
    matrix=sp.load_npz(out/'counts.npz')
    assert matrix.shape == (8, WIDTH)
    events=np.load(out/'events.npy')
    np.testing.assert_array_equal(events, [1,0,0,1,0,0,0,0])
    assert matrix[:,token_columns(['o:future'],'peers')[0]].nnz == 0
    assert (out/'complete.json').exists()


def test_bank_fit_uses_historical_views_and_keeps_inference_transform_frozen(tmp_path):
    hosts=['a','b']
    flows=pd.concat([sample()]*30,ignore_index=True)
    x,events=count_flows(flows,hosts)
    directories=[]
    for day in [2,3]:
        out=tmp_path/f'day{day}'; out.mkdir(); directories.append(out)
        sp.save_npz(out/'counts.npz',sp.vstack([x]*4,format='csr'))
        np.save(out/'events.npy',np.tile(events,4))
        write_json(out/'complete.json',{'day':day,'interval':[(day-1)*86400+36000,(day-1)*86400+50400]})
    bank=ExpandedBank().fit(directories,hosts,9*86400+45000)
    assert bank.memory.shape==(16,WIDTH)
    original=bank.tfidf.idf_.copy()
    transformed=bank.transform(fingerprint_counts(flows,hosts))
    assert transformed.shape==(2,WIDTH)
    np.testing.assert_allclose(np.asarray(transformed.multiply(transformed).sum(axis=1)).ravel(),1)
    future=flows.replace({'peer1':'future_peer'})
    bank.transform(fingerprint_counts(future,hosts))
    np.testing.assert_array_equal(bank.tfidf.idf_,original)
