import numpy as np
import pandas as pd
import scipy.sparse as sp

from experiments.fingerprint_localports import local_counts,retain_frequent


def test_local_ports_keep_anonymous_source_values_and_ignore_self_flows():
    frame=pd.DataFrame({'src':['a','a','b'],'dst':['b','a','a'],
        'proto':['6']*3,'src_port':['Port123','Port777','Port456'],'dst_port':['443','443','80']})
    a=local_counts(frame,['a','b'])
    assert a.sum()==4
    changed=frame.copy();changed.loc[0,'src_port']='Port999'
    assert (local_counts(changed,['a','b'])!=a).nnz>0
    swapped=local_counts(frame,['b','a'])
    np.testing.assert_array_equal(swapped.toarray(),a[[1,0]].toarray())


def test_frequent_filter_removes_ephemeral_ports_and_preserves_top_tokens():
    a=sp.csr_matrix([[1.,1.,20,30,40],[0,1,1,1,1]])
    out=retain_frequent(a,minimum=10,fraction=0.,topk=2)
    np.testing.assert_array_equal(out.toarray(),[[0,0,0,30,40],[0,0,0,0,0]])
