import numpy as np
import pandas as pd
import pytest
from .run import summarize,neighbor_analysis


def test_summarize_counts_cohort_and_excludes_self(tmp_path):
    p=tmp_path/'window.csv'
    pd.DataFrame([
      dict(src='a',dst='b',bytes=10,packets=2,ts=0,duration=1,proto='6',dst_port='443'),
      dict(src='a',dst='a',bytes=10,packets=2,ts=1,duration=0,proto='6',dst_port='443'),
      dict(src='x',dst='a',bytes=5,packets=1,ts=2,duration=1,proto='6',dst_port='80'),
    ]).to_csv(p,index=False)
    r=summarize(p,pd.Index(['a','b']),0,60)
    assert r['rows']==3 and r['self_flows']==1
    assert r['out'].tolist()==[1,0] and r['in'].tolist()==[1,1]
    assert sum(r['minute_flow_counts'])==2 and r['boundary_mismatches']==0


def test_nearest_activity_and_overlap_use_nonself_hosts():
    z=np.array([[1.,0.],[.9,.1],[0.,1.]])
    r=neighbor_analysis(z,np.array([300,301,1000]),[{'a','b'},{'a','b'},{'c'}])
    assert r['n']==3
    assert r['source_partner_jaccard_nearest_mean']==pytest.approx(2/3)
    assert r['source_partner_jaccard_random_mean']>=0
