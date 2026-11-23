import os
import json
import numpy as np
import pandas as pd
import pytest
from experiments.similarity_graph.data import Window,read_window
from experiments.similarity_graph.matching import GraphMatcher
from experiments.similarity_graph.test_matching import observation
from experiments.temporal_graph_mixer.common.evaluate import geometry


def test_count_cache_invalidates_when_content_changes_with_same_stat(tmp_path):
    path=tmp_path/'flows.csv'
    rows=pd.DataFrame([['a','b','6','80',1,1,0,0,1,0]],
        columns=['src','dst','proto','dst_port','bytes','packets','resp_bytes','resp_packets','ts','duration'])
    rows.to_csv(path,index=False);stat=path.stat();w=Window('source',2,0,10,str(path))
    first=read_window(w,['a','b'],tmp_path/'cache')
    rows['bytes']=9;rows.to_csv(path,index=False)
    assert path.stat().st_size==stat.st_size
    os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    second=read_window(w,['a','b'],tmp_path/'cache')
    fresh=read_window(w,['a','b'],tmp_path/'fresh')
    assert not np.array_equal(first.numeric,second.numeric)
    np.testing.assert_array_equal(second.numeric,fresh.numeric)


def test_refit_matcher_rebuilds_partner_vocabulary_and_profiles():
    hosts=[f'H{i}' for i in range(24)]
    old=[observation(d,[{f'p:o:old{i}':300} for i in range(24)])for d in (2,3)]
    new=[observation(d,[{f'p:o:new{23-i}':300} for i in range(24)])for d in (2,3)]
    model=GraphMatcher().fit(old,hosts)
    old_vocab=model.vectorizer.get_feature_names_out().tolist()
    model.fit(new,hosts);fresh=GraphMatcher().fit(new,hosts)
    assert old_vocab!=model.vectorizer.get_feature_names_out().tolist()
    a,b=model.unary(new[0]);x,y=fresh.unary(new[0])
    np.testing.assert_array_equal(a,x);np.testing.assert_allclose(b,y,atol=1e-12)


def test_geometry_cache_respects_host_order(reference,processed,tmp_path):
    from pipeline.data import ProcessedData
    import shutil
    hosts=ProcessedData(processed).cohort
    source=pd.DataFrame(np.random.default_rng(17).normal(size=(len(hosts),8)),index=hosts)
    ref=tmp_path/'reference';shutil.copytree(reference/'test',ref)
    meta=json.loads((ref/'manifest.json').read_text());meta['neighbor_ceiling']=.2
    (ref/'manifest.json').write_text(json.dumps(meta))
    first=geometry(source,ref,reference/'validation',tmp_path/'cache')
    changed=source.copy();changed.index=source.index[::-1]
    reused=geometry(changed,ref,reference/'validation',tmp_path/'cache')
    fresh=geometry(changed,ref,reference/'validation',tmp_path/'fresh')
    assert reused['metrics']==pytest.approx(fresh['metrics'])
    assert first['metrics']['pair_roc_auc']!=fresh['metrics']['pair_roc_auc']
