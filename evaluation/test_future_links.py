"""Тесты на синтетике; про LANL ничего не утверждают."""
import numpy as np
import pandas as pd
import pytest

from evaluation.future_links import build_edges, load_candidate_edges, score_embeddings


def fixture(n=12):
    hosts = [f"h{i:02}" for i in range(n)]
    def make(edges, t):
        return pd.DataFrame([{
            "src": a, "dst": b, "dst_port": "443", "proto": "tcp",
            "bytes": 100, "packets": 4, "ts": pd.Timestamp(t)
        } for a,b in edges])
    source = make([(hosts[i], hosts[(i+1)%n]) for i in range(n)],
                  "2026-10-01T09:00:00")
    target = make([(hosts[i], hosts[(i+2)%n]) for i in range(n)],
                  "2026-10-01T10:00:00")
    return source, target, hosts


def test_fixed_candidates_and_scoring(tmp_path):
    source, target, hosts = fixture()
    pairs = build_edges(source, target, hosts, seed=42)
    again = build_edges(source, target, list(reversed(hosts)), seed=42)
    pd.testing.assert_frame_equal(pairs, again)
    assert set(pairs.label) == {0,1}
    assert pairs.label.sum() == len(pairs)//2
    assert (pairs.host_a < pairs.host_b).all()
    path = tmp_path/"edges.csv"
    pairs.to_csv(path,index=False)
    loaded = load_candidate_edges(path)
    rng=np.random.default_rng(123)
    emb=pd.DataFrame(rng.normal(size=(len(hosts),8)),index=hosts)
    scored=score_embeddings(emb,loaded)
    assert 0 <= scored["roc_auc"] <= 1
    assert 0 <= scored["pr_auc"] <= 1
    assert scored["n_examples"] == len(pairs)
    assert scored["split"]=="test"
    second=score_embeddings(emb.iloc[::-1],loaded)
    assert second["roc_auc"]==pytest.approx(scored["roc_auc"])
    assert second["pr_auc"]==pytest.approx(scored["pr_auc"])


def test_time_leakage_guard():
    source,target,hosts=fixture()
    target["ts"]=source["ts"].max()
    with pytest.raises(ValueError,match="overlap"):
        build_edges(source,target,hosts)


def test_impossible_negative_ratio():
    source,target,hosts=fixture(5)
    with pytest.raises(ValueError,match="Insufficient negative"):
        build_edges(source,target,hosts,negatives_per_positive=3)


def test_missing_hosts_and_invalid_candidates(tmp_path):
    source,target,hosts=fixture()
    pairs=build_edges(source,target,hosts)
    rng=np.random.default_rng(8)
    emb=pd.DataFrame(rng.normal(size=(len(hosts),4)),index=hosts)
    with pytest.raises(ValueError,match="lacks candidate hosts"):
        score_embeddings(emb.iloc[:-1],pairs)
    pairs.loc[pairs.index[0],"label"]=2
    path=tmp_path/"bad.csv"
    pairs.to_csv(path,index=False)
    with pytest.raises(ValueError,match="0/1"):
        load_candidate_edges(path)


def test_refuses_single_label_and_zero_norm():
    source,target,hosts=fixture()
    pairs=build_edges(source,target,hosts)
    rng=np.random.default_rng(8)
    emb=pd.DataFrame(rng.normal(size=(len(hosts),4)),index=hosts)
    all_pos=pairs[pairs.label==1].copy()
    with pytest.raises(ValueError,match="both positive and negative"):
        score_embeddings(emb,all_pos)
    emb.iloc[0]=0
    with pytest.raises(ValueError,match="Zero-norm"):
        score_embeddings(emb,pairs)


def test_refuses_mixed_window_pair():
    source,target,hosts=fixture()
    pairs=build_edges(source,target,hosts)
    pairs.loc[pairs.index[0],"target_window"]="unrelated"
    emb=pd.DataFrame(np.arange(12*3).reshape(12,3)+1.,index=hosts)
    with pytest.raises(ValueError,match="Do not mix"):
        score_embeddings(emb,pairs)
