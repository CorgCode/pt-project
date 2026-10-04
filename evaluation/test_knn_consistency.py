import numpy as np
import pandas as pd
import pytest

from knn_consistency import (build_reference_features, evaluate_embedding,
                             neighborhood_metrics, top_k_neighbors)
from demo_knn_consistency import make_flows


def test_self_excluded_even_with_duplicates():
    X = np.zeros((6, 3))                      # all identical -> self may not be first
    nn = top_k_neighbors(X, 3, metric="euclidean")
    assert nn.shape == (6, 3)
    assert all(i not in nn[i] for i in range(6))


def test_k_validation():
    with pytest.raises(ValueError):
        top_k_neighbors(np.random.rand(5, 2), 5)


def test_identical_neighbourhoods_score_one():
    nn = top_k_neighbors(np.random.default_rng(0).normal(size=(30, 4)), 5)
    m = neighborhood_metrics(nn, nn)
    assert np.allclose(m.values, 1.0)


def test_disjoint_neighbourhoods_score_zero():
    ref = np.array([[1, 2], [2, 3], [3, 0], [0, 1]])
    emb = np.array([[3, 3], [0, 0], [1, 1], [2, 2]])
    m = neighborhood_metrics(ref, emb)
    assert np.allclose(m["consistency"], 0) and np.allclose(m["jaccard"], 0)


def test_ndcg_is_rank_aware():
    ref = np.array([[1, 2, 3]])
    same_set_good_order = neighborhood_metrics(ref, np.array([[1, 2, 3]]))
    same_set_bad_order = neighborhood_metrics(ref, np.array([[3, 2, 1]]))
    assert same_set_good_order.consistency[0] == same_set_bad_order.consistency[0] == 1
    assert same_set_good_order.ndcg[0] > same_set_bad_order.ndcg[0]


def test_good_beats_random():
    flows, roles = make_flows(n_hosts=150, flows_per_host=120)
    ref = build_reference_features(flows, hosts=list(roles.index))
    rng = np.random.default_rng(1)
    onehot = np.eye(roles.max() + 1)[roles.values]
    good = pd.DataFrame(onehot + rng.normal(0, .3, onehot.shape), index=roles.index)
    rand = pd.DataFrame(rng.normal(size=(len(roles), 8)), index=roles.index)
    g = evaluate_embedding(good, ref)
    r = evaluate_embedding(rand, ref)
    for k in (5, 10, 20):
        for metric in ("consistency", "jaccard", "ndcg"):
            assert g.loc[(k, metric), "mean"] > r.loc[(k, metric), "mean"]
