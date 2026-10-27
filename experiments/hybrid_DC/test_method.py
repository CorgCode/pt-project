import joblib
import json
import pickle
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from experiments.hybrid_DC.method import LeaderGraphHybrid, LinearNeighborhood


def vectors(seed=0, hosts=20):
    rng = np.random.default_rng(seed)
    index = pd.Index([f"h{i}" for i in range(hosts)], name="host_id")
    return (pd.DataFrame(rng.normal(size=(hosts, 5)), index=index, columns=[f"emb_{i}" for i in range(5)]),
            pd.DataFrame(rng.normal(size=(hosts, 3)), index=index, columns=[f"emb_{i}" for i in range(3)]))


def test_zero_weight_is_exact_baseline_even_without_graph():
    train, _ = vectors()
    later, _ = vectors(1)
    model = LeaderGraphHybrid(0).fit(train)
    pd.testing.assert_frame_equal(model.transform(later), later, check_exact=True)


def test_host_alignment_and_column_order_do_not_change_result():
    a, b = vectors()
    model = LeaderGraphHybrid(.5).fit(a, b)
    expected = model.transform(a, b)
    shuffled = b.sample(frac=1, random_state=5).loc[:, b.columns[::-1]]
    pd.testing.assert_frame_equal(model.transform(a.loc[:, a.columns[::-1]], shuffled), expected)


def test_heldout_transform_uses_frozen_training_scales_and_projection():
    a, b = vectors()
    later_a, later_b = vectors(5)
    model = LeaderGraphHybrid(1).fit(a, b)
    mean, components = model.graph_mean_.copy(), model.pca_.components_.copy()
    result = model.transform(later_a, later_b)
    expected = model.pca_.transform(np.hstack([later_a.to_numpy()/model.leader_scale_,
                 (later_b.to_numpy()-mean)/model.graph_scale_]))
    np.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(model.graph_mean_, mean)
    np.testing.assert_array_equal(model.pca_.components_, components)


def test_new_host_and_serialized_model_work(tmp_path):
    a, b = vectors()
    model = LeaderGraphHybrid(.25).fit(a, b)
    later_a, later_b = vectors(1, hosts=3)
    later_a.index = later_b.index = ["unseen-a", "unseen-b", "unseen-c"]
    path = tmp_path / "fusion.joblib"
    joblib.dump(model, path)
    pd.testing.assert_frame_equal(joblib.load(path).transform(later_a, later_b), model.transform(later_a, later_b))


@pytest.mark.parametrize("weight", [-1, np.nan, np.inf])
def test_invalid_weights(weight):
    with pytest.raises(ValueError):
        LeaderGraphHybrid(weight)


def test_missing_hosts_are_an_error():
    a, b = vectors()
    with pytest.raises(ValueError, match="нет 1 хостов"):
        LeaderGraphHybrid(1).fit(a, b.iloc[1:])


def test_duplicate_ids_are_an_error():
    a, b = vectors()
    b.index = ["duplicate"]*len(b)
    with pytest.raises(ValueError, match="повторяются"):
        LeaderGraphHybrid(1).fit(a, b)


def test_graph_aggregation_uses_neighbors_and_handles_isolated_host():
    x = np.array([[1., 0.], [0., 1.], [2., 2.]])
    c = sp.csr_matrix(([3.], ([0], [1])), shape=(3, 3))
    expected = [[1, 0, 0, .5], [0, 1, .5, 0], [2, 2, 0, 0]]
    np.testing.assert_allclose(LinearNeighborhood.aggregate(x, c), expected)


def test_graph_projection_is_inductive_and_does_not_depend_on_indices():
    x = np.random.default_rng(4).normal(size=(20, 6)).astype(np.float32)
    c = sp.eye(20, k=1, format="csr", dtype=np.float32)
    model = LinearNeighborhood(dim=4).fit(x, c, np.arange(15))
    perm = np.random.default_rng(1).permutation(20)
    np.testing.assert_allclose(model.transform(x[perm], c[perm][:,perm]), model.transform(x,c)[perm], atol=1e-6)


def test_empty_graph_profile_has_shared_fallback_without_host_id():
    x = np.random.default_rng(0).normal(size=(10, 4)).astype(np.float32)
    model = LinearNeighborhood(dim=3).fit(x, sp.csr_matrix((10, 10)), np.arange(10))
    result = model.transform(np.zeros((2, 4), np.float32), sp.csr_matrix((2, 2)))
    np.testing.assert_array_equal(result, [[1, 0, 0], [1, 0, 0]])


def test_selection_uses_validation_only_and_is_frozen(tmp_path, monkeypatch):
    from experiments.hybrid_DC import run
    from experiments.hybrid_DC.data import sha
    board = tmp_path / "board"
    (board / "processed").mkdir(parents=True)
    (board / "processed/manifest.json").write_text("{}")
    params = tmp_path / "params.json"
    params.write_text("{}")
    monkeypatch.setattr(run, "BOARD", board)
    monkeypatch.setattr(run, "PARAMS", params)
    work, graph_dir = tmp_path / "run", tmp_path / "graph"
    (work / "leader").mkdir(parents=True)
    a, _ = vectors()
    a.to_csv(work / "leader/embeddings.csv")
    a.to_csv(work / "leader/validation.csv")
    (graph_dir / "cache").mkdir(parents=True)
    (graph_dir / "cache/manifest.json").write_text(json.dumps({"target_manifest_sha256":sha(board / "processed/manifest.json")}))
    x = np.random.default_rng(0).normal(size=(20, 6)).astype(np.float32)
    c = sp.eye(20, k=1, format="csr", dtype=np.float32)
    graph = LinearNeighborhood().fit(x, c, np.arange(20))
    g = pd.DataFrame(graph.transform(x, c), index=a.index, columns=[f"emb_{i}" for i in range(graph.svd_.n_components)])
    for path in [graph_dir / "submissions/linear/embeddings.csv", graph_dir / "baselines_validation/linear/embeddings.csv", graph_dir / "baselines_validation/linear/validation.csv"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        g.to_csv(path)
    with (graph_dir / "features.pkl").open("wb") as stream:
        pickle.dump({"train":{"x":x,"counts":c},"validation":{"x":x,"counts":c},
                     "cohort":np.arange(20),"cohort_names":a.index.tolist()}, stream)
    calls=[]
    def score(folder, window):
        assert not (work / "leader/test.csv").exists()
        calls.append(window)
        return 2.-int(folder.name[-2:]), {"summary":[],"metrics":{}}
    monkeypatch.setattr(run, "metric_score", score)
    run.tune(work, graph_dir, [0., .5])
    selected=json.loads((work / "selection.json").read_text())
    assert selected["weight"] == 0 and selected["test_read"] is False
    assert calls == ["validation","validation"]
    monkeypatch.setattr(run, "load_embeddings", lambda *args: pytest.fail("замороженный подбор не должен читать новые данные"))
    run.tune(work, graph_dir, [0., 1.])
    assert len(calls) == 2
