import json

import numpy as np
import pytest
from scipy.stats import chisquare

from benchmark import methods
from benchmark.methods.conftest_helpers import TrainOnlyGuard
from benchmark.methods.graph_input import from_edges
from benchmark.methods.graphsage import (Adam, Batch, Features, GraphSAGE, infer, loss_and_grads, sample_neighbors,
                                         structural_features, train)
from experiments.run_experiment import run_experiment
from experiments.tune_methods import tune


def ring_chords(n=12):
    e = np.array([(i, (i + 1) % n) for i in range(n)] + [(i, (i + 5) % n) for i in range(n)])
    return from_edges(e[:, 0], e[:, 1], np.ones(len(e)), np.array([f"h{i}" for i in range(n)], dtype=object))


# ------------------------------------------------------------------ градиенты (ручной обратный проход)
def tiny_batch(rng, g, S2=2, S1=3):
    targets = np.array([0, 2, 5, 7, 9])
    nb2 = sample_neighbors(g, targets, S2, rng)
    nb1 = sample_neighbors(g, np.concatenate([targets, nb2.ravel()]), S1, rng)
    return Batch(targets, nb2, nb1, np.array([0, 1, 1, 3, 3]), np.array([2, 2, 4, 0, 1]), np.array([3, 4, 3]))   # повторяющиеся индексы


def numerical(f, W, eps=1e-6):
    out = np.zeros_like(W)
    for idx in np.ndindex(*W.shape):
        old = W[idx]
        W[idx] = old + eps; hi = f()
        W[idx] = old - eps; lo = f()
        W[idx] = old
        out[idx] = (hi - lo) / (2 * eps)
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_backward_pass_matches_finite_differences(seed):
    rng = np.random.default_rng(seed)
    g = ring_chords()
    X = rng.normal(size=(g.n, 3))
    W1, W2 = rng.normal(size=(6, 4)) * 0.6, rng.normal(size=(8, 3)) * 0.6
    b = tiny_batch(rng, g)
    loss, (g1, g2) = loss_and_grads(W1, W2, X, b)
    f = lambda: loss_and_grads(W1, W2, X, b, False)[0]
    assert loss == pytest.approx(f())
    assert np.allclose(g1, numerical(f, W1), atol=1e-7)
    assert np.allclose(g2, numerical(f, W2), atol=1e-7)


def test_loss_at_zero_scores_is_the_chance_value():
    """При почти ортогональных нормированных эмбеддингах потери ≈ (1 + Q) log 2 на пару."""
    rng = np.random.default_rng(0)
    g = ring_chords()
    X = rng.normal(size=(g.n, 3))
    W1, W2 = rng.normal(size=(6, 4)), rng.normal(size=(8, 3))
    b = tiny_batch(rng, g)
    loss = loss_and_grads(W1, W2, X, b, False)[0]
    assert 0.3 * (1 + 3) < loss < 1.31 * (1 + 3)                      # между лучшим (s=+/-1) и худшим случаями


# ------------------------------------------------------------------ выбор соседей
def test_sampled_neighbours_are_real_uniform_and_drawn_with_replacement():
    g = ring_chords()
    rng = np.random.default_rng(0)
    nodes = np.arange(g.n)
    s = sample_neighbors(g, nodes, 6, rng)                            # степень 4, значит повторы неизбежны
    assert g.has_edge(np.repeat(nodes, 6), s.ravel()).all()
    big = sample_neighbors(g, np.zeros(20000, dtype=np.int64), 1, rng).ravel()
    nb = g.neighbors(0)
    obs = np.array([(big == x).sum() for x in nb])
    assert chisquare(obs, np.full(len(nb), 20000 / len(nb))).pvalue > 1e-3


# ------------------------------------------------------------------ применение
def dense_infer(W1, W2, A, X):
    P = A / A.sum(1, keepdims=True)
    n1 = np.maximum(np.hstack([X, P @ X]) @ W1, 0.0)
    n1 = n1 / (np.linalg.norm(n1, axis=1, keepdims=True) + 1e-12)
    z = np.hstack([n1, P @ n1]) @ W2
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)


def test_inference_equals_a_dense_reference_and_is_deterministic():
    rng = np.random.default_rng(0)
    g = ring_chords()
    X = rng.normal(size=(g.n, 3))
    W1, W2 = rng.normal(size=(6, 4)), rng.normal(size=(8, 3))
    A = (g.adjacency().toarray() > 0).astype(float)
    assert np.allclose(infer(W1, W2, g, X), dense_infer(W1, W2, A, X))
    assert np.array_equal(infer(W1, W2, g, X), infer(W1, W2, g, X))


def test_inference_is_equivariant_to_relabelling_and_inductive():
    rng = np.random.default_rng(1)
    g = ring_chords()
    X = rng.normal(size=(g.n, 3))
    W1, W2 = rng.normal(size=(6, 4)), rng.normal(size=(8, 3))
    perm = rng.permutation(g.n)
    A = g.adjacency().toarray()
    Ap = np.zeros_like(A)
    Ap[np.ix_(perm, perm)] = A                                         # вершина i становится perm[i]
    Xp = np.zeros_like(X)
    Xp[perm] = X
    ep = dense_infer(W1, W2, (Ap > 0).astype(float), Xp)
    assert np.allclose(ep[perm], infer(W1, W2, g, X))                  # те же веса встраивают ЛЮБОЙ граф (индуктивность)
    bigger = ring_chords(30)
    assert infer(W1, W2, bigger, rng.normal(size=(30, 3))).shape == (30, 3)


# ------------------------------------------------------------------ Adam
def test_adam_first_step_is_lr_times_the_sign_and_it_minimises_a_quadratic():
    p = np.array([5.0, -3.0])
    opt = Adam([p], lr=0.1)
    opt.step([p], [np.array([2.0, -4.0])])
    assert np.allclose(p, [5.0 - 0.1, -3.0 + 0.1], atol=1e-6)         # первый шаг с поправкой смещения = lr * g / |g|
    q = np.array([4.0, -2.0])
    o2 = Adam([q], lr=0.2)
    for _ in range(300):
        o2.step([q], [2 * q])
    assert np.abs(q).max() < 0.05


# ------------------------------------------------------------------ обучение
def planted(seed, h=40):
    rng = np.random.default_rng(seed)
    n = 2 * h
    s, d = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < (0.25 if (i < h) == (j < h) else 0.02):
                s.append(i); d.append(j)
    return from_edges(np.array(s), np.array(d), np.ones(len(s)), np.array([f"h{i}" for i in range(n)], dtype=object)), h


def gap(E, h):
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    S = E @ E.T
    return (S[:h, :h].sum() - h + S[h:, h:].sum() - h) / (2 * h * (h - 1)) - S[:h, h:].mean()


def test_training_amplifies_a_weak_feature_signal_through_the_neighbourhood():
    """Шумные признаки сообщества: сырые и необученные эмбеддинги почти не разделяют сообщества, обученные — разделяют."""
    for seed in range(3):
        g, h = planted(seed)
        rng = np.random.default_rng(100 + seed)
        comm = np.r_[np.zeros(h), np.ones(h)]
        X = np.column_stack([comm - 0.5, rng.normal(size=(g.n, 3))]) + rng.normal(0, 1.5, (g.n, 4))
        kw = dict(hidden=16, samples=(10, 5), walk_length=5, num_walks=8, window=2, negatives=10, batch_size=128,
                  epochs=3, lr=0.01, seed=seed)
        W1, W2, hist = train(g, X, 8, **kw)
        W1u = np.random.default_rng(seed).normal(0, np.sqrt(2 / 8), (8, 16))
        W2u = np.random.default_rng(seed).normal(0, np.sqrt(1 / 32), (32, 8))
        trained, untrained, raw = gap(infer(W1, W2, g, X), h), gap(infer(W1u, W2u, g, X), h), gap(X, h)
        assert trained > untrained + 0.2 and trained > raw + 0.2, (seed, trained, untrained, raw)
        assert hist[-1] < hist[0]


def test_training_is_deterministic_and_seed_sensitive():
    g, h = planted(0, h=20)
    X = np.random.default_rng(0).normal(size=(g.n, 3))
    kw = dict(hidden=8, samples=(4, 3), walk_length=4, num_walks=2, window=2, negatives=5, batch_size=64, epochs=1, lr=0.01)
    a, b, c = train(g, X, 4, seed=1, **kw), train(g, X, 4, seed=1, **kw), train(g, X, 4, seed=2, **kw)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and not np.array_equal(a[0], c[0])


# ------------------------------------------------------------------ признаки
def test_structural_features_are_correct_on_a_known_graph():
    tri = from_edges(np.array([0, 1, 0, 2]), np.array([1, 2, 2, 3]), np.ones(4), np.array(list("abcd"), dtype=object))
    F = structural_features(tri)                                        # треугольник a,b,c и висячая d на c
    assert F[:, 0].tolist() == pytest.approx(np.log1p([2, 2, 3, 1]).tolist())
    assert F[:, 3].tolist() == pytest.approx([1.0, 1.0, 1 / 3, 0.0])     # коэффициент кластеризации


def test_features_validate_and_standardise_with_train_statistics(processed):
    from pipeline.data import ProcessedData
    from benchmark.methods.graph_input import from_flows
    d = ProcessedData(processed)
    with pytest.raises(ValueError, match="features должен быть"):
        Features("bogus")
    for kind, width in (("degree", 1), ("structural", 4), ("behavioral", None)):
        f = Features(kind)
        tr_flows = d.training_flows()
        g = from_flows(tr_flows)
        X = f.fit_transform(tr_flows, g)
        assert np.allclose(X.mean(0), 0, atol=1e-8) and np.isfinite(X).all()
        if width:
            assert X.shape == (g.n, width)
        te_flows = d.flows("test")
        gt = from_flows(te_flows)
        Xt = f.transform(te_flows, gt)
        assert Xt.shape[1] == X.shape[1] and np.isfinite(Xt).all()
        assert not np.allclose(Xt.mean(0), 0, atol=1e-8)                 # на позднем окне НЕ перенормируется


# ------------------------------------------------------------------ метод бенчмарка
@pytest.fixture(scope="module")
def data(processed):
    return TrainOnlyGuard(processed)


FAST = GraphSAGE.fast_params


def test_registered_validated_and_declares_shared_coordinates():
    assert "graphsage" in methods.available()
    m = GraphSAGE(dim=4)
    assert m.shared_coordinate_system is True and "Hamilton" in m.reference and m.overlap["pairs"].startswith("strong")
    for bad in ({"samples": [5]}, {"samples": [5, 0]}, {"features": "bogus"}):
        with pytest.raises(ValueError):
            GraphSAGE(dim=4, **bad)
    with pytest.raises(ValueError, match="неизвестные параметры"):
        GraphSAGE(dim=4, bogus=1)
    with pytest.raises(RuntimeError, match="fit"):
        GraphSAGE(dim=4).embed(object(), "train")


def test_output_contract_and_determinism(data):
    mk = lambda s: GraphSAGE(dim=5, seed=s, **FAST).fit(data).embed(data, "train")
    e = mk(1)
    assert list(e.index) == data.cohort and e.shape == (len(data.cohort), 5)
    assert np.isfinite(e.to_numpy()).all() and np.allclose(np.linalg.norm(e.to_numpy(), axis=1), 1.0)   # единичные векторы
    assert np.array_equal(e.to_numpy(), mk(1).to_numpy()) and not np.array_equal(e.to_numpy(), mk(2).to_numpy())


def test_later_windows_use_the_same_weights(data):
    m = GraphSAGE(dim=4, seed=0, **FAST).fit(data)
    w1, w2 = m.W1.copy(), m.W2.copy()
    hosts = data.evaluable("test")
    e = m.embed(data, "test", hosts)
    assert list(e.index) == hosts and np.array_equal(w1, m.W1) and np.array_equal(w2, m.W2)     # применение никогда не обучает
    assert np.array_equal(e.to_numpy(), m.embed(data, "test", hosts).to_numpy())                  # кэшируется, детерминированно


def test_fit_reads_only_the_train_window(processed):
    d = TrainOnlyGuard(processed)
    d.fitting = True
    try:
        GraphSAGE(dim=4, **FAST).fit(d)
    finally:
        d.fitting = False


def test_scored_embedding_is_independent_of_later_windows(processed):
    from pipeline.data import ProcessedData
    a_data, b_data = ProcessedData(processed), ProcessedData(processed)
    b_data.flows = lambda w, orig=b_data.flows: orig(w).assign(bytes=lambda f: f["bytes"] * 5) if w != "train" else orig(w)
    ea = GraphSAGE(dim=4, seed=0, **FAST).fit(a_data).embed(a_data, "train")
    eb = GraphSAGE(dim=4, seed=0, **FAST).fit(b_data).embed(b_data, "train")
    assert np.array_equal(ea.to_numpy(), eb.to_numpy())


def test_runs_through_the_runner_with_shared_coordinates_and_the_tuner(processed, reference, tmp_path):
    m = run_experiment(processed, reference, tmp_path / "o", dim=6, n_boot=0, do_audit=False, methods=["graphsage"],
                       method_config={"graphsage": FAST})
    assert "graphsage" in m["methods"] and "graphsage" not in m["missing_expected_methods"]
    assert m["graph_methods"]["graphsage"]["shared_coordinate_system"] is True
    ident = json.loads((tmp_path / "o" / "identity" / "test" / "graphsage.json").read_text())
    assert ident["shared_coordinate_system_declared"] is True and "top1" in ident["none"]       # сырые координаты осмысленны
    res = json.loads((tmp_path / "o" / "results" / "test" / "graphsage.json").read_text())
    assert res["metric_status"]["pairwise_roc_auc"] == "ok"
    out = tune(processed, reference, "graphsage", {"hidden": [8, 16], "lr": [0.01, 0.02]}, dim=4, metric="pair_roc_auc", fast=True)
    assert out["n_configurations"] == 4 and out["test_reference_read"] is False
