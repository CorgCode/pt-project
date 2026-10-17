import functools
import json

import numpy as np
import pytest
from scipy.stats import chisquare

from benchmark import methods
from benchmark.methods.conftest_helpers import TrainOnlyGuard
from benchmark.methods.deepwalk import DeepWalk
from benchmark.methods.graph_input import from_edges
from benchmark.methods.struc2vec import (EMPTY_DEGREE, Multilayer, Struc2Vec, candidate_pairs, dtw, ring_sequences,
                                         structural_distances, structural_walks)
from experiments.run_experiment import run_experiment
from experiments.tune_methods import tune


def mk(edges, n=None):
    e = np.array(edges)
    n = n or int(e.max()) + 1
    return from_edges(e[:, 0], e[:, 1], np.ones(len(e)), np.array([f"h{i}" for i in range(n)], dtype=object))


# ------------------------------------------------------------------ DTW против независимого перебора
def brute_dtw(a, ca, b, cb):
    """Рекурсия из учебника с мемоизацией (независимо от итеративной реализации)."""
    cost = lambda i, j: (max(a[i], b[j]) / min(a[i], b[j]) - 1.0) * max(ca[i], cb[j])

    @functools.lru_cache(None)
    def d(i, j):
        if i == 0 and j == 0:
            return cost(0, 0)
        best = min(d(i - 1, j) if i else np.inf, d(i, j - 1) if j else np.inf, d(i - 1, j - 1) if i and j else np.inf)
        return cost(i, j) + best

    return d(len(a) - 1, len(b) - 1)


def test_dtw_matches_the_recursive_definition_on_random_sequences():
    rng = np.random.default_rng(0)
    for _ in range(60):
        n, m = rng.integers(1, 7, 2)
        a, b = np.sort(rng.choice(np.arange(1, 40), n, replace=False)).astype(float), np.sort(rng.choice(np.arange(1, 40), m, replace=False)).astype(float)
        ca, cb = rng.integers(1, 6, n).astype(float), rng.integers(1, 6, m).astype(float)
        assert dtw(a, ca, b, cb) == pytest.approx(brute_dtw(tuple(a), tuple(ca), tuple(b), tuple(cb)))


def test_dtw_known_values_and_properties():
    one = lambda v, c=1.0: (np.array([float(v)]), np.array([c]))
    assert dtw(*one(2), *one(2)) == 0.0
    assert dtw(*one(1), *one(2)) == pytest.approx(1.0)                      # 2/1 - 1
    assert dtw(*one(2), *one(8)) == pytest.approx(3.0)                      # 8/2 - 1
    assert dtw(*one(2, 3.0), *one(4, 5.0)) == pytest.approx(1.0 * 5.0)      # цена d * max(count)
    a, ca, b, cb = np.array([1.0, 3, 9]), np.array([2.0, 1, 1]), np.array([2.0, 3]), np.array([1.0, 4])
    assert dtw(a, ca, b, cb) == pytest.approx(dtw(b, cb, a, ca))            # симметрично
    assert dtw(a, ca, a, ca) == 0.0


def test_empty_rings():
    e = (np.array([]), np.array([]))
    assert dtw(*e, *e) == 0.0
    leaf = (np.array([1.0]), np.array([1.0]))
    assert dtw(*e, *leaf) == pytest.approx(1.0 / EMPTY_DEGREE - 1.0)         # непустое кольцо никогда не на расстоянии 0
    assert dtw(*e, *leaf) == dtw(*leaf, *e) > 0


# ------------------------------------------------------------------ кольца
def test_ring_degree_sequences_on_a_path_and_a_star():
    path = mk([(0, 1), (1, 2), (2, 3), (3, 4)])
    s = ring_sequences(path, np.array([2]), 2)
    assert s[0][0][0].tolist() == [2.0] and s[0][0][1].tolist() == [1.0]            # кольцо 0 — сама вершина (степень 2)
    assert s[1][0][0].tolist() == [2.0] and s[1][0][1].tolist() == [2.0]            # кольцо 1 = {1,3}, обе степени 2
    assert s[2][0][0].tolist() == [1.0] and s[2][0][1].tolist() == [2.0]            # кольцо 2 = {0,4}, обе листья
    star = mk([(0, i) for i in range(1, 6)])
    s = ring_sequences(star, np.array([1]), 2)
    assert s[1][0][0].tolist() == [5.0] and s[2][0][0].tolist() == [1.0] and s[2][0][1].tolist() == [4.0]


def test_rings_beyond_the_eccentricity_are_empty_and_the_visited_state_is_reset():
    g = mk([(0, 1), (1, 2)])
    seqs = ring_sequences(g, np.array([0, 1, 2]), 3)
    assert len(seqs[3][0][0]) == 0                                                  # на расстоянии 3 от вершины 0 никого
    again = ring_sequences(g, np.array([0]), 2)
    assert again[1][0][0].tolist() == seqs[1][0][0].tolist()                        # состояние не протекает между вершинами


# ------------------------------------------------------------------ структурное расстояние
def two_copies():
    base = [(0, 1), (0, 2), (0, 3), (3, 4), (3, 5), (4, 5), (5, 6)]
    return mk(base + [(a + 7, b + 7) for a, b in base], n=14)


def test_isomorphic_neighbourhoods_have_distance_zero_at_every_layer():
    g = two_copies()
    seqs = ring_sequences(g, np.arange(14), 3)
    pairs = np.array([[i, i + 7] for i in range(7)])                                 # одна и та же позиция в обеих копиях
    assert np.allclose(structural_distances(seqs, pairs), 0.0)
    other = np.array([[0, 1], [0, 6], [3, 6]])
    assert (structural_distances(seqs, other)[-1] > 0).all()                         # разные позиции различаются


def test_distances_are_cumulative_nonnegative_and_start_with_the_degree_ratio():
    g = two_copies()
    seqs = ring_sequences(g, np.arange(14), 3)
    pairs = candidate_pairs(g.degree, 4)
    f = structural_distances(seqs, pairs)
    assert (f >= 0).all() and (np.diff(f, axis=0) >= -1e-12).all()                   # f_k не убывает по k
    dg = g.degree.astype(float)
    ratio = np.maximum(dg[pairs[:, 0]], dg[pairs[:, 1]]) / np.minimum(dg[pairs[:, 0]], dg[pairs[:, 1]]) - 1
    assert np.allclose(f[0], ratio)                                                  # f_0 = отношение степеней - 1


def test_memoisation_does_not_change_the_result():
    g = two_copies()
    seqs = ring_sequences(g, np.arange(14), 2)
    pairs = candidate_pairs(g.degree, 6)
    f = structural_distances(seqs, pairs)
    ref = np.zeros(len(pairs))
    for k in range(3):                                                               # прямое накопление без кэша
        ref += [dtw(*seqs[k][i], *seqs[k][j]) for i, j in pairs]
        assert np.allclose(f[k], ref)


def test_candidate_pairs_are_close_in_degree_order_unique_and_cover_every_node():
    deg = np.array([1, 1, 1, 2, 2, 3, 5, 9, 9, 40])
    pairs = candidate_pairs(deg, 2)
    assert (pairs[:, 0] < pairs[:, 1]).all() and len({tuple(p) for p in pairs}) == len(pairs)
    order = np.lexsort((np.arange(10), deg))
    rank = np.empty(10, dtype=int)
    rank[order] = np.arange(10)
    assert (np.abs(rank[pairs[:, 0]] - rank[pairs[:, 1]]) <= 2).all()
    assert set(pairs.ravel()) == set(range(10))


# ------------------------------------------------------------------ многослойный граф и блуждания
def ring_pairs():
    pairs = candidate_pairs(np.array([1, 1, 2, 2, 3, 3]), 2)
    f = np.vstack([np.full(len(pairs), 0.5), np.full(len(pairs), 1.5)])               # K = 1: два слоя
    return pairs, f


def test_multilayer_weights_and_layer_change_probabilities():
    pairs, f = ring_pairs()
    ml = Multilayer(6, pairs, f)
    A = ml.indices
    assert np.allclose(ml.weights[ml.indptr[0]:ml.indptr[1]], np.exp(-0.5))                       # веса слоя 0
    assert np.allclose(ml.weights[ml.indptr[6]:ml.indptr[7]], np.exp(-1.5))                       # веса слоя 1
    # все веса равны -> никто не выше среднего -> Gamma = 0 -> w_up = log(e) = 1 -> p_up = 1/2
    assert np.allclose(ml.gamma, 0) and np.allclose(ml.up_weight, 1.0) and np.allclose(ml.p_up, 0.5)
    assert A.max() < 12 and (A[ml.indptr[6]:ml.indptr[12]] >= 6).all()                            # рёбра слоя 1 остаются в слое 1


def test_gamma_counts_edges_heavier_than_the_layer_average():
    pairs = np.array([[0, 1], [0, 2], [1, 2], [2, 3]])
    f = np.array([[0.0, 0.0, 5.0, 5.0]])                                                          # два тяжёлых, два лёгких ребра
    ml = Multilayer(4, pairs, f)
    assert ml.gamma[0].tolist() == [2, 1, 1, 0]                                                   # тяжёлые: (0,1) и (0,2)
    assert ml.up_weight[0, 0] == pytest.approx(np.log(2 + np.e)) and ml.p_up[0, 0] == pytest.approx(np.log(2 + np.e) / (np.log(2 + np.e) + 1))


def test_walk_statistics_stay_probability_and_layer_moves():
    pairs, f = ring_pairs()
    ml = Multilayer(6, pairs, f)
    walks = structural_walks(ml, 400, 5, stay_prob=0.5, seed=0)
    assert walks.shape == (6 * 400, 5) and walks.min() >= 0 and walks.max() < 6
    # при stay_prob = 0 блуждание чередует слои 0,1,0,1 и не ходит по рёбрам слоя 0, поэтому соседние вершины
    # не обязаны быть соседями; при stay_prob = 1 слой 0 не покидается и каждый шаг идёт по его ребру
    allowed = {(int(a), int(b)) for a, b in pairs} | {(int(b), int(a)) for a, b in pairs}
    stay = structural_walks(ml, 50, 8, stay_prob=1.0, seed=1)
    steps = {(int(a), int(b)) for a, b in zip(stay[:, :-1].ravel(), stay[:, 1:].ravel())}
    assert steps <= allowed
    move = structural_walks(ml, 50, 8, stay_prob=0.0, seed=1)
    assert (move[:, 1:] == move[:, :-1]).all()                                                    # смена слоя не меняет вершину


def test_stay_probability_is_respected():
    pairs, f = ring_pairs()
    ml = Multilayer(6, pairs, f)
    w = structural_walks(ml, 3000, 2, stay_prob=0.3, seed=2)
    moved_layer = (w[:, 1] == w[:, 0])                  # смена слоя сохраняет вершину, шаг в слое — меняет (петель нет)
    assert abs(moved_layer.mean() - 0.7) < 0.03


def test_walks_are_deterministic():
    pairs, f = ring_pairs()
    ml = Multilayer(6, pairs, f)
    assert np.array_equal(structural_walks(ml, 5, 10, 0.3, 1), structural_walks(ml, 5, 10, 0.3, 1))
    assert not np.array_equal(structural_walks(ml, 5, 10, 0.3, 1), structural_walks(ml, 5, 10, 0.3, 2))


# ------------------------------------------------------------------ в чём смысл struc2vec
def roles_graph(K=8, L=10):
    edges, roles, comp, nid = [], [], [], 0
    for k in range(K):
        hub = nid; nid += 1; roles.append("hub"); comp.append(k)
        leaves = list(range(nid, nid + L)); nid += L; roles += ["leaf"] * L; comp += [k] * L
        edges += [(hub, l) for l in leaves]
        att = nid; nid += 1; roles.append("attached"); comp.append(k); edges.append((hub, att))
        cl = list(range(nid, nid + 3)); nid += 3; roles += ["clique"] * 3; comp += [k] * 3
        m = [att] + cl
        edges += [(m[i], m[j]) for i in range(4) for j in range(i + 1, 4)]
    return mk(edges, n=nid), np.array(roles), np.array(comp)


def role_accuracy(E, roles, comp):
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    S = E @ E.T
    S[comp[:, None] == comp[None, :]] = -9
    nn = S.argmax(1)
    return {r: float(np.mean(roles[nn[roles == r]] == r)) for r in ("hub", "attached", "leaf", "clique")}


def test_nodes_with_the_same_structural_role_match_across_disconnected_components():
    g, roles, comp = roles_graph()
    E = Struc2Vec(dim=16, seed=0, num_walks=10, walk_length=40, window=5, epochs=2).embed_graph(g)
    acc = role_accuracy(E, roles, comp)
    assert all(v >= 0.95 for v in acc.values()), acc
    dw = role_accuracy(DeepWalk(dim=16, seed=0, num_walks=8, walk_length=30, window=5, epochs=2).embed_graph(g), roles, comp)
    assert np.mean(list(acc.values())) > np.mean(list(dw.values())) + 0.4


def test_focus_restricts_the_model_to_the_requested_nodes():
    g, roles, comp = roles_graph(K=4)
    focus = np.flatnonzero(roles != "leaf")
    m = Struc2Vec(dim=8, seed=0, **Struc2Vec.fast_params)
    E = m.embed_graph(g, focus)
    assert (np.linalg.norm(E[focus], axis=1) > 0).all() and (E[roles == "leaf"] == 0).all()


# ------------------------------------------------------------------ обёртка метода
@pytest.fixture(scope="module")
def data(processed):
    return TrainOnlyGuard(processed)


def test_registered_and_validated():
    assert "struc2vec" in methods.available()
    m = Struc2Vec(dim=4)
    assert m.uses_focus and "Ribeiro" in m.reference and m.overlap["pairs"].startswith("weak")
    for bad in ({"stay_prob": 1.5}, {"layers": -1}):
        with pytest.raises(ValueError):
            Struc2Vec(dim=4, **bad)
    with pytest.raises(ValueError, match="неизвестные параметры"):
        Struc2Vec(dim=4, bogus=1)


def test_output_contract_determinism_and_only_cohort_hosts_are_embedded(data):
    mk_ = lambda s: Struc2Vec(dim=5, seed=s, **Struc2Vec.fast_params).fit(data).embed(data, "train")
    e = mk_(1)
    assert list(e.index) == data.cohort and np.isfinite(e.to_numpy()).all() and (np.linalg.norm(e.to_numpy(), axis=1) > 0).all()
    assert np.array_equal(e.to_numpy(), mk_(1).to_numpy()) and not np.array_equal(e.to_numpy(), mk_(2).to_numpy())


def test_a_later_request_for_more_hosts_extends_the_focus_instead_of_failing(data):
    m = Struc2Vec(dim=4, seed=0, **Struc2Vec.fast_params).fit(data)
    part = data.cohort[:40]
    first = m.embed(data, "train", part)
    allh = m.embed(data, "train", data.cohort)
    assert list(first.index) == part and list(allh.index) == data.cohort and np.isfinite(allh.to_numpy()).all()


def test_fit_reads_only_the_train_window(processed):
    d = TrainOnlyGuard(processed)
    d.fitting = True
    try:
        Struc2Vec(dim=4, **Struc2Vec.fast_params).fit(d)
    finally:
        d.fitting = False


def test_runs_through_the_runner_and_the_tuner(processed, reference, tmp_path):
    m = run_experiment(processed, reference, tmp_path / "o", dim=5, n_boot=0, do_audit=False, methods=["struc2vec"],
                       method_config={"struc2vec": Struc2Vec.fast_params})
    assert "struc2vec" in m["methods"] and "struc2vec" not in m["missing_expected_methods"]
    assert m["graph_methods"]["struc2vec"]["n_candidate_pairs"] > 0
    res = json.loads((tmp_path / "o" / "results" / "test" / "struc2vec.json").read_text())
    assert res["metric_status"]["triplet_accuracy"] == "ok"
    assert (tmp_path / "o" / "identity" / "test" / "struc2vec.json").exists()           # позднее окно обучено на оцениваемых хостах
    out = tune(processed, reference, "struc2vec", {"layers": [1, 2], "stay_prob": [0.3, 0.7]}, dim=4, metric="pair_roc_auc", fast=True)
    assert out["n_configurations"] == 4 and out["test_reference_read"] is False
