import json

import numpy as np
import pytest
import scipy.sparse as sp

from methods import graph as methods
from methods.graph.conftest_helpers import TrainOnlyGuard
from methods.graph.deepwalk import DeepWalk
from methods.graph.graph_input import from_edges
from methods.graph.rolx import (BASE_NAMES, RolX, aggregate, base_features, kl_bits, log_bin, mdl_roles, nmf_roles,
                                    prepare_matrix, refex, triangles)
from experiments.run_experiment import run_experiment
from experiments.tune_methods import tune


def mk(edges, n=None):
    e = np.array(edges)
    n = n or int(e.max()) + 1
    return from_edges(e[:, 0], e[:, 1], np.ones(len(e)), np.array([f"h{i}" for i in range(n)], dtype=object))


# ------------------------------------------------------------------ базовые признаки (посчитаны вручную)
def test_base_features_triangle_star_path_and_clique():
    tri = base_features(mk([(0, 1), (1, 2), (0, 2)]))
    assert (tri == [2, 2, 3, 0]).all()                                  # степень 2, рёбер в эго-сети 3, наружу ни одного
    star = base_features(mk([(0, 1), (0, 2), (0, 3)]))
    assert star[0].tolist() == [3, 3, 3, 0] and star[1].tolist() == [1, 1, 1, 2]
    path = base_features(mk([(0, 1), (1, 2)]))
    assert path[1].tolist() == [2, 2, 2, 0] and path[0].tolist() == [1, 1, 1, 1]
    k4 = base_features(mk([(i, j) for i in range(4) for j in range(i + 1, 4)]))
    assert (k4 == [3, 3, 6, 0]).all()
    assert BASE_NAMES == ["degree", "strength", "ego_edges", "ego_out_edges"]


def test_egonet_identity_holds_on_random_graphs():
    """Рёбра внутри и наружу эго-сети покрывают все рёбра соседей: sum deg(u) = 2T + deg + out."""
    rng = np.random.default_rng(0)
    e = {tuple(sorted(rng.integers(0, 40, 2))) for _ in range(160)}
    g = mk([x for x in e if x[0] != x[1]], n=40)
    B = g.adjacency().toarray() > 0
    F = base_features(g)
    for v in np.flatnonzero(g.degree):
        nb = np.flatnonzero(B[v])
        inside = B[np.ix_(nb, nb)].sum() / 2
        assert F[v, 2] == len(nb) + inside
        assert F[v, 3] == B[nb].sum() - 2 * inside - len(nb)


def test_triangles_match_the_adjacency_cube_and_chunking_does_not_matter():
    rng = np.random.default_rng(1)
    g = mk([x for x in {tuple(sorted(rng.integers(0, 30, 2))) for _ in range(120)} if x[0] != x[1]], n=30)
    A = (g.adjacency().toarray() > 0).astype(float)
    want = np.diag(A @ A @ A) / 2
    B = g.adjacency().copy()
    B.data[:] = 1.0
    assert np.allclose(triangles(B), want) and np.allclose(triangles(B, chunk=4), want)


def test_aggregate_sum_and_mean_over_neighbours():
    g = mk([(0, 1), (0, 2), (2, 3)])
    B = g.adjacency().copy()
    B.data[:] = 1.0
    F = np.array([[1.0], [2.0], [3.0], [4.0]])
    out = aggregate(B, F)
    assert out[:, 0].tolist() == [5.0, 1.0, 5.0, 3.0]                   # суммы по соседям
    assert out[:, 1].tolist() == [2.5, 1.0, 2.5, 3.0]                   # средние


# ------------------------------------------------------------------ корзины и прореживание
def test_log_bin_example_ties_and_monotonicity():
    assert log_bin(np.array([1, 1, 2, 3, 3, 3, 10, 10])).tolist() == [0, 0, 0, 1, 1, 1, 3, 3]
    x = np.random.default_rng(0).normal(size=500)
    b = log_bin(x)
    assert (np.diff(b[np.argsort(x)]) >= 0).all()                       # монотонно по значению
    assert len(set(log_bin(np.full(10, 7.0)).tolist())) == 1                     # равные значения всегда в одной корзине


def test_log_bin_sizes_are_geometric():
    counts = np.bincount(log_bin(np.arange(1024.0)))
    assert counts[:4].tolist() == [512, 256, 128, 64]


def test_refex_prunes_duplicates_and_constants():
    star = mk([(0, i) for i in range(1, 8)])                            # здесь сила == степени
    F, names = refex(star, rounds=0)
    assert "strength" not in names and "degree" in names
    with pytest.raises(ValueError, match="непостоянного"):
        refex(mk([(i, (i + 1) % 10) for i in range(10)]))                # кольцо: все признаки постоянны


def test_recursion_refines_the_partition_of_nodes():
    """Агрегация по соседям различает вершины, которые локальные признаки не различают (k-шаговая структура)."""
    rng = np.random.default_rng(2)
    g = mk([x for x in {tuple(sorted(rng.integers(0, 80, 2))) for _ in range(190)} if x[0] != x[1]], n=80)
    distinct = lambda F: len({tuple(np.round(r, 9)) for r in F})
    base, _ = refex(g, rounds=0)
    r1, _ = refex(g, rounds=1)
    r2, _ = refex(g, rounds=2)
    assert distinct(r1) > distinct(base) and distinct(r2) >= distinct(r1)


def test_rounds_max_features_and_similarity_control_the_feature_count():
    rng = np.random.default_rng(2)
    g = mk([x for x in {tuple(sorted(rng.integers(0, 60, 2))) for _ in range(200)} if x[0] != x[1]], n=60)
    n0 = refex(g, rounds=0)[0].shape[1]
    n2 = refex(g, rounds=2)[0].shape[1]
    assert n2 > n0
    assert refex(g, rounds=2, max_features=5)[0].shape[1] <= 5
    assert refex(g, rounds=2, s=2)[0].shape[1] < n2                      # более мягкая похожесть прореживает сильнее


# ------------------------------------------------------------------ NMF и MDL
def test_nmf_roles_shape_nonnegativity_determinism():
    V = prepare_matrix(np.random.default_rng(0).random((50, 8)) * 20)
    G = nmf_roles(V, 4, 0)
    assert G.shape == (50, 4) and (G >= 0).all() and np.array_equal(G, nmf_roles(V, 4, 0))
    assert nmf_roles(V, 99, 0).shape[1] == 8                              # не больше числа признаков


def test_prepare_matrix_is_nonnegative_and_scaled():
    V = prepare_matrix(np.array([[0.0, 5.0], [10.0, 50.0]]))
    assert V.min() >= 0 and np.allclose(V.max(axis=0), 1.0)


def test_kl_bits_is_zero_for_a_perfect_fit_and_positive_otherwise():
    V = np.random.default_rng(0).random((6, 4))
    assert kl_bits(V, V) == pytest.approx(0.0, abs=1e-9) and kl_bits(V, V * 1.5) > 0


def test_mdl_mechanics():
    V = prepare_matrix(np.random.default_rng(1).random((80, 6)) * 30)
    best, costs = mdl_roles(V, 2, 5, 0)
    assert best == min(costs, key=costs.get) and set(costs) == {2, 3, 4, 5}
    assert best == mdl_roles(V, 2, 5, 0)[0]


# ------------------------------------------------------------------ смысл RolX: структурные роли в разных компонентах
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


def other_component_role_accuracy(E, roles, comp):
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    S = E @ E.T
    S[comp[:, None] == comp[None, :]] = -9                                # кандидаты — только из других компонент
    nn = S.argmax(1)
    return {r: float(np.mean(roles[nn[roles == r]] == r)) for r in ("hub", "attached", "leaf", "clique")}


def test_nodes_with_the_same_structural_role_match_across_disconnected_components():
    g, roles, comp = roles_graph()
    acc = other_component_role_accuracy(RolX(dim=6, seed=0).embed_graph(g), roles, comp)
    assert all(v >= 0.95 for v in acc.values()), acc
    dw = DeepWalk(dim=16, seed=0, num_walks=8, walk_length=30, window=5, epochs=2).embed_graph(g)
    contrast = other_component_role_accuracy(dw, roles, comp)
    # у DeepWalk нет механизма для ролей (он видит только совместные появления внутри компоненты); сигнал роли
    # у него случаен (word2vec кодирует частоту вершины, то есть степень) и для части ролей не работает
    assert min(contrast.values()) < 0.5 and np.mean(list(acc.values())) > np.mean(list(contrast.values())) + 0.4


def test_embedding_is_invariant_to_node_relabelling():
    g, roles, comp = roles_graph(K=5)
    perm = np.random.default_rng(0).permutation(g.n)
    a = g.adjacency()
    P = sp.csr_matrix((np.ones(g.n), (perm, np.arange(g.n))), shape=(g.n, g.n))      # вершина i -> perm[i]
    ap = (P @ a @ P.T).tocoo()
    keep = ap.row < ap.col
    g2 = from_edges(ap.row[keep], ap.col[keep], ap.data[keep], g.nodes)
    E1 = RolX(dim=5, seed=0).embed_graph(g)
    E2 = RolX(dim=5, seed=0).embed_graph(g2)
    n1, n2 = E1 / np.linalg.norm(E1, axis=1, keepdims=True), E2 / np.linalg.norm(E2, axis=1, keepdims=True)
    assert np.allclose((n1 @ n1.T), (n2 @ n2.T)[np.ix_(perm, perm)], atol=1e-4)      # та же структура похожести


# ------------------------------------------------------------------ обёртка метода
@pytest.fixture(scope="module")
def data(processed):
    return TrainOnlyGuard(processed)


def test_registered_with_metadata():
    assert "rolx" in methods.available()
    m = RolX(dim=6)
    assert "Henderson" in m.reference and m.shared_coordinate_system is False and m.overlap["pairs"].startswith("weak")
    with pytest.raises(ValueError, match="неизвестные параметры"):
        RolX(dim=6, bogus=1)


def test_output_contract_determinism_and_normalisation(data):
    mk_ = lambda: RolX(dim=6, seed=1).fit(data).embed(data, "train")
    e = mk_()
    X = e.to_numpy()
    assert list(e.index) == data.cohort and X.shape == (len(data.cohort), 6) and (X >= 0).all()
    assert np.allclose(X.sum(axis=1), 1.0)                                # принадлежность ролям нормирована по L1
    assert np.array_equal(X, mk_().to_numpy())
    raw = RolX(dim=6, seed=1, normalize=False).fit(data).embed(data, "train").to_numpy()
    assert not np.allclose(raw.sum(axis=1), 1.0)


def test_n_roles_overrides_the_dimension_and_mdl_is_supported(data):
    assert RolX(dim=6, n_roles=3).fit(data).embed(data, "train").shape[1] == 3
    m = RolX(dim=5, n_roles="mdl", **RolX.fast_params).fit(data)
    assert 2 <= m.embed(data, "train").shape[1] <= 5 and m.info()["n_structural_features"] > 0


def test_fit_reads_only_the_train_window(processed):
    d = TrainOnlyGuard(processed)
    d.fitting = True
    try:
        RolX(dim=4, **RolX.fast_params).fit(d)
    finally:
        d.fitting = False


def test_runs_through_the_runner_and_the_tuner(processed, reference, tmp_path):
    m = run_experiment(processed, reference, tmp_path / "o", dim=6, n_boot=0, do_audit=False, methods=["rolx"],
                       method_config={"rolx": RolX.fast_params})
    assert "rolx" in m["methods"] and "rolx" not in m["missing_expected_methods"]
    assert m["graph_methods"]["rolx"]["n_structural_features"] > 0
    res = json.loads((tmp_path / "o" / "results" / "test" / "rolx.json").read_text())
    assert res["metric_status"]["triplet_accuracy"] == "ok"
    out = tune(processed, reference, "rolx", {"rounds": [1, 2], "n_roles": [3, 5]}, dim=5, metric="pair_roc_auc", fast=True)
    assert out["n_configurations"] == 4 and out["test_reference_read"] is False
