import numpy as np
import pandas as pd
import pytest

from benchmark.methods.graph_input import from_edges, from_flows


def flows(rows):
    return pd.DataFrame(rows, columns=["src", "dst"])


def test_undirected_weights_count_both_directions_and_drop_self_flows():
    g = from_flows(flows([("a", "b")] * 3 + [("b", "a")] * 2 + [("a", "c"), ("c", "c")]))
    assert list(g.nodes) == ["a", "b", "c"] and g.n_edges == 2
    a = g.adjacency().toarray()
    assert a[0, 1] == a[1, 0] == pytest.approx(np.log1p(5))          # 3 + 2 flows
    assert a[0, 2] == a[2, 0] == pytest.approx(np.log1p(1)) and a[2, 2] == 0   # self flow dropped


def test_weight_modes():
    rows = [("a", "b")] * 4 + [("b", "c")]
    assert from_flows(flows(rows), "count").adjacency().toarray()[0, 1] == 4
    assert from_flows(flows(rows), "binary").adjacency().toarray()[0, 1] == 1
    with pytest.raises(ValueError):
        from_flows(flows(rows), "nope")


def test_csr_is_symmetric_sorted_and_positive():
    rng = np.random.default_rng(0)
    rows = [(f"h{rng.integers(30)}", f"h{rng.integers(30)}") for _ in range(400)]
    g = from_flows(flows(rows))
    a = g.adjacency()
    assert (abs(a - a.T) > 1e-12).nnz == 0 and (g.weights > 0).all()
    for v in range(g.n):
        nb = g.neighbors(v)
        assert (np.diff(nb) > 0).all()                               # sorted, no duplicates
    assert list(g.nodes) == sorted(g.nodes)


def test_has_edge_matches_a_python_set():
    rng = np.random.default_rng(1)
    g = from_flows(flows([(f"h{rng.integers(25)}", f"h{rng.integers(25)}") for _ in range(120)]))
    edges = {(v, int(x)) for v in range(g.n) for x in g.neighbors(v)}
    a, b = rng.integers(0, g.n, 500), rng.integers(0, g.n, 500)
    assert g.has_edge(a, b).tolist() == [(int(i), int(j)) in edges for i, j in zip(a, b)]


def test_index_of_and_unknown_hosts():
    g = from_flows(flows([("a", "b"), ("b", "c")]))
    assert g.index_of(["c", "a"]).tolist() == [2, 0]
    with pytest.raises(KeyError, match="no edge"):
        g.index_of(["a", "ghost"])


def test_empty_or_self_only_flows_are_errors():
    with pytest.raises(ValueError, match="non-self"):
        from_flows(flows([("a", "a")]))


def test_degree_strength_and_cumulative_weights():
    g = from_flows(flows([("a", "b")] * 2 + [("a", "c")]))
    assert g.degree.tolist() == [2, 1, 1]
    assert g.strength[0] == pytest.approx(np.log1p(2) + np.log1p(1))
    assert g.cum[-1] == pytest.approx(g.weights.sum()) and g.cum[0] == 0


def test_from_edges_roundtrip():
    g = from_edges(np.array([0, 1]), np.array([1, 2]), np.array([1.0, 2.0]), np.array(["x", "y", "z"], dtype=object))
    assert g.n == 3 and g.n_edges == 2 and g.adjacency().toarray()[1, 2] == 2.0
