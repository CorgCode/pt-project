import numpy as np
import pandas as pd
import pytest

from methods.graph.graph_input import from_edges, from_flows


def flows(rows):
    return pd.DataFrame(rows, columns=["src", "dst"])


def test_undirected_weights_count_both_directions_and_drop_self_flows():
    g = from_flows(flows([("a", "b")] * 3 + [("b", "a")] * 2 + [("a", "c"), ("c", "c")]))
    assert list(g.nodes) == ["a", "b", "c"] and g.n_edges == 2
    a = g.adjacency().toarray()
    assert a[0, 1] == a[1, 0] == pytest.approx(np.log1p(5))          # 3 + 2 потока
    assert a[0, 2] == a[2, 0] == pytest.approx(np.log1p(1)) and a[2, 2] == 0   # петля выброшена


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
        assert (np.diff(nb) > 0).all()                               # по возрастанию, без дублей
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
    with pytest.raises(KeyError, match="нет рёбер"):
        g.index_of(["a", "ghost"])


def test_empty_or_self_only_flows_are_errors():
    with pytest.raises(ValueError, match="между разными хостами"):
        from_flows(flows([("a", "a")]))


def test_degree_strength_and_cumulative_weights():
    g = from_flows(flows([("a", "b")] * 2 + [("a", "c")]))
    assert g.degree.tolist() == [2, 1, 1]
    assert g.strength[0] == pytest.approx(np.log1p(2) + np.log1p(1))
    assert g.cum[-1] == pytest.approx(g.weights.sum()) and g.cum[0] == 0


def test_from_edges_roundtrip():
    g = from_edges(np.array([0, 1]), np.array([1, 2]), np.array([1.0, 2.0]), np.array(["x", "y", "z"], dtype=object))
    assert g.n == 3 and g.n_edges == 2 and g.adjacency().toarray()[1, 2] == 2.0


def _toy_flows():
    import pandas as pd
    rows = []
    for i in range(6):                       # 6 клиентов ходят на web-сервер и DNS
        rows += [(f"c{i}", "web", "6", "443"), (f"c{i}", "dns", "17", "53")]
    rows += [("c0", "c1", "6", "Port1"), ("x", "web", "6", "443")]
    return pd.DataFrame(rows, columns=["src", "dst", "proto", "dst_port"])


def test_hosts_graph_is_unchanged_by_the_new_options():
    import numpy as np
    from methods.graph.graph_input import from_flows
    f = _toy_flows()
    g = from_flows(f)
    assert set(g.nodes) == {"c0", "c1", "c2", "c3", "c4", "c5", "web", "dns", "x"}
    assert np.isclose(g.weights[g.indptr[0]:g.indptr[1]].max(), np.log1p(1))


def test_service_graph_connects_hosts_that_use_the_same_services():
    from methods.graph.graph_input import from_flows
    g = from_flows(_toy_flows(), graph="services", min_service_hosts=5)
    assert "uses:6/443" in set(g.nodes)                                           # общий сервис клиентов
    assert "serves:6/443" in set(g.nodes)            # редкий, но у web других сервисов нет — оставлен, чтобы не изолировать
    c0, svc = g.index_of(["c0"])[0], g.index_of(["uses:6/443"])[0]
    assert svc in set(g.neighbors(c0)) and "uses:6/Port1" not in set(g.nodes)     # редкий порт выброшен
    assert all(g.degree[g.index_of([h])[0]] > 0 for h in ("web", "dns"))          # сервер с одними редкими — не изолирован


def test_hub_norm_weakens_edges_of_hubs():
    import numpy as np
    from methods.graph.graph_input import from_flows
    f = _toy_flows()
    plain, norm = from_flows(f), from_flows(f, weight="hub_norm")
    web = plain.index_of(["web"])[0]
    w_plain = plain.weights[plain.indptr[web]:plain.indptr[web + 1]].mean()
    w_norm = norm.weights[norm.indptr[web]:norm.indptr[web + 1]].mean()
    assert w_norm < w_plain / 2                                                   # у хаба веса ощутимо меньше
    with __import__("pytest").raises(ValueError, match="graph"):
        from_flows(f, graph="nope")
