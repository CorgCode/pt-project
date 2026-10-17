import numpy as np
import pytest
from scipy.stats import chisquare

from methods.graph.graph_input import from_edges
from methods.graph.walks import random_walks, second_order_step, walk_corpus


def graph(edges, n=None, weights=None):
    e = np.array(edges)
    n = n or int(e.max()) + 1
    return from_edges(e[:, 0], e[:, 1], np.ones(len(e)) if weights is None else np.asarray(weights, float),
                      np.array([f"h{i}" for i in range(n)], dtype=object))


def ring(n=12):
    return graph([(i, (i + 1) % n) for i in range(n)])


def test_walks_only_follow_edges_and_start_where_asked():
    g = graph([(0, 1), (1, 2), (2, 3), (3, 0), (1, 3), (3, 4)])
    for p, q in ((1.0, 1.0), (0.5, 2.0)):
        w = random_walks(g, np.arange(g.n), 30, p=p, q=q, rng=np.random.default_rng(0))
        assert w.shape == (g.n, 30) and w.dtype == np.int32 and (w[:, 0] == np.arange(g.n)).all()
        assert g.has_edge(w[:, :-1].ravel(), w[:, 1:].ravel()).all()


def test_deterministic_given_the_generator_and_different_otherwise():
    g = ring()
    a = walk_corpus(g, 3, 20, p=0.5, q=2.0, seed=4)
    assert np.array_equal(a, walk_corpus(g, 3, 20, p=0.5, q=2.0, seed=4))
    assert not np.array_equal(a, walk_corpus(g, 3, 20, p=0.5, q=2.0, seed=5))


def test_corpus_has_num_walks_per_node():
    g = graph([(0, 1), (1, 2), (2, 0), (2, 3)])
    w = walk_corpus(g, 5, 8, seed=0)
    assert w.shape == (5 * g.n, 8)
    assert np.bincount(w[:, 0], minlength=g.n).tolist() == [5] * g.n


def test_invalid_arguments():
    g = ring()
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="walk_length"):
        random_walks(g, [0], 1, rng=rng)
    with pytest.raises(ValueError, match="> 0"):
        random_walks(g, [0], 5, p=0.0, rng=rng)
    iso = graph([(0, 1)], n=3)
    with pytest.raises(ValueError, match="изолированной"):
        random_walks(iso, [2], 5, rng=rng)


def test_first_order_transitions_are_proportional_to_edge_weights():
    g = graph([(0, 1), (0, 2), (0, 3)], weights=[1.0, 2.0, 7.0])
    w = random_walks(g, np.zeros(40000, dtype=int), 2, rng=np.random.default_rng(1))
    obs = np.bincount(w[:, 1], minlength=4)[1:]
    assert chisquare(obs, 40000 * np.array([0.1, 0.2, 0.7])).pvalue > 1e-3


# ---- перекос node2vec против аналитического распределения --------------------------------
#   t=0 -> v=1; соседи v: 0 (t, возврат), 2 (сосед t), 3 и 4 (на расстоянии 2 от t)
BIAS = graph([(0, 1), (1, 2), (1, 3), (1, 4), (0, 2)], weights=[1.0, 1.0, 1.0, 3.0, 1.0])


def analytic(p, q):
    w = {0: 1.0 / p, 2: 1.0, 3: 1.0 / q, 4: 3.0 / q}                 # w(v,x) * alpha(t,x); веса рёбер вокруг v=1: 1,1,1,3
    s = sum(w.values())
    return np.array([w[0], w[2], w[3], w[4]]) / s


@pytest.mark.parametrize("p,q", [(1.0, 1.0), (0.25, 4.0), (4.0, 0.25), (0.5, 2.0), (2.0, 0.5)])
def test_second_order_step_matches_the_node2vec_distribution(p, q):
    n = 60000
    t, v = np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)
    nxt = second_order_step(BIAS, t, v, p, q, np.random.default_rng(3))
    obs = np.array([(nxt == x).sum() for x in (0, 2, 3, 4)])
    assert chisquare(obs, n * analytic(p, q)).pvalue > 1e-3, (obs / n, analytic(p, q))


def test_exact_fallback_agrees_when_rejection_is_forced_to_give_up():
    p, q = 0.01, 100.0                          # вероятность принятия ~1e-4 для большинства предложений
    n = 4000
    t, v = np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)
    nxt = second_order_step(BIAS, t, v, p, q, np.random.default_rng(5), max_rounds=1)    # почти всё — точным перебором
    obs = np.array([(nxt == x).sum() for x in (0, 2, 3, 4)])
    assert chisquare(obs, n * analytic(p, q)).pvalue > 1e-3


def test_return_parameter_controls_backtracking():
    g = ring(20)
    rates = []
    for p in (0.2, 1.0, 5.0):
        w = random_walks(g, np.arange(g.n).repeat(40), 30, p=p, q=1.0, rng=np.random.default_rng(2))
        rates.append((w[:, 2:] == w[:, :-2]).mean())                # доля немедленных возвратов
    assert rates[0] > rates[1] > rates[2]


def test_inout_parameter_controls_exploration():
    # на графе из цепочки клик большой q (держаться рядом) оставляет блуждание в меньшей окрестности, чем малый q
    g = graph([(i, j) for i in range(5) for j in range(i + 1, 5)] + [(4, 5)] + [(i, j) for i in range(5, 10) for j in range(i + 1, 10)])
    spans = []
    for q in (0.2, 5.0):
        w = random_walks(g, np.zeros(2000, dtype=int), 25, p=1.0, q=q, rng=np.random.default_rng(0))
        spans.append((w >= 5).mean())                                # доля времени в дальней клике
    assert spans[0] > spans[1]
