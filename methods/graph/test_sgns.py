import numpy as np
import pytest
from scipy.special import expit

from methods.graph.graph_input import from_edges
from methods.graph.sgns import SkipGram, hs_loss_grads, huffman, neg_loss_grads, scatter_sub
from methods.graph.walks import walk_corpus


# ------------------------------------------------------------------ дерево Хаффмана
def test_huffman_codes_are_prefix_free_and_complete():
    freq = np.array([45, 13, 12, 16, 9, 5], dtype=float)           # пример из учебника
    paths, codes, lens = huffman(freq)
    words = ["".join(map(str, codes[i, :lens[i]])) for i in range(len(freq))]
    assert not any(a != b and b.startswith(a) for a in words for b in words)
    assert sum(2.0 ** -l for l in lens) == pytest.approx(1.0)       # равенство Крафта для полного двоичного дерева
    assert lens[0] == 1 and lens[4] == lens[5] == 4               # самые частые — короче всех, самые редкие — глубже
    assert (paths[:, 0] == paths[0, 0]).all()                      # каждый путь начинается в корне
    assert paths.max() == len(freq) - 2                            # внутренние узлы 0..n-2


def test_huffman_two_nodes_and_errors():
    paths, codes, lens = huffman(np.array([5.0, 1.0]))
    assert lens.tolist() == [1, 1] and sorted(codes[:, 0].tolist()) == [0, 1]
    with pytest.raises(ValueError):
        huffman(np.array([1.0]))


def test_huffman_is_deterministic_with_ties():
    f = np.ones(9)
    assert all(np.array_equal(a, b) for a, b in zip(huffman(f), huffman(f)))


# ------------------------------------------------------------------ проверка градиентов (конечные разности)
def numerical_grad(f, W, eps=1e-6):
    g = np.zeros_like(W)
    for idx in np.ndindex(*W.shape):
        old = W[idx]
        W[idx] = old + eps
        hi = f()
        W[idx] = old - eps
        lo = f()
        W[idx] = old
        g[idx] = (hi - lo) / (2 * eps)
    return g


def dense(idx, grad, shape):
    out = np.zeros(shape)
    np.add.at(out, idx, grad)
    return out


def test_negative_sampling_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    n, d = 7, 4
    W_in, W_out = rng.normal(size=(n, d)) * 0.5, rng.normal(size=(n, d)) * 0.5
    c, pos = np.array([0, 1, 1, 3, 0]), np.array([2, 2, 4, 5, 6])        # повторяющиеся центры и контексты
    negs = rng.integers(0, n, (5, 3))
    loss, g_in, oi, og = neg_loss_grads(W_in, W_out, c, pos, negs)
    f = lambda: neg_loss_grads(W_in, W_out, c, pos, negs)[0]
    assert loss == pytest.approx(f())
    assert np.allclose(dense(c, g_in, W_in.shape), numerical_grad(f, W_in), atol=1e-6)
    assert np.allclose(dense(oi, og, W_out.shape), numerical_grad(f, W_out), atol=1e-6)


def test_hierarchical_softmax_gradients_match_finite_differences():
    rng = np.random.default_rng(1)
    n, d = 9, 3
    paths, codes, lens = huffman(rng.integers(1, 20, n).astype(float))
    W_in, W_out = rng.normal(size=(n, d)) * 0.5, rng.normal(size=(n - 1, d)) * 0.5
    c, ctx = np.array([0, 4, 4, 8, 2, 2]), np.array([1, 1, 7, 3, 3, 0])
    loss, g_in, oi, og = hs_loss_grads(W_in, W_out, c, ctx, paths, codes, lens)
    f = lambda: hs_loss_grads(W_in, W_out, c, ctx, paths, codes, lens)[0]
    assert loss == pytest.approx(f())
    assert np.allclose(dense(c, g_in, W_in.shape), numerical_grad(f, W_in), atol=1e-6)
    assert np.allclose(dense(oi, og, W_out.shape), numerical_grad(f, W_out), atol=1e-6)


def test_hs_probabilities_over_all_leaves_sum_to_one():
    rng = np.random.default_rng(2)
    n, d = 8, 3
    paths, codes, lens = huffman(rng.integers(1, 20, n).astype(float))
    W_in, W_out = rng.normal(size=(n, d)), rng.normal(size=(n - 1, d))
    for c in range(n):
        total = 0.0
        for x in range(n):
            loss = hs_loss_grads(W_in, W_out, np.array([c]), np.array([x]), paths, codes, lens)[0]
            total += np.exp(-loss)
        assert total == pytest.approx(1.0)                         # настоящее распределение по контекстам


def test_scatter_sub_sums_repeated_indices():
    W = np.zeros((5, 2))
    ref = W.copy()
    idx = np.array([3, 1, 3, 3, 0])
    grad = np.arange(10, dtype=float).reshape(5, 2)
    scatter_sub(W, idx, grad, 0.5)
    np.subtract.at(ref, idx, 0.5 * grad)
    assert np.allclose(W, ref)


# ------------------------------------------------------------------ обучение
def planted(n_half=25, seed=0):
    rng = np.random.default_rng(seed)
    n = 2 * n_half
    src, dst = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < (0.3 if (i < n_half) == (j < n_half) else 0.01):
                src.append(i); dst.append(j)
    return from_edges(np.array(src), np.array(dst), np.ones(len(src)), np.array([f"h{i}" for i in range(n)], dtype=object)), n_half


@pytest.mark.parametrize("objective,margin", [("neg", 0.25), ("hs", 0.2)])
def test_recovers_planted_communities(objective, margin):
    g, h = planted()
    walks = walk_corpus(g, 10, 40, seed=1)
    sg = SkipGram(g.n, 16, objective=objective, window=5, epochs=3, seed=2, node_freq=g.strength).fit(walks)
    E = sg.embeddings / np.linalg.norm(sg.embeddings, axis=1, keepdims=True)
    S = E @ E.T
    within = (S[:h, :h].sum() - h + S[h:, h:].sum() - h) / (2 * h * (h - 1))
    across = S[:h, h:].mean()
    assert within > across + margin, (within, across)


def test_loss_decreases_and_run_is_deterministic():
    g, _ = planted()
    walks = walk_corpus(g, 6, 30, seed=1)
    mk = lambda seed: SkipGram(g.n, 8, window=4, epochs=4, seed=seed, node_freq=g.strength, max_chunk_pairs=20000).fit(walks)
    a, b, c = mk(3), mk(3), mk(4)
    assert a.loss_history[-1] < a.loss_history[0] * 0.95
    assert np.array_equal(a.embeddings, b.embeddings) and not np.array_equal(a.embeddings, c.embeddings)


def test_expected_pair_count_is_accurate():
    sg = SkipGram(10, 4, window=5, seed=0)
    walks = np.random.default_rng(0).integers(0, 10, (400, 30))
    counts = [len(sg._pairs(walks)[0]) for _ in range(5)]
    assert np.mean(counts) == pytest.approx(sg.expected_pairs(400, 30), rel=0.03)


def test_pairs_respect_the_window_and_both_directions():
    sg = SkipGram(100, 4, window=3, seed=0)
    walks = np.arange(40).reshape(1, 40)
    c, x = sg._pairs(walks)
    assert (np.abs(c - x) <= 3).all() and (c != x).all()
    pairs = set(zip(c.tolist(), x.tolist()))
    assert (5, 6) in pairs and (6, 5) in pairs                      # соседние вершины образуют пары в обе стороны


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError):
        SkipGram(5, 4, objective="softmax")
    with pytest.raises(ValueError):
        SkipGram(5, 0)
    with pytest.raises(ValueError):
        SkipGram(5, 4, node_freq=np.ones(3))
    sg = SkipGram(5, 4)
    with pytest.raises(ValueError, match="walks"):
        sg.fit(np.array([[0, 7, 1]]))
    with pytest.raises(ValueError, match="walks"):
        sg.fit(np.array([[0], [1]]))
