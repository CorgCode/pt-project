"""Skip-gram для эмбеддингов вершин (numpy, детерминированно, без лишних зависимостей).

Две целевые функции на парах (центр c, контекст x) из корпуса блужданий:
  neg  negative sampling (как в word2vec / node2vec):
         L = -log s(v_c . u_x) - sum_{k=1..K} log s(-v_c . u_{n_k}),   n_k ~ freq^0.75
  hs   hierarchical softmax по дереву Хаффмана (как в статье DeepWalk):
         L = -sum_{j in path(x)} log s( (1 - 2 b_j) * v_c . theta_j )

Эмбеддинги — строки W_in. Окно динамическое (размер для каждого центра случаен от 1 до window), в обе стороны.
Мини-батч SGD с линейно убывающим шагом; градиенты одной вершины в батче суммируются. Один seed — один результат.
"""
from __future__ import annotations

import heapq

import numpy as np
from scipy.special import expit

from pipeline.speed import progress


# ------------------------------------------------------------------------------------ дерево Хаффмана
def huffman(freq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Коды Хаффмана для листьев 0..n-1: (пути по внутренним узлам, биты, длины), дополнены -1 / 0."""
    n = len(freq)
    if n < 2:
        raise ValueError("нужно хотя бы две вершины")
    heap = [(float(f), i) for i, f in enumerate(freq)]
    heapq.heapify(heap)
    parent = np.zeros(2 * n - 1, dtype=np.int64)
    bit = np.zeros(2 * n - 1, dtype=np.int8)
    nxt = n
    while len(heap) > 1:
        f1, a = heapq.heappop(heap)
        f2, b = heapq.heappop(heap)
        parent[a] = parent[b] = nxt
        bit[b] = 1
        heapq.heappush(heap, (f1 + f2, nxt))
        nxt += 1
    root = nxt - 1
    chains = []
    for leaf in range(n):
        nodes, bits, cur = [], [], leaf
        while cur != root:
            bits.append(bit[cur])
            cur = parent[cur]
            nodes.append(cur - n)                      # внутренние узлы нумеруются 0..n-2
        chains.append((nodes[::-1], bits[::-1]))       # от корня к листу
    lens = np.array([len(c[0]) for c in chains], dtype=np.int64)
    paths = np.full((n, lens.max()), -1, dtype=np.int64)
    codes = np.zeros((n, lens.max()), dtype=np.int8)
    for i, (nodes, bits) in enumerate(chains):
        paths[i, :len(nodes)], codes[i, :len(bits)] = nodes, bits
    return paths, codes, lens


# ------------------------------------------------------------------------------------ функции потерь
def neg_loss_grads(W_in, W_out, c, pos, negs):
    """Сумма потерь и градиенты батча: (loss, g_in (B,d), out_idx, out_grad)."""
    v, up, un = W_in[c], W_out[pos], W_out[negs]                      # (B,d) (B,d) (B,K,d)
    s_pos, s_neg = (v * up).sum(1), np.einsum("bd,bkd->bk", v, un)
    loss = -np.log(expit(s_pos) + 1e-300).sum() - np.log(expit(-s_neg) + 1e-300).sum()
    d_pos, d_neg = expit(s_pos) - 1.0, expit(s_neg)                    # dL/ds
    g_in = d_pos[:, None] * up + np.einsum("bk,bkd->bd", d_neg, un)
    out_idx = np.concatenate([pos, negs.reshape(-1)])
    out_grad = np.concatenate([d_pos[:, None] * v, (d_neg[:, :, None] * v[:, None, :]).reshape(-1, v.shape[1])])
    return loss, g_in, out_idx, out_grad


def hs_loss_grads(W_in, W_out, c, ctx, paths, codes, lens):
    v = W_in[c]
    P, C, L = paths[ctx], codes[ctx], lens[ctx]
    mask = np.arange(P.shape[1])[None, :] < L[:, None]
    theta = W_out[np.where(mask, P, 0)]                                # (B,Lmax,d)
    sign = 1.0 - 2.0 * C
    z = sign * np.einsum("bd,bld->bl", v, theta)
    loss = -(np.log(expit(z) + 1e-300) * mask).sum()
    ds = (expit(z) - 1.0) * sign * mask                                # dL/d(v.theta)
    g_in = np.einsum("bl,bld->bd", ds, theta)
    out_grad = (ds[:, :, None] * v[:, None, :]).reshape(-1, v.shape[1])
    return loss, g_in, np.where(mask, P, 0).reshape(-1), out_grad


def scatter_sub(W: np.ndarray, idx: np.ndarray, grad: np.ndarray, lr: float) -> None:
    """W[idx] -= lr * grad, повторные индексы суммируются."""
    order = np.argsort(idx, kind="stable")
    sidx = idx[order]
    first = np.r_[True, sidx[1:] != sidx[:-1]]
    W[sidx[first]] -= lr * np.add.reduceat(grad[order], np.flatnonzero(first), axis=0)


# ------------------------------------------------------------------------------------ обучение
class SkipGram:
    def __init__(self, n_nodes: int, dim: int, *, objective: str = "neg", window: int = 10, negative: int = 5,
                 lr: float = 0.025, min_lr: float = 1e-4, epochs: int = 1, batch_size: int = 512, seed: int = 0,
                 node_freq: np.ndarray | None = None, max_chunk_pairs: int = 4_000_000, engine: str = "numba"):
        if engine not in ("numba", "numpy"):
            raise ValueError("engine должен быть 'numba' или 'numpy'")
        self.engine = engine
        if objective not in ("neg", "hs"):
            raise ValueError("objective должен быть 'neg' или 'hs'")
        if min(n_nodes, dim, window, epochs, batch_size) < 1 or (objective == "neg" and negative < 1):
            raise ValueError("n_nodes, dim, window, epochs, batch_size (и negative для 'neg') должны быть >= 1")
        self.n, self.dim, self.objective = n_nodes, dim, objective
        self.window, self.negative, self.lr, self.min_lr = window, negative, lr, min_lr
        self.epochs, self.batch_size, self.max_chunk_pairs = epochs, batch_size, max_chunk_pairs
        self.rng = np.random.default_rng(seed)
        freq = np.ones(n_nodes) if node_freq is None else np.asarray(node_freq, dtype=float)
        if len(freq) != n_nodes or (freq < 0).any():
            raise ValueError("node_freq: по одному неотрицательному значению на вершину")
        self.W_in = (self.rng.random((n_nodes, dim)) - 0.5) / dim
        if objective == "neg":
            self.W_out = np.zeros((n_nodes, dim))
            p = np.maximum(freq, 1e-12) ** 0.75
            self._neg_cdf = np.cumsum(p / p.sum())
        else:
            self.W_out = np.zeros((n_nodes - 1, dim))
            self.paths, self.codes, self.lens = huffman(freq)
        self.loss_history: list[float] = []

    # -------------------------------------------------------------------------------- пары
    def _pairs(self, walks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        b = self.rng.integers(1, self.window + 1, size=walks.shape)       # своё окно у каждого центра
        cs, xs = [], []
        for o in range(1, min(self.window, walks.shape[1] - 1) + 1):
            fwd = b[:, :-o] >= o                                          # центр i, контекст i+o
            cs.append(walks[:, :-o][fwd]); xs.append(walks[:, o:][fwd])
            bwd = b[:, o:] >= o                                           # центр i+o, контекст i
            cs.append(walks[:, o:][bwd]); xs.append(walks[:, :-o][bwd])
        return np.concatenate(cs), np.concatenate(xs)

    def expected_pairs(self, n_walks: int, length: int) -> float:
        return n_walks * sum(2 * (length - o) * (self.window - o + 1) / self.window
                             for o in range(1, min(self.window, length - 1) + 1))

    # -------------------------------------------------------------------------------- обучение
    def fit(self, walks: np.ndarray) -> "SkipGram":
        walks = np.asarray(walks)
        if walks.ndim != 2 or walks.shape[1] < 2 or walks.min() < 0 or walks.max() >= self.n:
            raise ValueError("walks — массив (n_walks, длина>=2) индексов вершин из [0, n_nodes)")
        total = self.epochs * self.expected_pairs(len(walks), walks.shape[1])
        done = 0.0
        bar = progress(total=int(total), desc="skip-gram", unit=" пар", unit_scale=True, leave=False)
        per_chunk = max(1, int(self.max_chunk_pairs / max(1.0, self.expected_pairs(1, walks.shape[1]))))
        for _ in range(self.epochs):
            order = self.rng.permutation(len(walks))
            for s in range(0, len(walks), per_chunk):
                c, x = self._pairs(walks[order[s:s + per_chunk]])
                perm = self.rng.permutation(len(c))
                c, x = c[perm], x[perm]
                starts = np.arange(0, len(c), self.batch_size)
                lrs = np.maximum(self.min_lr, self.lr * (1.0 - (done + starts) / max(total, 1.0)))
                if self.engine == "numba":
                    chunk_loss = self._chunk_numba(c, x, lrs)
                else:
                    chunk_loss = sum(self._step(c[i:i + self.batch_size], x[i:i + self.batch_size], lr)
                                     for i, lr in zip(starts, lrs))
                done += len(c)
                self.loss_history.append(chunk_loss / max(len(c), 1))
                bar.update(len(c))
                bar.set_postfix_str(f"потери {self.loss_history[-1]:.3f}")
        bar.close()
        return self

    def _chunk_numba(self, c: np.ndarray, x: np.ndarray, lrs: np.ndarray) -> float:
        from methods.graph import kernels
        c, x = c.astype(np.int64), x.astype(np.int64)
        if self.objective == "neg":                      # те же случайные числа, что по батчам в _step
            negs = np.minimum(np.searchsorted(self._neg_cdf, self.rng.random((len(c), self.negative))), self.n - 1)
            return float(kernels.sgns_neg_epoch(self.W_in, self.W_out, c, x, negs.astype(np.int64), lrs,
                                                self.batch_size))
        return float(kernels.sgns_hs_epoch(self.W_in, self.W_out, c, x, self.paths, self.codes.astype(np.int64),
                                           self.lens, lrs, self.batch_size))

    def _step(self, c: np.ndarray, x: np.ndarray, lr: float) -> float:
        if self.objective == "neg":
            negs = np.minimum(np.searchsorted(self._neg_cdf, self.rng.random((len(c), self.negative))), self.n - 1)
            loss, g_in, oi, og = neg_loss_grads(self.W_in, self.W_out, c, x, negs)
        else:
            loss, g_in, oi, og = hs_loss_grads(self.W_in, self.W_out, c, x, self.paths, self.codes, self.lens)
        scatter_sub(self.W_in, c, g_in, lr)
        scatter_sub(self.W_out, oi, og, lr)
        return float(loss)

    @property
    def embeddings(self) -> np.ndarray:
        return self.W_in
