"""struc2vec (Ribeiro, Saverese, Figueiredo, KDD 2017): эмбеддинги по структурной роли.

Для вершины u и k = 0..K кольцо R_k(u) — вершины на расстоянии ровно k, s(R_k(u)) — их упорядоченные степени.
Структурное расстояние копится по слоям:

    f_k(u, v) = f_{k-1}(u, v) + g(s(R_k(u)), s(R_k(v))),     f_{-1} = 0,

g — DTW с ценой d(a, b) = max(a, b) / min(a, b) - 1. Слой k многослойного графа: веса w_k(u, v) = exp(-f_k(u, v));
переход вверх с весом log(Gamma_k(u) + e), вниз — с весом 1. Блуждание остаётся в слое с вероятностью stay_prob.
Посещённые вершины (без номера слоя) — корпус для skip-gram.

Оптимизации из статьи: сжатие последовательностей степеней в пары (степень, количество) с кэшем DTW (OPT1);
расстояния только для близких по степени пар (OPT2); только K+1 слоёв (OPT3, по умолчанию 3).

Вершины многослойного графа — только нужные хосты (когорта или оцениваемые), степени — по ПОЛНОМУ графу окна.
Строки остальных хостов нулевые. Окна обучаются отдельно.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from methods.graph.base import GraphEmbeddingMethod
from methods.graph.graph_input import Graph
from methods.graph.sgns import SkipGram
from methods.graph.walks import _draw_neighbor_index

EMPTY_DEGREE = 0.5


# ------------------------------------------------------------------------------------ кольца и DTW
def _gather_neighbors(g: Graph, nodes: np.ndarray) -> np.ndarray:
    lens = g.degree[nodes]
    offs = np.repeat(g.indptr[nodes] - np.r_[0, np.cumsum(lens)[:-1]], lens) + np.arange(lens.sum())
    return g.indices[offs]


def ring_sequences(g: Graph, focus: np.ndarray, K: int) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """seqs[k][i] — сжатая последовательность степеней кольца на расстоянии k от focus[i]."""
    deg = g.degree
    seqs = [[None] * len(focus) for _ in range(K + 1)]
    visited = np.zeros(g.n, dtype=bool)
    for i, u in enumerate(focus):
        ring, touched = np.array([u]), [np.array([u])]
        visited[u] = True
        for k in range(K + 1):
            v, c = np.unique(deg[ring], return_counts=True) if len(ring) else (np.array([]), np.array([]))
            seqs[k][i] = (v.astype(float), c.astype(float))
            if k == K:
                break
            nb = _gather_neighbors(g, ring) if len(ring) else np.array([], dtype=np.int64)
            nb = np.unique(nb[~visited[nb]])
            visited[nb] = True
            touched.append(nb)
            ring = nb
        for t in touched:
            visited[t] = False
    return seqs


def dtw(a: np.ndarray, ca: np.ndarray, b: np.ndarray, cb: np.ndarray) -> float:
    """DTW двух сжатых последовательностей, цена d(x, y) * max(count_x, count_y)."""
    if len(a) == 0 and len(b) == 0:
        return 0.0
    if len(a) == 0:
        a, ca = np.array([EMPTY_DEGREE]), np.array([1.0])
    if len(b) == 0:
        b, cb = np.array([EMPTY_DEGREE]), np.array([1.0])
    hi, lo = np.maximum.outer(a, b), np.minimum.outer(a, b)
    C = (hi / lo - 1.0) * np.maximum.outer(ca, cb)
    n, m = C.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        Di, Dp, Ci = D[i], D[i - 1], C[i - 1]
        for j in range(1, m + 1):
            Di[j] = Ci[j - 1] + min(Dp[j], Dp[j - 1], Di[j - 1])
    return float(D[n, m])


def candidate_pairs(degree: np.ndarray, per_side: int) -> np.ndarray:
    """OPT2: пары вершин не дальше per_side позиций друг от друга по степени (уникальные, i < j)."""
    order = np.lexsort((np.arange(len(degree)), degree))
    pos = np.arange(len(order))
    a, b = [], []
    for d in range(1, per_side + 1):
        a.append(order[pos[:-d]]); b.append(order[pos[d:]])
    a, b = np.concatenate(a), np.concatenate(b)
    pairs = np.column_stack([np.minimum(a, b), np.maximum(a, b)])
    return np.unique(pairs, axis=0)


def structural_distances(seqs, pairs: np.ndarray) -> np.ndarray:
    """(K+1, P): f_k для каждой пары-кандидата, накапливаются по слоям, DTW кэшируется."""
    K1 = len(seqs)
    canon = []                                   # одинаковые последовательности получают один id и один DTW
    for k in range(K1):
        table, ids = {}, np.empty(len(seqs[k]), dtype=np.int64)
        for i, (v, c) in enumerate(seqs[k]):
            ids[i] = table.setdefault((v.tobytes(), c.tobytes()), len(table))
        canon.append((ids, table))
    f = np.zeros((K1, len(pairs)))
    run = np.zeros(len(pairs))
    for k in range(K1):
        ids = canon[k][0]
        cache: dict[tuple[int, int], float] = {}
        gk = np.empty(len(pairs))
        for t, (i, j) in enumerate(pairs):
            key = (min(ids[i], ids[j]), max(ids[i], ids[j]))
            if key not in cache:
                cache[key] = dtw(*seqs[k][i], *seqs[k][j])
            gk[t] = cache[key]
        run = run + gk
        f[k] = run
    return f


# ------------------------------------------------------------------------------------ многослойный граф и блуждания
class Multilayer:
    """Состояния (слой k, вершина u) = k * n + u; веса в слое exp(-f_k); вероятности смены слоя."""

    def __init__(self, n: int, pairs: np.ndarray, f: np.ndarray):
        K1 = f.shape[0]
        self.n, self.layers = n, K1
        w = np.exp(-f)                                                    # (K1, P)
        rows, cols, vals, gamma = [], [], [], np.zeros((K1, n))
        for k in range(K1):
            avg = w[k].mean() if w.shape[1] else 0.0
            heavy = w[k] > avg * (1.0 + 1e-9)          # допуск: равные веса не должны считаться выше своего же среднего
            for end in (0, 1):
                src, dst = pairs[:, end], pairs[:, 1 - end]
                rows.append(k * n + src); cols.append(k * n + dst); vals.append(w[k])
                np.add.at(gamma[k], src, heavy.astype(float))
        A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(K1 * n,) * 2)
        A.sort_indices()
        self.indptr, self.indices = A.indptr.astype(np.int64), A.indices.astype(np.int64)
        self.weights = A.data.astype(np.float64)
        self.cum = np.concatenate([[0.0], np.cumsum(self.weights)])
        self.up_weight = np.log(gamma + np.e)                             # log(Gamma_k(u) + e); вес вниз — 1
        self.p_up = self.up_weight / (self.up_weight + 1.0)
        self.gamma = gamma


def structural_walks(ml: Multilayer, num_walks: int, walk_length: int, stay_prob: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, K1 = ml.n, ml.layers
    out = []
    for _ in range(num_walks):
        u = rng.permutation(n).astype(np.int64)
        k = np.zeros(n, dtype=np.int64)
        walk = np.empty((n, walk_length), dtype=np.int64)
        walk[:, 0] = u
        for s in range(1, walk_length):
            stay = rng.random(n) < stay_prob
            state = k * n + u
            has = ml.indptr[state + 1] > ml.indptr[state]
            stay &= has
            nxt = np.where(stay, ml.indices[_draw_neighbor_index(ml, state, rng)] % n, u)
            up = np.where(k == 0, True, np.where(k == K1 - 1, False, rng.random(n) < ml.p_up[k, u]))
            if K1 == 1:
                up = np.zeros(n, dtype=bool)
            k = np.where(stay, k, np.where(up, np.minimum(k + 1, K1 - 1), np.maximum(k - 1, 0)))
            u = nxt
            walk[:, s] = u
        out.append(walk)
    return np.concatenate(out)


class Struc2Vec(GraphEmbeddingMethod):
    name = "struc2vec"
    uses_focus = True
    description = "multilayer structural-similarity graph (DTW of ring degree sequences) + biased walks + skip-gram"
    reference = "Ribeiro et al., struc2vec: Learning Node Representations from Structural Identity, KDD 2017"
    default_params = {"layers": 3, "stay_prob": 0.3, "walk_length": 80, "num_walks": 10, "window": 10, "epochs": 1,
                      "lr": 0.025, "objective": "hs", "negative": 5, "batch_size": 512, "weight": "log1p",
                      "candidates_per_side": None}
    fast_params = {"layers": 2, "num_walks": 2, "walk_length": 10, "window": 3}
    overlap = {"neighbors": "weak: degree structure correlates with behaviour",
               "edges": "weak: structural identity, not specific peers, so link persistence is not encoded",
               "pairs": "weak: shared peers are not encoded (structure ignores identity)"}

    def __init__(self, dim: int = 16, seed: int = 0, **params):
        super().__init__(dim, seed, **params)
        if not 0.0 <= self.params["stay_prob"] <= 1.0 or self.params["layers"] < 0:
            raise ValueError("struc2vec: stay_prob в [0, 1], layers >= 0")

    # базовый класс передаёт нужные хосты — только они становятся вершинами многослойного графа
    def embed_graph(self, g: Graph, focus: np.ndarray | None = None) -> np.ndarray:
        p = self.params
        focus = np.flatnonzero(g.degree > 0) if focus is None else np.asarray(focus, dtype=np.int64)
        if len(focus) < 3:
            raise ValueError("struc2vec нужно хотя бы 3 вершины")
        K = p["layers"]
        seqs = ring_sequences(g, focus, K)
        per_side = p["candidates_per_side"] or int(np.ceil(np.log2(len(focus))))
        pairs = candidate_pairs(g.degree[focus], min(per_side, len(focus) - 1))
        f = structural_distances(seqs, pairs)
        ml = Multilayer(len(focus), pairs, f)
        walks = structural_walks(ml, p["num_walks"], p["walk_length"], p["stay_prob"], self.seed)
        freq = np.bincount(walks.ravel(), minlength=len(focus)).astype(float)
        sg = SkipGram(len(focus), self.dim, objective=p["objective"], window=p["window"], negative=p["negative"],
                      lr=p["lr"], epochs=p["epochs"], batch_size=p["batch_size"], seed=self.seed, node_freq=freq)
        emb = np.zeros((g.n, self.dim))
        emb[focus] = sg.fit(walks).embeddings
        self.n_candidate_pairs = len(pairs)
        return emb

    def info(self):
        out = super().info()
        out["n_candidate_pairs"] = getattr(self, "n_candidate_pairs", None)
        return out


METHODS = {"struc2vec": Struc2Vec}
