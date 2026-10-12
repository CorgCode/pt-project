"""struc2vec (Ribeiro, Saverese, Figueiredo, KDD 2017): embeddings from structural identity.

For every node u and every k = 0..K, R_k(u) is the ring of nodes at distance exactly k and s(R_k(u)) its ordered
degree sequence (degrees in the whole window graph). Structural distance accumulates over layers

    f_k(u, v) = f_{k-1}(u, v) + g(s(R_k(u)), s(R_k(v))),     f_{-1} = 0,

with g the dynamic time warping distance and ground cost d(a, b) = max(a, b) / min(a, b) - 1. Layer k of a
multilayer graph has edge weights w_k(u, v) = exp(-f_k(u, v)); u_k is joined to u_{k+1} with weight
log(Gamma_k(u) + e), Gamma_k(u) = number of layer-k edges at u heavier than the layer average, and to u_{k-1}
with weight 1. A walk stays in its layer with probability `stay_prob` (q in the paper) and moves to a
neighbour proportionally to w_k; otherwise it changes layer up with probability w_up / (w_up + 1). The visited
node ids (layer dropped) form the corpus of a skip-gram model.

Optimisations from the paper, which keep it tractable:
  OPT1  degree sequences are compressed to (degree, count) pairs; DTW runs on the compressed sequences with cost
        d(a, b) * max(count_a, count_b), and results are memoised for identical sequence pairs;
  OPT2  f_k is computed only for pairs of nodes close in degree order (2*ceil(log2 n) candidates per node);
  OPT3  only K+1 layers (`layers`, default 3) are built.
An empty ring is compared as one node of degree 1/2, so a non-empty ring is always at positive distance from it.

Scope: only the hosts to be embedded (the cohort for the train window, the evaluable hosts for a later window)
become nodes of the multilayer graph; their degree sequences are computed on the FULL window graph. Rows of other
hosts are zero. The coordinates of different windows are fitted separately (not shared).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from benchmark.methods.base import GraphEmbeddingMethod
from benchmark.methods.graph_input import Graph
from benchmark.methods.sgns import SkipGram
from benchmark.methods.walks import _draw_neighbor_index

EMPTY_DEGREE = 0.5


# ------------------------------------------------------------------------------------ rings and DTW
def _gather_neighbors(g: Graph, nodes: np.ndarray) -> np.ndarray:
    lens = g.degree[nodes]
    offs = np.repeat(g.indptr[nodes] - np.r_[0, np.cumsum(lens)[:-1]], lens) + np.arange(lens.sum())
    return g.indices[offs]


def ring_sequences(g: Graph, focus: np.ndarray, K: int) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """seqs[k][i] = compressed degree sequence (values, counts) of the ring at distance k of focus[i]."""
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
    """DTW between two compressed degree sequences, cost d(x, y) * max(count_x, count_y)."""
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
    """OPT2: pairs of nodes within `per_side` positions of each other in degree order (unique, i < j)."""
    order = np.lexsort((np.arange(len(degree)), degree))
    pos = np.arange(len(order))
    a, b = [], []
    for d in range(1, per_side + 1):
        a.append(order[pos[:-d]]); b.append(order[pos[d:]])
    a, b = np.concatenate(a), np.concatenate(b)
    pairs = np.column_stack([np.minimum(a, b), np.maximum(a, b)])
    return np.unique(pairs, axis=0)


def structural_distances(seqs, pairs: np.ndarray) -> np.ndarray:
    """(K+1, P): f_k for every candidate pair, accumulated over layers with memoised DTW."""
    K1 = len(seqs)
    canon = []                                   # canonical sequence ids so identical sequences share one DTW
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


# ------------------------------------------------------------------------------------ multilayer graph and walks
class Multilayer:
    """States (layer k, node u) as k * n + u; same-layer edges weighted exp(-f_k); layer-change probabilities."""

    def __init__(self, n: int, pairs: np.ndarray, f: np.ndarray):
        K1 = f.shape[0]
        self.n, self.layers = n, K1
        w = np.exp(-f)                                                    # (K1, P)
        rows, cols, vals, gamma = [], [], [], np.zeros((K1, n))
        for k in range(K1):
            avg = w[k].mean() if w.shape[1] else 0.0
            heavy = w[k] > avg * (1.0 + 1e-9)          # tolerance: equal weights must not count as above their own mean
            for end in (0, 1):
                src, dst = pairs[:, end], pairs[:, 1 - end]
                rows.append(k * n + src); cols.append(k * n + dst); vals.append(w[k])
                np.add.at(gamma[k], src, heavy.astype(float))
        A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(K1 * n,) * 2)
        A.sort_indices()
        self.indptr, self.indices = A.indptr.astype(np.int64), A.indices.astype(np.int64)
        self.weights = A.data.astype(np.float64)
        self.cum = np.concatenate([[0.0], np.cumsum(self.weights)])
        self.up_weight = np.log(gamma + np.e)                             # log(Gamma_k(u) + e); down weight is 1
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
            raise ValueError("struc2vec: stay_prob must be in [0, 1] and layers >= 0")

    # the base class passes the hosts to embed so that only they become nodes of the multilayer graph
    def embed_graph(self, g: Graph, focus: np.ndarray | None = None) -> np.ndarray:
        p = self.params
        focus = np.flatnonzero(g.degree > 0) if focus is None else np.asarray(focus, dtype=np.int64)
        if len(focus) < 3:
            raise ValueError("struc2vec needs at least 3 nodes to embed")
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
