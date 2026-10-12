"""Random walks on a weighted graph: first-order (DeepWalk) and second-order biased (node2vec).

Second-order walk (Grover & Leskovec 2016). Having just traversed edge (t, v), the next node x is drawn
from the neighbours of v with probability proportional to  w(v, x) * alpha(t, x)  where

    alpha = 1/p   if x == t              (return)
            1     if x is adjacent to t  (stay at distance 1)
            1/q   otherwise              (move outward, distance 2)

Sampling is exact: a neighbour is proposed proportionally to w(v, x) (binary search on cumulative
weights) and accepted with probability alpha / max(1/p, 1, 1/q) (vectorised rejection sampling); the few
walks that exceed `max_rounds` rejections are completed by exact enumeration. p = q = 1 is the first-order
weighted walk. Everything is vectorised over walks, deterministic given the generator, and never leaves the
graph (every node on a walk has degree >= 1).
"""
from __future__ import annotations

import numpy as np

from benchmark.methods.graph_input import Graph


def _draw_neighbor_index(g: Graph, v: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Global CSR index of one neighbour of each v, drawn proportionally to edge weight."""
    start, end = g.indptr[v], g.indptr[v + 1]
    r = g.cum[start] + rng.random(len(v)) * (g.cum[end] - g.cum[start])
    return np.clip(np.searchsorted(g.cum, r, side="right") - 1, start, end - 1)


def _alpha(g: Graph, t: np.ndarray, x: np.ndarray, p: float, q: float) -> np.ndarray:
    out = np.full(len(x), 1.0 / q)
    out[g.has_edge(t, x)] = 1.0
    out[x == t] = 1.0 / p
    return out


def _exact_step(g: Graph, t: int, v: int, p: float, q: float, rng: np.random.Generator) -> int:
    nb = g.neighbors(v)
    w = g.weights[g.indptr[v]:g.indptr[v + 1]] * _alpha(g, np.full(len(nb), t), nb, p, q)
    return int(nb[np.searchsorted(np.cumsum(w), rng.random() * w.sum(), side="right").clip(0, len(nb) - 1)])


def second_order_step(g: Graph, t: np.ndarray, v: np.ndarray, p: float, q: float, rng: np.random.Generator,
                      max_rounds: int = 64) -> np.ndarray:
    """Next node for each walk that just traversed (t -> v): exact node2vec transition (see module docstring)."""
    a_max = max(1.0 / p, 1.0, 1.0 / q)
    pending = np.arange(len(v))
    chosen = np.empty(len(v), dtype=np.int64)
    for _ in range(max_rounds):
        x = g.indices[_draw_neighbor_index(g, v[pending], rng)]
        ok = rng.random(len(pending)) < _alpha(g, t[pending], x, p, q) / a_max
        chosen[pending[ok]] = x[ok]
        pending = pending[~ok]
        if not len(pending):
            break
    for i in pending:                                       # rare: exact enumeration
        chosen[i] = _exact_step(g, int(t[i]), int(v[i]), p, q, rng)
    return chosen


def random_walks(g: Graph, starts: np.ndarray, walk_length: int, *, p: float = 1.0, q: float = 1.0,
                 rng: np.random.Generator, max_rounds: int = 64) -> np.ndarray:
    """(len(starts), walk_length) int32 array; row i is a walk beginning at starts[i]."""
    if walk_length < 2:
        raise ValueError("walk_length must be >= 2")
    if p <= 0 or q <= 0:
        raise ValueError("p and q must be positive")
    starts = np.asarray(starts, dtype=np.int64)
    if (g.degree[starts] == 0).any():
        raise ValueError("a walk cannot start at an isolated node")
    walks = np.empty((len(starts), walk_length), dtype=np.int64)
    walks[:, 0] = starts
    walks[:, 1] = g.indices[_draw_neighbor_index(g, starts, rng)]
    first_order = p == 1.0 and q == 1.0
    for s in range(2, walk_length):
        t, v = walks[:, s - 2], walks[:, s - 1]
        walks[:, s] = (g.indices[_draw_neighbor_index(g, v, rng)] if first_order
                       else second_order_step(g, t, v, p, q, rng, max_rounds))
    return walks.astype(np.int32)


def walk_corpus(g: Graph, num_walks: int, walk_length: int, *, p: float = 1.0, q: float = 1.0,
                seed: int = 0, nodes: np.ndarray | None = None) -> np.ndarray:
    """`num_walks` passes over the nodes (a fresh random order each pass), as in DeepWalk / node2vec."""
    rng = np.random.default_rng(seed)
    pool = np.flatnonzero(g.degree > 0) if nodes is None else np.asarray(nodes)
    out = []
    for _ in range(num_walks):
        order = rng.permutation(pool)
        out.append(random_walks(g, order, walk_length, p=p, q=q, rng=rng))
    return np.concatenate(out)
