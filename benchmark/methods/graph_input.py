"""Weighted undirected host graph from canonical flows (CSR), shared by all graph methods.

Nodes are ALL hosts seen in the window's flows (cohort hosts and their peers), sorted by host_id;
edge weight between two hosts = log1p(number of flows in either direction) (or `count` / `binary`).
Self flows are dropped. Row neighbours are sorted, which makes edge lookup a binary search.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
import scipy.sparse as sp

WEIGHTS = ("log1p", "count", "binary")


@dataclass(frozen=True)
class Graph:
    nodes: np.ndarray            # host ids, sorted, dtype object
    indptr: np.ndarray           # (n+1,) int64
    indices: np.ndarray          # (2m,) int64, sorted within each row
    weights: np.ndarray          # (2m,) float64, > 0

    @property
    def n(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.indices) // 2

    @cached_property
    def degree(self) -> np.ndarray:
        return np.diff(self.indptr)

    @cached_property
    def strength(self) -> np.ndarray:
        return np.add.reduceat(self.weights, self.indptr[:-1]) * (self.degree > 0)

    @cached_property
    def cum(self) -> np.ndarray:
        """cum[i] = sum(weights[:i]); lets a neighbour be drawn proportionally to weight by binary search."""
        return np.concatenate([[0.0], np.cumsum(self.weights)])

    @cached_property
    def keys(self) -> np.ndarray:
        """Sorted int64 keys src * n + dst of all directed edges (for vectorised edge-existence tests)."""
        src = np.repeat(np.arange(self.n, dtype=np.int64), self.degree)
        return src * self.n + self.indices

    def has_edge(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        key = np.asarray(a, dtype=np.int64) * self.n + np.asarray(b, dtype=np.int64)
        pos = np.searchsorted(self.keys, key)
        return (pos < len(self.keys)) & (self.keys[np.minimum(pos, len(self.keys) - 1)] == key)

    def index_of(self, hosts) -> np.ndarray:
        """Row index of each host; KeyError (listing examples) if any host is not a node."""
        pos = np.searchsorted(self.nodes, np.asarray(list(hosts), dtype=object))
        pos = np.minimum(pos, self.n - 1)
        bad = [h for h, i in zip(hosts, pos) if self.nodes[i] != h]
        if bad:
            raise KeyError(f"{len(bad)} host(s) have no edge in this window's graph (e.g. {bad[:5]})")
        return pos

    def neighbors(self, v: int) -> np.ndarray:
        return self.indices[self.indptr[v]:self.indptr[v + 1]]

    def adjacency(self) -> sp.csr_matrix:
        return sp.csr_matrix((self.weights, self.indices, self.indptr), shape=(self.n, self.n))


def from_edges(src: np.ndarray, dst: np.ndarray, w: np.ndarray, nodes: np.ndarray) -> Graph:
    """Build from integer endpoint arrays (one row per undirected edge, no self loops, no duplicates)."""
    n = len(nodes)
    a = sp.coo_matrix((np.r_[w, w], (np.r_[src, dst], np.r_[dst, src])), shape=(n, n)).tocsr()
    a.sum_duplicates()
    a.sort_indices()
    return Graph(nodes=np.asarray(nodes, dtype=object), indptr=a.indptr.astype(np.int64),
                 indices=a.indices.astype(np.int64), weights=a.data.astype(np.float64))


def from_flows(flows: pd.DataFrame, weight: str = "log1p") -> Graph:
    if weight not in WEIGHTS:
        raise ValueError(f"weight must be one of {WEIGHTS}")
    f = flows[["src", "dst"]].astype(str)
    f = f[f["src"] != f["dst"]]
    if f.empty:
        raise ValueError("no non-self flows: cannot build a graph")
    nodes, inv = np.unique(np.concatenate([f["src"].to_numpy(), f["dst"].to_numpy()]), return_inverse=True)
    a, b = inv[:len(f)], inv[len(f):]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = lo.astype(np.int64) * len(nodes) + hi
    ukey, counts = np.unique(key, return_counts=True)
    w = {"log1p": np.log1p(counts), "count": counts.astype(float), "binary": np.ones(len(counts))}[weight]
    return from_edges(ukey // len(nodes), ukey % len(nodes), w, nodes)
