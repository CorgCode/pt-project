"""RolX (Henderson et al., KDD 2012): role discovery from recursive structural features.

1. ReFeX (Henderson et al., KDD 2011) builds structural features without node attributes:
   * base features of every node v: degree, strength (weighted degree), edges inside its egonet
     (deg(v) + triangles(v)) and edges leaving its egonet (sum_{u in N(v)} deg(u) - 2 triangles(v) - deg(v));
   * recursive features: at each round the SUM and the MEAN over the neighbours of every feature added in the
     previous round (so round k describes the k-hop structure);
   * pruning: features are binned logarithmically (vertical binning, fraction p of the remaining nodes per bin) and
     a candidate is dropped if some kept feature differs from it by at most `s` bins on every node; constant
     features are dropped; rounds stop when nothing new survives (or after `rounds`).
2. Roles: the feature matrix (log1p, scaled per column to [0, 1]) is factorised V ~ G F by non-negative matrix
   factorisation with the generalised KL divergence (sklearn, deterministic NNDSVD initialisation). The rows of G
   (node-role memberships, here L1-normalised) are the embedding; `n_roles` defaults to the embedding dimension.
   `n_roles="mdl"` selects the number of roles in [2, dim] by minimum description length
   (model cost b * r * (n + f) bits with b = log2(1/precision), plus the generalised-KL error counted in units of
   that precision, i.e. a model entry and a reconstruction error are measured on the same scale). On clean graphs
   with few distinct structural features the model cost dominates and MDL favours the minimum number of roles, so it
   is an option, not the default.

Nothing in the algorithm uses node identity: nodes with the same structural role get similar vectors even in
different connected components. Proximity methods such as DeepWalk have no mechanism for that (they only see
co-occurrence within a component; any role signal they carry is incidental, e.g. via node frequency/degree). Coordinates (the roles) are
fitted per window, so they are not shared between windows.
"""
from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

from benchmark.methods.base import GraphEmbeddingMethod
from benchmark.methods.graph_input import Graph

BASE_NAMES = ["degree", "strength", "ego_edges", "ego_out_edges"]


def _binary(g: Graph) -> sp.csr_matrix:
    a = g.adjacency().copy()
    a.data = np.ones_like(a.data)
    return a


def triangles(B: sp.csr_matrix, chunk: int = 2000) -> np.ndarray:
    """Number of triangles through each node (row-chunked so hubs do not blow up memory)."""
    out = np.zeros(B.shape[0])
    for s in range(0, B.shape[0], chunk):
        rows = B[s:s + chunk]
        out[s:s + chunk] = np.asarray((rows @ B).multiply(rows).sum(axis=1)).ravel() / 2.0
    return out


def base_features(g: Graph) -> np.ndarray:
    """(n, 4): degree, strength, egonet internal edges, egonet outgoing edges."""
    B = _binary(g)
    deg = np.asarray(B.sum(axis=1)).ravel()
    tri = triangles(B)
    nbr_deg = B @ deg
    return np.column_stack([deg, g.strength, deg + tri, nbr_deg - 2 * tri - deg])


def aggregate(B: sp.csr_matrix, F: np.ndarray) -> np.ndarray:
    """[sum over neighbours, mean over neighbours] of every column of F."""
    deg = np.asarray(B.sum(axis=1)).ravel()
    s = B @ F
    return np.hstack([s, s / np.maximum(deg, 1.0)[:, None]])


def log_bin(x: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Vertical logarithmic binning: bin 0 holds the lowest fraction p of nodes, bin 1 fraction p of the rest, ...

    Ties share a bin (mid-rank percentile), so identical values are never split.
    """
    n = len(x)
    order = np.argsort(x, kind="stable")
    sx = x[order]
    first = np.searchsorted(sx, sx, side="left")
    last = np.searchsorted(sx, sx, side="right")
    c = ((first + last) / 2.0) / n                                  # mid-rank percentile in (0, 1)
    bins = np.empty(n, dtype=np.int64)
    bins[order] = np.floor(np.log(1.0 - c) / np.log(1.0 - p)).astype(np.int64)
    return bins


def refex(g: Graph, rounds: int = 3, s: int = 0, bin_p: float = 0.5, max_features: int = 200) -> tuple[np.ndarray, list[str]]:
    """Recursive structural features with pruning. Returns (n, f) float matrix and feature names."""
    B = _binary(g)
    cand = base_features(g)
    names = list(BASE_NAMES)
    kept_cols: list[np.ndarray] = []
    kept_bins: list[np.ndarray] = []
    kept_names: list[str] = []

    def accept(M: np.ndarray, nms: list[str]) -> list[int]:
        added = []
        for j in range(M.shape[1]):
            col = M[:, j]
            if np.ptp(col) == 0 or len(kept_cols) >= max_features:
                continue                                                # constant (or capacity reached)
            b = log_bin(col, bin_p)
            if any(np.abs(b - kb).max() <= s for kb in kept_bins):
                continue                                                # similar to a kept feature
            kept_cols.append(col); kept_bins.append(b); kept_names.append(nms[j]); added.append(j)
        return added

    added = accept(cand, names)
    for r in range(1, rounds + 1):
        if not added:
            break
        prev, prev_names = cand[:, added], [names[j] for j in added]
        cand = aggregate(B, prev)
        names = [f"sum({n})" for n in prev_names] + [f"mean({n})" for n in prev_names]
        added = accept(cand, names)
    if not kept_cols:
        raise ValueError("ReFeX produced no non-constant features (is the graph regular?)")
    return np.column_stack(kept_cols), kept_names


def prepare_matrix(F: np.ndarray) -> np.ndarray:
    """log1p and per-column scaling to [0, 1]: non-negative input for NMF with comparable scales."""
    V = np.log1p(np.maximum(F, 0.0))
    return V / np.maximum(V.max(axis=0), 1e-12)


def nmf_roles(V: np.ndarray, r: int, seed: int, max_iter: int = 400) -> np.ndarray:
    r = min(r, min(V.shape))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = NMF(n_components=r, init="nndsvda", solver="mu", beta_loss="kullback-leibler", max_iter=max_iter,
                    random_state=seed)
        G = model.fit_transform(V)
    return G


def kl_bits(V: np.ndarray, R: np.ndarray) -> float:
    """Generalised KL divergence sum(V log(V/R) - V + R) in bits."""
    R = np.maximum(R, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(V > 0, V * np.log(V / R), 0.0)
    return float((t - V + R).sum() / np.log(2.0))


def mdl_roles(V: np.ndarray, rmin: int, rmax: int, seed: int, precision: float = 1.0 / 64) -> tuple[int, dict]:
    """Number of roles in [rmin, rmax] minimising model cost + reconstruction error (both in bits)."""
    n, f = V.shape
    b = np.log2(1.0 / precision)
    costs = {}
    for r in range(rmin, min(rmax, min(n, f)) + 1):
        model = NMF(n_components=r, init="nndsvda", solver="mu", beta_loss="kullback-leibler", max_iter=400, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            G = model.fit_transform(V)
        # error in units of the precision: KL is homogeneous of degree 1, so KL(V/d || R/d) = KL(V || R) / d
        costs[r] = b * r * (n + f) + kl_bits(V, G @ model.components_) / precision
    return min(costs, key=costs.get), costs


class RolX(GraphEmbeddingMethod):
    name = "rolx"
    description = "ReFeX recursive structural features + NMF role memberships (structural-role embedding)"
    reference = "Henderson et al., RolX: Structural Role Extraction & Mining in Large Graphs, KDD 2012"
    default_params = {"n_roles": None, "rounds": 3, "similarity_bins": 0, "bin_fraction": 0.5, "max_features": 200,
                      "normalize": True, "nmf_iter": 400, "weight": "log1p"}
    fast_params = {"rounds": 2, "nmf_iter": 100}
    overlap = {"neighbors": "weak: degree/volume-like structure correlates with behaviour",
               "edges": "weak: structural roles, not specific peers, so link persistence is not encoded",
               "pairs": "weak: shared peers are not encoded (roles ignore identity)"}

    def embed_graph(self, g):
        p = self.params
        F, self.feature_names = refex(g, p["rounds"], p["similarity_bins"], p["bin_fraction"], p["max_features"])
        V = prepare_matrix(F)
        if p["n_roles"] == "mdl":
            r, self.mdl_costs = mdl_roles(V, 2, self.dim, self.seed)
        else:
            r = int(p["n_roles"] or self.dim)
        G = nmf_roles(V, r, self.seed, p["nmf_iter"])
        if p["normalize"]:
            G = G / np.maximum(G.sum(axis=1, keepdims=True), 1e-12)
        return G

    def info(self):
        out = super().info()
        out["n_structural_features"] = len(getattr(self, "feature_names", []))
        return out


METHODS = {"rolx": RolX}
