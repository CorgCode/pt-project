"""k-NN consistency / neighborhood preservation for host embeddings.

Idea: for every host take its top-k neighbours under an independent reference
similarity (built from raw behaviour) and its top-k neighbours in embedding
space, then compare the two sets. Higher overlap => the embedding preserves
behavioural similarity.

Leakage rule: the reference must not be the evaluated embedding representation.
Behavioural features may be reused carefully if the method trained on related
features; prefer held-out feature groups, different time windows, or independent
similarity views (see evaluation/README_knn_consistency.md).

Public API (plug-in friendly):
    build_reference_features(flows, hosts=None) -> pd.DataFrame
    evaluate_embedding(embedding, reference, ks=(5, 10, 20)) -> pd.DataFrame
    neighborhood_metrics(ref_nn, emb_nn) -> pd.DataFrame (per host)
    summarize(per_host) -> pd.DataFrame
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# 1. Reference similarity (independent of any embedding)
# --------------------------------------------------------------------------

def _row_normalize(x: pd.DataFrame) -> pd.DataFrame:
    s = x.sum(axis=1).replace(0, 1.0)
    return x.div(s, axis=0)


def build_reference_features(
    flows: pd.DataFrame,
    hosts: list | None = None,
    top_ports: int = 20,
    weights: dict | None = None,
) -> pd.DataFrame:
    """Hand-crafted behavioural profile per host, one row per host.

    `flows` columns: src, dst, dst_port, proto, bytes, packets, ts (datetime or
    unix seconds). Blocks (each scaled to equal norm, then weighted):
      volume   - log1p of flows/bytes/packets, out and in
      peers    - log1p of distinct out/in peers (degree only, NOT peer identity)
      ratio    - out/(out+in) share for flows and bytes
      ports    - distribution over the top-N destination ports (as server+client)
      proto    - distribution over protocols
      hourly   - 24-bin activity histogram (temporal profile)
    Leakage: the reference must never be the evaluated embedding itself (or a
    copy / linear transform of it). Behavioural features may overlap with what the
    method was trained on only with care; preferred options are held-out feature
    groups, a different time window, or an independent similarity view. Peer
    *identity* is excluded by default for this reason: graph embeddings
    (node2vec etc.) trained on the same communication graph would reproduce it.
    """
    f = flows.copy()
    if not np.issubdtype(f["ts"].dtype, np.datetime64):
        f["ts"] = pd.to_datetime(f["ts"], unit="s")
    f["hour"] = f["ts"].dt.hour
    if hosts is None:
        hosts = sorted(set(f["src"]) | set(f["dst"]))
    idx = pd.Index(hosts)

    out = f.groupby("src").agg(out_flows=("dst", "size"), out_bytes=("bytes", "sum"),
                               out_pkts=("packets", "sum"), out_peers=("dst", "nunique"))
    inn = f.groupby("dst").agg(in_flows=("src", "size"), in_bytes=("bytes", "sum"),
                               in_pkts=("packets", "sum"), in_peers=("src", "nunique"))
    base = out.join(inn, how="outer").reindex(idx).fillna(0.0)

    volume = np.log1p(base[["out_flows", "out_bytes", "out_pkts",
                            "in_flows", "in_bytes", "in_pkts"]])
    peers = np.log1p(base[["out_peers", "in_peers"]])
    ratio = pd.DataFrame({
        "flow_out_share": base.out_flows / (base.out_flows + base.in_flows).replace(0, 1),
        "byte_out_share": base.out_bytes / (base.out_bytes + base.in_bytes).replace(0, 1),
    }, index=idx)

    def dist(col, keep=None):
        both = pd.concat([f[["src", col]].rename(columns={"src": "h"}),
                          f[["dst", col]].rename(columns={"dst": "h"})])
        if keep is not None:
            both = both[both[col].isin(keep)]
        return _row_normalize(both.groupby(["h", col]).size().unstack(fill_value=0)
                              .reindex(idx).fillna(0.0))

    ports = dist("dst_port", f["dst_port"].value_counts().head(top_ports).index)
    proto = dist("proto")
    hourly = _row_normalize(pd.concat([
        f[["src", "hour"]].rename(columns={"src": "h"}),
        f[["dst", "hour"]].rename(columns={"dst": "h"})]).groupby(["h", "hour"]).size()
        .unstack(fill_value=0).reindex(index=idx, columns=range(24)).fillna(0.0))

    w = {"volume": 1.0, "peers": 1.0, "ratio": 1.0, "ports": 1.0, "proto": 0.5, "hourly": 1.0}
    w.update(weights or {})
    blocks = {"volume": volume, "peers": peers, "ratio": ratio,
              "ports": ports, "proto": proto, "hourly": hourly}
    parts = []
    for name, b in blocks.items():
        z = StandardScaler().fit_transform(b.values)
        z = np.nan_to_num(z)
        z = z / np.sqrt(max(z.shape[1], 1))          # equal total variance per block
        parts.append(pd.DataFrame(z * w[name], index=idx).add_prefix(f"{name}_"))
    return pd.concat(parts, axis=1)


# --------------------------------------------------------------------------
# 2. Neighbour search (self excluded)
# --------------------------------------------------------------------------

def top_k_neighbors(X: np.ndarray, k: int, metric: str = "cosine") -> np.ndarray:
    """(n, k) array of neighbour indices, sorted by similarity, self excluded."""
    X = np.asarray(X, dtype=float)
    n = len(X)
    if not 1 <= k < n:
        raise ValueError(f"need 1 <= k < n_hosts, got k={k}, n={n}")
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="brute").fit(X)
    idx = nn.kneighbors(X, return_distance=False)
    out = np.empty((n, k), dtype=int)
    for i, row in enumerate(idx):
        row = row[row != i]          # duplicates may push self out of position 0
        out[i] = row[:k]
    return out


# --------------------------------------------------------------------------
# 3. Metrics
# --------------------------------------------------------------------------

def neighborhood_metrics(ref_nn: np.ndarray, emb_nn: np.ndarray) -> pd.DataFrame:
    """Per-host Consistency@k, Jaccard@k and rank-aware nDCG@k.

    consistency = |R ∩ E| / k
    jaccard     = |R ∩ E| / |R ∪ E|
    ndcg        = DCG of embedding list with graded relevance (k - ref_rank) for
                  reference neighbours, 0 otherwise, normalised by the ideal DCG.
                  Rewards putting the closest reference neighbours first.
    """
    n, k = ref_nn.shape
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    ideal = (np.arange(k, 0, -1) * discounts).sum()
    cons, jac, ndcg = np.empty(n), np.empty(n), np.empty(n)
    for i in range(n):
        rel = {h: k - r for r, h in enumerate(ref_nn[i])}
        inter = len(rel.keys() & set(emb_nn[i]))
        cons[i] = inter / k
        jac[i] = inter / (2 * k - inter)
        gains = np.array([rel.get(h, 0) for h in emb_nn[i]], dtype=float)
        ndcg[i] = (gains * discounts).sum() / ideal
    return pd.DataFrame({"consistency": cons, "jaccard": jac, "ndcg": ndcg})


def summarize(per_host: pd.DataFrame) -> pd.DataFrame:
    return per_host.agg(["mean", "median", "std"]).T


def evaluate_embedding(
    embedding: pd.DataFrame | np.ndarray,
    reference: pd.DataFrame | np.ndarray,
    ks=(5, 10, 20),
    emb_metric: str = "cosine",
    ref_metric: str = "cosine",
    return_per_host: bool = False,
):
    """Compare an embedding to the reference over several k.

    Both inputs must have the same host order (DataFrames are aligned by index).
    Returns a summary DataFrame indexed by (k, metric) with mean/median/std,
    and optionally {k: per-host DataFrame}.
    """
    if isinstance(embedding, pd.DataFrame) and isinstance(reference, pd.DataFrame):
        reference = reference.loc[embedding.index]
        hosts = embedding.index
    else:
        hosts = None
    E, R = np.asarray(embedding, float), np.asarray(reference, float)
    if len(E) != len(R):
        raise ValueError("embedding and reference must cover the same hosts")

    rows, per_host = [], {}
    for k in ks:
        ph = neighborhood_metrics(top_k_neighbors(R, k, ref_metric),
                                  top_k_neighbors(E, k, emb_metric))
        if hosts is not None:
            ph.index = hosts
        per_host[k] = ph
        s = summarize(ph)
        s.index = pd.MultiIndex.from_product([[k], s.index], names=["k", "metric"])
        rows.append(s)
    summary = pd.concat(rows)
    return (summary, per_host) if return_per_host else summary
