"""Frozen log-derived neighborhood reference for cross-method host evaluation.

The file defines a BEHAVIORAL PROXY, not ground-truth functional roles.
Generate neighbors ONCE from reference features; use the same file for all
submissions. Prefer held-out features from a later time window (#45).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.knn_consistency import (
    neighborhood_metrics,
    summarize,
    top_k_neighbors,
)


COLUMNS = ("reference_window", "host_id", "neighbor_id", "rank", "split")


def save_frozen_neighbors(
    reference: pd.DataFrame,
    output: Path,
    *,
    reference_window: str,
    k_max: int = 20,
    split: str = "test",
) -> Path:
    """Generate reproducible top-k reference neighbors, excluding self.

    reference must use host_id as its index. Rows are sorted by host_id so that
    file order and tie behavior are independent of the provided row ordering.
    """
    if not reference_window.strip():
        raise ValueError("reference_window must be specified")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if reference.index.has_duplicates or reference.index.isna().any():
        raise ValueError("Reference host IDs must be nonmissing and unique")
    if len(reference) <= k_max or k_max < 1:
        raise ValueError("k_max must satisfy 1 <= k_max < n_hosts")
    if not all(isinstance(h, str) and h.strip() for h in reference.index):
        raise ValueError("Reference host IDs must be nonblank strings")
    if not np.isfinite(reference.to_numpy(dtype=float)).all():
        raise ValueError("Reference features must be finite")

    reference = reference.sort_index()
    hosts = reference.index.tolist()
    nearest = top_k_neighbors(reference.to_numpy(dtype=float), k_max)
    rows = [
        (reference_window, host, hosts[int(neighbor)], rank, split)
        for i, host in enumerate(hosts)
        for rank, neighbor in enumerate(nearest[i], start=1)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(output, index=False)
    return output


def read_frozen_neighbors(
    path: Path, hosts: list[str], *, k_max: int, split: str = "test"
) -> np.ndarray:
    """Validate full coverage; return neighbor indices in EXACT embedding order.

    Unjudged and partial neighbor lists are not silently treated as negatives.
    """
    if len(hosts) != len(set(hosts)) or not hosts:
        raise ValueError("Embedding hosts must be nonempty and unique")
    if k_max < 1 or k_max >= len(hosts):
        raise ValueError("k_max must satisfy 1 <= k_max < n_hosts")
    df = pd.read_csv(
        path,
        dtype={"reference_window": "string", "host_id": "string",
               "neighbor_id": "string", "split": "string", "rank": "string"},
    )
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"neighbors.csv missing columns: {sorted(missing)}")
    if df.empty or df[list(COLUMNS)].isna().any().any():
        raise ValueError("neighbors.csv has missing values or is empty")
    df = df.loc[df["split"] == split].copy()
    if df.empty:
        raise ValueError(f"neighbors.csv contains no {split} rows")
    if df["reference_window"].nunique() != 1:
        raise ValueError("Cannot combine multiple reference windows in one evaluation")
    if df["host_id"].str.strip().eq("").any() or df["neighbor_id"].str.strip().eq("").any():
        raise ValueError("Blank host ID in neighbors.csv")
    valid = set(hosts)
    if not set(df["host_id"]).issubset(valid) or not set(df["neighbor_id"]).issubset(valid):
        raise ValueError("neighbors.csv contains hosts outside embedding cohort")
    if set(df["host_id"]) != valid:
        raise ValueError("Missing reference neighbors for some embedding hosts")
    if (df["host_id"] == df["neighbor_id"]).any():
        raise ValueError("A host cannot be its own reference neighbor")
    rank = pd.to_numeric(df["rank"], errors="coerce")
    if rank.isna().any() or (rank < 1).any() or (rank != np.floor(rank)).any():
        raise ValueError("Neighbor ranks must be positive integers")
    df["rank"] = rank.astype(int)
    if df.duplicated(["host_id", "rank"]).any():
        raise ValueError("Duplicate neighbor ranks per host")
    if df.duplicated(["host_id", "neighbor_id"]).any():
        raise ValueError("Duplicate reference neighbors per host")

    lookup = {host: i for i, host in enumerate(hosts)}
    ordered = np.empty((len(hosts), k_max), dtype=int)
    for i, host in enumerate(hosts):
        rows = df[df["host_id"] == host].sort_values("rank")
        ranks = rows["rank"].tolist()
        if ranks != list(range(1, len(rows) + 1)) or len(rows) < k_max:
            raise ValueError(f"Missing/non-contiguous neighbor ranks for {host}")
        ordered[i] = [lookup[h] for h in rows["neighbor_id"].iloc[:k_max]]
    return ordered


def evaluate_frozen_neighbors(
    embeddings: pd.DataFrame, path: Path,
    ks: tuple[int, ...] = (5, 10, 20),
    *, split: str = "test", return_per_host: bool = False,
):
    """Compute exactly the existing consistency, Jaccard and nDCG metrics.

    With return_per_host=True also return {k: per-host DataFrame} (index = embedding order).
    """
    if embeddings.index.has_duplicates or not ks:
        raise ValueError("Embeddings must have unique hosts and ks must not be empty")
    highest = max(ks)
    if any(k < 1 or k >= len(embeddings) for k in ks):
        raise ValueError("Each k must be between 1 and n_hosts - 1")
    ref = read_frozen_neighbors(path, list(embeddings.index), k_max=highest, split=split)
    emb = embeddings.to_numpy(dtype=float)
    rows, per_host = [], {}
    for k in ks:
        eb_nn = top_k_neighbors(emb, k)
        ph = neighborhood_metrics(ref[:, :k], eb_nn)
        ph.index = embeddings.index
        per_host[k] = ph
        stats = summarize(ph)
        stats.index = pd.MultiIndex.from_product([[k], stats.index], names=["k", "metric"])
        rows.append(stats)
    summary = pd.concat(rows)
    return (summary, per_host) if return_per_host else summary
