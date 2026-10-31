"""Кластеризация эмбеддингов и оценка структуры кластеров."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import normalize


def cluster_embeddings(
    emb: pd.DataFrame,
    *,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    metric: str = "euclidean",
) -> tuple[dict, pd.DataFrame]:
    """Кластеризовать хосты; шум HDBSCAN обозначает меткой -1."""
    if not isinstance(emb, pd.DataFrame) or emb.empty:
        raise ValueError("emb должен быть непустым DataFrame")
    if emb.index.has_duplicates:
        raise ValueError("повторяющиеся host_id")
    if min_cluster_size < 2:
        raise ValueError("min_cluster_size должен быть >= 2")

    try:
        X = emb.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("эмбеддинги должны содержать только числа") from exc

    if X.ndim != 2 or X.shape[1] == 0 or not np.isfinite(X).all():
        raise ValueError("эмбеддинги должны быть конечной непустой матрицей")
    if len(X) < min_cluster_size:
        raise ValueError("число хостов меньше min_cluster_size")
    if metric == "cosine":
        if (np.linalg.norm(X, axis=1) == 0).any():
            raise ValueError("нулевой вектор: cosine не определён")
        X = normalize(X, norm="l2")
    elif metric != "euclidean":
        raise ValueError("metric должен быть 'euclidean' или 'cosine'")

    labels = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
    ).fit_predict(X)

    clustered = labels != -1
    n_clusters = len(set(labels[clustered]))
    n_clustered = int(clustered.sum())
    details = {
        "n_hosts": int(len(labels)),
        "n_clusters": int(n_clusters),
        "n_noise": int((~clustered).sum()),
        "noise_fraction": float((~clustered).mean()),
        "clustered_fraction": float(clustered.mean()),
        "silhouette": None,
        "davies_bouldin": None,
    }

    if n_clusters >= 2 and n_clustered > n_clusters:
        X_eval = X[clustered]
        y_eval = labels[clustered]
        details["silhouette"] = float(silhouette_score(X_eval, y_eval, metric=metric))
        details["davies_bouldin"] = float(davies_bouldin_score(X_eval, y_eval))

    assignments = pd.DataFrame(
        {"host_id": emb.index.astype(str), "cluster": labels}
    ).set_index("host_id")
    return details, assignments


def clustering_ari(
    assignments: pd.DataFrame,
    expected: pd.Series,
) -> float:
    """ARI по известным меткам; шум не считается отдельным кластером."""
    if not assignments.index.is_unique or not expected.index.is_unique:
        raise ValueError("повторяющиеся host_id")
    if set(assignments.index) != set(expected.index):
        raise ValueError("разные host_id в assignments и expected")

    labels = assignments.loc[expected.index, "cluster"].to_numpy()
    truth = expected.to_numpy()
    keep = labels != -1
    if keep.sum() < 2:
        raise ValueError("недостаточно хостов без шума для ARI")
    return float(adjusted_rand_score(truth[keep], labels[keep]))
