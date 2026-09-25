"""Оценка качества представлений и кластеризации.

Без разметки (основной путь, т.к. роли узлов заранее неизвестны):
  * silhouette, Davies–Bouldin, Calinski–Harabasz по кластерам;
  * k-NN «согласованность» — насколько устойчиво соседи попадают в один кластер.

По разметке (только для валидации на синтетике, в обучении не используется):
  * ARI / NMI между кластерами и истинными ролями;
  * k-NN role purity — доля ближайших соседей той же роли (качество поиска похожих).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors


def unsupervised_metrics(emb: np.ndarray, labels: np.ndarray) -> dict:
    out = {}
    if len(set(labels)) < 2:
        return {"silhouette": float("nan"), "davies_bouldin": float("nan"),
                "calinski_harabasz": float("nan")}
    out["silhouette"] = float(silhouette_score(emb, labels))
    out["davies_bouldin"] = float(davies_bouldin_score(emb, labels))
    out["calinski_harabasz"] = float(calinski_harabasz_score(emb, labels))
    return out


def knn_cluster_consistency(emb: np.ndarray, labels: np.ndarray, k: int = 8) -> float:
    """Доля ближайших соседей, попадающих в тот же кластер (устойчивость)."""
    n = emb.shape[0]
    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(emb)
    _, idx = nn.kneighbors(emb)
    same = 0
    total = 0
    for i in range(n):
        for j in idx[i][1:]:
            same += int(labels[i] == labels[j])
            total += 1
    return float(same / total) if total else float("nan")


def knn_role_purity(emb_df: pd.DataFrame, roles: pd.Series, k: int = 8) -> float:
    """Доля ближайших соседей той же истинной роли (по размеченным хостам)."""
    devices = [d for d in emb_df.index if d in roles.index]
    if len(devices) < 3:
        return float("nan")
    sub = emb_df.loc[devices]
    role_vec = roles.loc[devices].values
    k = min(k, len(devices) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(sub.values)
    _, idx = nn.kneighbors(sub.values)
    same, total = 0, 0
    for i in range(len(devices)):
        for j in idx[i][1:]:
            same += int(role_vec[i] == role_vec[j])
            total += 1
    return float(same / total) if total else float("nan")


def supervised_metrics(labels: np.ndarray, emb_index, roles: pd.Series) -> dict:
    """ARI/NMI кластеров против истинных ролей (по размеченным хостам)."""
    lab = pd.Series(labels, index=emb_index)
    devices = [d for d in emb_index if d in roles.index]
    if len(devices) < 3:
        return {"ari": float("nan"), "nmi": float("nan")}
    y_true = roles.loc[devices].values
    y_pred = lab.loc[devices].values
    return {
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
    }


def evaluate_method(
    emb_df: pd.DataFrame,
    labels: np.ndarray,
    roles: pd.Series | None,
    k: int = 8,
) -> dict:
    res = unsupervised_metrics(emb_df.values, labels)
    res["knn_cluster_consistency"] = knn_cluster_consistency(emb_df.values, labels, k)
    if roles is not None:
        res.update(supervised_metrics(labels, emb_df.index, roles))
        res["knn_role_purity"] = knn_role_purity(emb_df, roles, k)
    return res
