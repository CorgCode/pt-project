"""Устойчивость кластеров хостов между окнами train и test (issue #21).

Метки кластеров HDBSCAN произвольны, поэтому напрямую их сравнивать нельзя.
Сравниваются пары хостов: ARI / NMI по jointly кластерированным хостам и Jaccard
множеств пар «в одном кластере». Дополнительно показывается, насколько результат
отличается от случайного эмбеддинга ``random``, оценённого тем же конвейером.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from evaluation.clustering import (DEFAULT_METRIC, DEFAULT_MIN_CLUSTER_SIZE,
                                   DEFAULT_MIN_SAMPLES, cluster_embeddings)

COLUMNS = [
    "status", "n_common_hosts", "common_clustered_hosts", "ari_train_test", "nmi_train_test",
    "pair_jaccard", "train_clusters", "test_clusters", "train_noise_fraction", "test_noise_fraction",
    "train_silhouette", "test_silhouette", "delta_ari_vs_random", "delta_nmi_vs_random",
    "delta_jaccard_vs_random",
]


def _same_cluster_pair_count(labels: np.ndarray) -> int:
    """Сколько пар хостов оказались в одном кластере (шум не считается кластером)."""
    valid = labels[labels != -1]
    if not len(valid):
        return 0
    counts = np.unique(valid, return_counts=True)[1]
    return int(np.sum(counts * (counts - 1) // 2))


def _pair_jaccard(train: np.ndarray, test: np.ndarray) -> float:
    """Доля общих «однокластерных» пар среди всех пар, отнесённых хоть к одному разбиению."""
    train_pairs = _same_cluster_pair_count(train)
    test_pairs = _same_cluster_pair_count(test)
    valid = (train != -1) & (test != -1)
    if valid.any():
        pairs = np.column_stack((train[valid], test[valid]))
        _, counts = np.unique(pairs, axis=0, return_counts=True)
        intersection = int(np.sum(counts * (counts - 1) // 2))
    else:
        intersection = 0
    union = train_pairs + test_pairs - intersection
    return 0.0 if union == 0 else float(intersection / union)


def clustering_consistency(
    embeddings: dict[str, dict[str, pd.DataFrame]],
    *,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int | None = DEFAULT_MIN_SAMPLES,
    metric: str = DEFAULT_METRIC,
) -> pd.DataFrame:
    """Сравнить кластеризации train и test для каждого метода; метод -> метрики устойчивости.

    Методы без обоих окон, с битыми host_id или слишком маленькой общей когортой
    получают строку со статусом вместо ошибки: один метод не должен ронять весь прогон.
    """
    rows: list[dict] = []
    for method, windows in sorted(embeddings.items()):
        base = {"method": method}
        if "train" not in windows or "test" not in windows:
            rows.append({**base, "status": "нет эмбеддингов для обоих окон"})
            continue
        train, test = windows["train"], windows["test"]
        if not train.index.is_unique or not test.index.is_unique:
            rows.append({**base, "status": "недоступно: повторяющиеся host_id"})
            continue
        common = train.index.intersection(test.index).sort_values()
        if len(common) < min_cluster_size:
            rows.append({**base, "status": "недостаточно общих хостов", "n_common_hosts": int(len(common))})
            continue
        try:
            train_details, train_assign = cluster_embeddings(
                train.loc[common], min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric)
            test_details, test_assign = cluster_embeddings(
                test.loc[common], min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric)
        except ValueError as exc:
            rows.append({**base, "status": f"недоступно: {exc}", "n_common_hosts": int(len(common))})
            continue

        train_labels = train_assign.loc[common, "cluster"].to_numpy(dtype=int)
        test_labels = test_assign.loc[common, "cluster"].to_numpy(dtype=int)
        jointly = (train_labels != -1) & (test_labels != -1)
        joint_train, joint_test = train_labels[jointly], test_labels[jointly]
        if len(joint_train) >= 2 and len(np.unique(joint_train)) >= 2 and len(np.unique(joint_test)) >= 2:
            ari = float(adjusted_rand_score(joint_train, joint_test))
            nmi = float(normalized_mutual_info_score(joint_train, joint_test))
        else:
            ari = nmi = 0.0

        rows.append({
            **base,
            "status": "ok",
            "n_common_hosts": int(len(common)),
            "common_clustered_hosts": int(jointly.sum()),
            "ari_train_test": ari,
            "nmi_train_test": nmi,
            "pair_jaccard": _pair_jaccard(train_labels, test_labels),
            "train_clusters": train_details["n_clusters"],
            "test_clusters": test_details["n_clusters"],
            "train_noise_fraction": train_details["noise_fraction"],
            "test_noise_fraction": test_details["noise_fraction"],
            "train_silhouette": train_details["silhouette"],
            "test_silhouette": test_details["silhouette"],
        })

    if not rows:
        return pd.DataFrame(columns=["method", *COLUMNS]).set_index("method")
    result = pd.DataFrame(rows).set_index("method")
    for column in COLUMNS:
        if column not in result.columns:
            result[column] = np.nan
    result = result[COLUMNS]

    if "random" in result.index and result.at["random", "status"] == "ok":
        valid = result["status"] == "ok"
        for out_col, base_col in (("delta_ari_vs_random", "ari_train_test"),
                                  ("delta_nmi_vs_random", "nmi_train_test"),
                                  ("delta_jaccard_vs_random", "pair_jaccard")):
            chance = result.at["random", base_col]
            if pd.notna(chance):
                result.loc[valid, out_col] = result.loc[valid, base_col] - chance
    return result
