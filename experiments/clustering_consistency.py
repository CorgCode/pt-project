"""Сравнение кластерных разбиений между временными окнами."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from evaluation.clustering import cluster_embeddings


def _same_cluster_pair_count(labels: np.ndarray) -> int:
    valid = labels[labels != -1]
    if not len(valid):
        return 0
    counts = np.unique(valid, return_counts=True)[1]
    return int(np.sum(counts * (counts - 1) // 2))


def _pair_jaccard(train: np.ndarray, test: np.ndarray) -> float:
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
    min_cluster_size: int = 5,
    min_samples: int = 5,
) -> pd.DataFrame:
    rows = []

    for method, windows in sorted(embeddings.items()):
        if "train" not in windows or "test" not in windows:
            rows.append({
                "method": method,
                "status": "нет embeddings для обоих окон",
            })
            continue

        train, test = windows["train"], windows["test"]
        if not train.index.is_unique or not test.index.is_unique:
            raise ValueError(f"{method}: повторяющиеся host_id")

        common = train.index.intersection(test.index, sort=False)
        if len(common) < min_cluster_size:
            rows.append({
                "method": method,
                "status": "недостаточно общих хостов",
                "n_common_hosts": int(len(common)),
            })
            continue

        try:
            train_details, train_assignments = cluster_embeddings(
                train.loc[common],
                metric="cosine",
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
            )
            test_details, test_assignments = cluster_embeddings(
                test.loc[common],
                metric="cosine",
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
            )
        except ValueError as exc:
            rows.append({
                "method": method,
                "status": f"недоступно: {exc}",
                "n_common_hosts": int(len(common)),
            })
            continue

        train_labels = train_assignments.loc[common, "cluster"].to_numpy()
        test_labels = test_assignments.loc[common, "cluster"].to_numpy()

        jointly_clustered = (train_labels != -1) & (test_labels != -1)
        joint_train = train_labels[jointly_clustered]
        joint_test = test_labels[jointly_clustered]

        if (
            len(joint_train) >= 2
            and len(np.unique(joint_train)) >= 2
            and len(np.unique(joint_test)) >= 2
        ):
            ari = float(adjusted_rand_score(joint_train, joint_test))
        else:
            ari = 0.0

        rows.append({
            "method": method,
            "status": "ok",
            "n_common_hosts": int(len(common)),
            "common_clustered_hosts": int(jointly_clustered.sum()),
            "ari_train_test": ari,
            "pair_jaccard": _pair_jaccard(train_labels, test_labels),
            "train_clusters": train_details["n_clusters"],
            "test_clusters": test_details["n_clusters"],
            "train_noise_fraction": train_details["noise_fraction"],
            "test_noise_fraction": test_details["noise_fraction"],
            "train_clustered_fraction": train_details["clustered_fraction"],
            "test_clustered_fraction": test_details["clustered_fraction"],
            "train_silhouette": train_details["silhouette"],
            "test_silhouette": test_details["silhouette"],
        })

    result = pd.DataFrame(rows).set_index("method")
    for column in ("delta_ari_vs_random", "delta_jaccard_vs_random"):
        result[column] = np.nan

    if "random" in result.index and result.loc["random", "status"] == "ok":
        random_ari = result.loc["random", "ari_train_test"]
        random_jaccard = result.loc["random", "pair_jaccard"]
        valid = result["status"] == "ok"
        result.loc[valid, "delta_ari_vs_random"] = (
            result.loc[valid, "ari_train_test"] - random_ari
        )
        result.loc[valid, "delta_jaccard_vs_random"] = (
            result.loc[valid, "pair_jaccard"] - random_jaccard
        )

    return result
