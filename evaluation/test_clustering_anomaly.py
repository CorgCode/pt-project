import numpy as np
import pandas as pd

from evaluation.clustering import cluster_embeddings
from experiments.clustering_consistency import clustering_consistency


def test_mixed_features_and_embedded_outliers_are_handled():
    rng = np.random.default_rng(42)

    centers = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [5.0, 5.0, 0.0, 0.0],
        [0.0, 5.0, 5.0, 0.0],
    ])

    clouds = [
        rng.normal(loc=center, scale=0.25, size=(40, 4))
        for center in centers
    ]
    outliers = np.array([
        [0.1, 0.1, 5.0, 5.0],
        [5.0, 0.0, 5.0, 5.0],
        [2.5, 2.5, 2.5, 2.5],
    ])

    X = np.vstack(clouds + [outliers])
    emb = pd.DataFrame(
        X,
        index=[f"host-{i}" for i in range(len(X))]
    )

    details, assignments = cluster_embeddings(
        emb,
        min_cluster_size=8,
        min_samples=5,
        metric="cosine",
    )

    assert details["n_clusters"] >= 2
    assert len(assignments) == len(emb)
    assert 0.0 <= details["noise_fraction"] <= 1.0


def test_train_test_consistency_is_compared_with_independent_random_baseline():
    rng = np.random.default_rng(123)
    n_per_group = 50
    centers = np.array([
        [5.0, 0.0, 0.0, 0.0],
        [0.0, 5.0, 0.0, 0.0],
        [0.0, 0.0, 5.0, 0.0],
    ])

    train_parts = []
    test_parts = []

    for center in centers:
        train_parts.append(
            rng.normal(center, 0.2, size=(n_per_group, 4))
        )
        test_parts.append(
            rng.normal(center, 0.2, size=(n_per_group, 4))
        )

    train = pd.DataFrame(
        np.vstack(train_parts),
        index=[f"host-{i}" for i in range(3 * n_per_group)],
    )
    test = pd.DataFrame(
        np.vstack(test_parts),
        index=train.index,
    )

    random_train = pd.DataFrame(
        rng.normal(size=(len(train), 4)),
        index=train.index,
    )
    random_test = pd.DataFrame(
        rng.normal(size=(len(test), 4)),
        index=test.index,
    )

    result = clustering_consistency(
        {
            "behavioral": {"train": train, "test": test},
            "random": {"train": random_train, "test": random_test},
        },
        min_cluster_size=10,
        min_samples=5,
    )

    assert result.loc["behavioral", "status"] == "ok"
    assert result.loc["random", "status"] == "ok"
    assert result.loc["behavioral", "ari_train_test"] > (
        result.loc["random", "ari_train_test"]
    )
    assert result.loc["behavioral", "delta_ari_vs_random"] > 0
    assert "host_id" not in result.columns
    assert "cluster" not in result.columns
