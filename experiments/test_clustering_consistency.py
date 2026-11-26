import numpy as np
import pandas as pd

from experiments.clustering_consistency import clustering_consistency


def _stable_pair(rng, n_per_group=40):
    centers = np.array([[6.0, 0.0, 0.0, 0.0], [0.0, 6.0, 0.0, 0.0], [0.0, 0.0, 6.0, 0.0]])
    train_parts, test_parts = [], []
    for center in centers:
        train_parts.append(rng.normal(center, 0.25, size=(n_per_group, 4)))
        test_parts.append(rng.normal(center, 0.25, size=(n_per_group, 4)))
    index = [f"host-{i}" for i in range(3 * n_per_group)]
    return (pd.DataFrame(np.vstack(train_parts), index=index),
            pd.DataFrame(np.vstack(test_parts), index=index))


def test_consistency_beats_random_baseline():
    rng = np.random.default_rng(10)
    train, test = _stable_pair(rng)
    random_train = pd.DataFrame(rng.normal(size=train.shape), index=train.index)
    random_test = pd.DataFrame(rng.normal(size=test.shape), index=test.index)

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
    assert result.loc["behavioral", "ari_train_test"] > result.loc["random", "ari_train_test"]
    assert result.loc["behavioral", "delta_ari_vs_random"] > 0
    assert result.loc["behavioral", "delta_jaccard_vs_random"] >= 0
    assert "host_id" not in result.columns


def test_consistency_reports_missing_windows_and_small_overlap():
    rng = np.random.default_rng(11)
    train, test = _stable_pair(rng, n_per_group=8)
    test.index = [f"other-{i}" for i in range(len(test))]

    result = clustering_consistency(
        {"broken": {"train": train}, "disjoint": {"train": train, "test": test}},
        min_cluster_size=10,
        min_samples=5,
    )

    assert result.loc["broken", "status"] == "нет эмбеддингов для обоих окон"
    assert result.loc["disjoint", "status"] == "недостаточно общих хостов"
    assert result.loc["disjoint", "n_common_hosts"] == 0


def test_consistency_handles_duplicate_host_ids_without_crashing_run():
    rng = np.random.default_rng(12)
    train, test = _stable_pair(rng, n_per_group=12)
    dup = train.copy()
    dup.index = [dup.index[0], dup.index[0], *list(dup.index[2:])]

    result = clustering_consistency(
        {"dup": {"train": dup, "test": test}, "ok": {"train": train, "test": test}},
        min_cluster_size=8,
        min_samples=3,
    )

    assert "повторяющиеся host_id" in result.loc["dup", "status"]
    assert result.loc["ok", "status"] == "ok"
