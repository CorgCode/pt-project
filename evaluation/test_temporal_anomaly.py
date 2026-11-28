import numpy as np
import pandas as pd

from evaluation.temporal_anomaly import detect_cluster_transitions, temporal_anomaly_report


def _stable_windows(rng, n_per_group=40):
    centers = np.array([[6.0, 0.0, 0.0, 0.0], [0.0, 6.0, 0.0, 0.0], [0.0, 0.0, 6.0, 0.0]])
    train_parts, test_parts = [], []
    for c in centers:
        train_parts.append(rng.normal(c, 0.2, size=(n_per_group, 4)))
        test_parts.append(rng.normal(c, 0.2, size=(n_per_group, 4)))
    index = [f"h{i}" for i in range(3 * n_per_group)]
    return pd.DataFrame(np.vstack(train_parts), index=index), pd.DataFrame(np.vstack(test_parts), index=index)


def test_stable_clusters_have_low_transition_rate():
    rng = np.random.default_rng(0)
    train, test = _stable_windows(rng)
    summary, transitions = detect_cluster_transitions(train, test, min_cluster_size=10, min_samples=5)

    assert summary["status"] == "ok"
    assert summary["n_common_hosts"] == len(train)
    assert summary["transition_rate"] < 0.2
    assert summary["n_stable"] > summary["n_anomalous_transitions"]
    assert set(transitions["transition_type"]).issubset(
        {"stable", "changed", "to_noise", "from_noise", "noise_noise"})
    assert transitions["is_anomalous"].dtype == bool


def test_cluster_change_is_detected():
    rng = np.random.default_rng(1)
    train, test = _stable_windows(rng, n_per_group=30)
    # Перемещаем группу хостов в другой центр — это подозрительный переход.
    moved = list(range(30))
    test.iloc[moved] = rng.normal([0.0, 0.0, 6.0, 0.0], 0.2, size=(30, 4))

    summary, transitions = detect_cluster_transitions(train, test, min_cluster_size=8, min_samples=3)

    # Перемещение 30 хостов в другой центр вызывает структурное изменение:
    # кластеры сливаются/пересопоставляются, поэтому аномальными могут быть
    # перемещённые хосты или хосты целевого кластера — проверяем факт перехода.
    assert summary["status"] == "ok"
    assert summary["n_anomalous_transitions"] >= 20
    assert summary["transition_rate"] > 0.1
    affected = set(transitions.index[transitions["is_anomalous"]])
    moved_ids = {f"h{i}" for i in moved}
    target_ids = {f"h{i}" for i in range(60, 90)}
    assert affected & (moved_ids | target_ids)


def test_report_compares_with_random_baseline():
    rng = np.random.default_rng(2)
    train, test = _stable_windows(rng)
    rnd_train = pd.DataFrame(rng.normal(size=train.shape), index=train.index)
    rnd_test = pd.DataFrame(rng.normal(size=test.shape), index=test.index)

    report = temporal_anomaly_report(
        {
            "behavioral": {"train": train, "test": test},
            "random": {"train": rnd_train, "test": rnd_test},
        },
        min_cluster_size=10,
        min_samples=5,
    )

    assert report.loc["behavioral", "status"] == "ok"
    assert report.loc["random", "status"] == "ok"
    # Случайные эмбеддинги дают не более стабильные сопоставления, чем устойчивые кластеры.
    assert report.loc["behavioral", "transition_rate"] <= report.loc["random", "transition_rate"] + 0.2
    assert "delta_transition_rate_vs_random" in report.columns
    assert pd.notna(report.loc["behavioral", "delta_transition_rate_vs_random"])


def test_missing_windows_and_duplicates_are_reported():
    rng = np.random.default_rng(3)
    train, test = _stable_windows(rng, n_per_group=12)
    dup = train.copy()
    dup.index = [dup.index[0], dup.index[0], *list(dup.index[2:])]

    report = temporal_anomaly_report(
        {"broken": {"train": train}, "dup": {"train": dup, "test": test}},
        min_cluster_size=8,
        min_samples=3,
    )

    assert report.loc["broken", "status"] == "нет эмбеддингов для обоих окон"
    assert "повторяющиеся host_id" in report.loc["dup", "status"]