
import numpy as np
import pandas as pd

from evaluation.temporal_anomaly import temporal_anomaly


def _flows(hosts, hour=1):
    rows = []
    timestamp = int(
        pd.Timestamp(
            f"2025-01-01 {hour:02d}:00:00",
            tz="UTC",
        ).timestamp()
    )

    for i, host in enumerate(hosts):
        peer = f"peer-{i}"
        change = (hour - 1) * (i + 1) * 10

        rows.append({
            "src": host,
            "dst": peer,
            "dst_port": 80,
            "proto": 6,
            "bytes": 100 + i + change,
            "packets": 2 + i + (hour - 1) * (i + 1),
            "ts": timestamp,
        })

        rows.append({
            "src": peer,
            "dst": host,
            "dst_port": 443,
            "proto": 6,
            "bytes": 50 + i + change // 2,
            "packets": 1 + i + (hour - 1) * (i + 1),
            "ts": timestamp,
        })

    return pd.DataFrame(rows)


def test_shared_coordinates_returns_temporal_scores():
    rng = np.random.default_rng(42)
    hosts = [f"h{i:03d}" for i in range(30)]

    train = rng.normal(size=(30, 8))
    test = train.copy()
    test[:5] += 2.0

    result = temporal_anomaly(
        pd.DataFrame(train, index=hosts),
        pd.DataFrame(test, index=hosts),
        _flows(hosts, 1),
        _flows(hosts, 2),
        hosts,
        shared_coordinate_system=True,
    )

    assert result["n_scored_hosts"] == 30
    assert result["top_k"] == 3
    assert result["rank_correlation"] is not None
    assert len(result["per_host"]) == 30


def test_inactive_hosts_are_excluded_not_zero_filled():
    rng = np.random.default_rng(7)
    hosts = [f"h{i:03d}" for i in range(30)]
    active = hosts[:25]

    emb = pd.DataFrame(
        rng.normal(size=(30, 8)),
        index=hosts,
    )

    result = temporal_anomaly(
        emb,
        emb.copy(),
        _flows(active, 1),
        _flows(active, 2),
        hosts,
        shared_coordinate_system=True,
    )

    assert result["n_common_active_hosts"] == 25
    assert result["n_scored_hosts"] == 25
    assert result["excluded_hosts"]["not_active_train"] == 5
    assert result["excluded_hosts"]["not_active_test"] == 5


def test_unrelated_coordinate_system_uses_heldout_hosts():
    rng = np.random.default_rng(12)
    hosts = [f"h{i:03d}" for i in range(40)]

    train = rng.normal(size=(40, 8))
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    test = train @ q

    result = temporal_anomaly(
        pd.DataFrame(train, index=hosts),
        pd.DataFrame(test, index=hosts),
        _flows(hosts, 1),
        _flows(hosts, 2),
        hosts,
        shared_coordinate_system=False,
        seed=5,
    )

    assert result["alignment"] == "procrustes"
    assert result["n_alignment_hosts"] == 20
    assert result["n_scored_hosts"] == 20
