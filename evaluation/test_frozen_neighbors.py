"""Замороженные соседи общие для всех и проверяются до оценки."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from evaluation.evaluate import run
from evaluation.frozen_neighbors import (
    evaluate_frozen_neighbors,
    read_frozen_neighbors,
    save_frozen_neighbors,
)


@pytest.fixture
def case(tmp_path):
    rng = np.random.default_rng(2026)
    hosts = [f"H{i:03d}" for i in range(27)]
    data = rng.normal(size=(len(hosts), 5))
    features = pd.DataFrame(data, index=hosts,
                            columns=[f"ref_{i}" for i in range(5)])
    ref_dir = tmp_path / "reference"
    ref_dir.mkdir()
    features.reset_index(names="host_id").to_csv(ref_dir / "features.csv", index=False)
    neighbors = save_frozen_neighbors(
        features.iloc[::-1], ref_dir / "neighbors.csv",
        reference_window="2026-10-10T11:00Z/2026-10-10T12:00Z", k_max=20,
    )
    embeddings = pd.DataFrame(data, index=hosts,
                              columns=[f"emb_{i}" for i in range(5)])
    return features, embeddings, neighbors, ref_dir


def test_saves_contract_and_self_excluded(case):
    features, embeddings, path, _ = case
    df = pd.read_csv(path)
    assert list(df.columns) == [
        "reference_window", "host_id", "neighbor_id", "rank", "split"
    ]
    assert len(df) == len(features) * 20
    assert (df.host_id != df.neighbor_id).all()
    assert set(df.split) == {"test"}
    assert sorted(df["rank"].unique()) == list(range(1, 21))
    summary = evaluate_frozen_neighbors(embeddings, path)
    for k in (5, 10, 20):
        assert summary.loc[(k, "consistency"), "mean"] == pytest.approx(1.0)
        assert summary.loc[(k, "jaccard"), "mean"] == pytest.approx(1.0)
        assert summary.loc[(k, "ndcg"), "mean"] == pytest.approx(1.0)


def test_embedding_row_order_is_irrelevant(case):
    _, emb, path, _ = case
    reordered = emb.sample(frac=1, random_state=7)
    summary = evaluate_frozen_neighbors(reordered, path)
    assert summary.loc[(10, "consistency"), "mean"] == pytest.approx(1.0)


@pytest.mark.parametrize("corruption,error", [
    ("duplicate_rank", "Duplicate neighbor ranks"),
    ("self_neighbor", "its own"),
    ("missing_host", "Missing reference neighbors"),
    ("gap", "Missing/non-contiguous"),
    ("unknown_neighbor", "outside embedding"),
    ("multiple_windows", "multiple reference windows"),
])
def test_bad_reference_is_rejected(case, tmp_path, corruption, error):
    _, emb, path, _ = case
    df = pd.read_csv(path, dtype={"host_id": "string", "neighbor_id": "string"})
    if corruption == "duplicate_rank":
        df.loc[0, "rank"] = 2
    elif corruption == "self_neighbor":
        df.loc[0, "neighbor_id"] = df.loc[0, "host_id"]
    elif corruption == "missing_host":
        df = df.loc[df.host_id != "H002"]
    elif corruption == "gap":
        df.loc[0, "rank"] = 99
    elif corruption == "unknown_neighbor":
        df.loc[0, "neighbor_id"] = "NOT_A_HOST"
    elif corruption == "multiple_windows":
        df.loc[0, "reference_window"] = "another-window"
    corrupted = tmp_path / "corrupted.csv"
    df.to_csv(corrupted, index=False)
    with pytest.raises(ValueError, match=error):
        read_frozen_neighbors(corrupted, emb.index.tolist(), k_max=20)


def test_missing_test_split_is_rejected(case, tmp_path):
    _, emb, path, _ = case
    df = pd.read_csv(path)
    df["split"] = "train"
    bad = tmp_path / "train_only.csv"
    df.to_csv(bad, index=False)
    with pytest.raises(ValueError, match="no test rows"):
        evaluate_frozen_neighbors(emb, bad)


def test_common_evaluator_prefers_frozen_reference(case, tmp_path):
    _, emb, neighbors, ref_dir = case
    csv = tmp_path / "embeddings.csv"
    emb.reset_index(names="host_id").to_csv(csv, index=False)
    first = run(csv, ref_dir, method="method_a")
    assert first["metrics"]["consistency@10_mean"] == pytest.approx(1)
    assert first["reference_info"]["neighbors_sha256"] is not None

    # убеждаемся, что оценка действительно читает файл соседей:
    # заменяем один список соседей другим (корректным) порядком
    table = pd.read_csv(neighbors)
    anchor = table.iloc[0]["host_id"]
    mask = table.host_id == anchor
    table.loc[mask, "neighbor_id"] = list(reversed(table.loc[mask, "neighbor_id"]))
    table.to_csv(neighbors, index=False)
    changed = run(csv, ref_dir, method="method_b")
    assert changed["metrics"]["ndcg@10_mean"] < first["metrics"]["ndcg@10_mean"]
    assert changed["reference_info"]["neighbors_sha256"] != first["reference_info"]["neighbors_sha256"]


def test_temporal_generator_emits_fixed_file(tmp_path):
    from evaluation.temporal_reference import generate_reference
    hosts = [f"h{i}" for i in range(24)]
    def make(start):
        rows = [{
            "src": hosts[i], "dst": hosts[(i + 1) % len(hosts)],
            "dst_port": 53 + i % 3, "proto": "tcp",
            "bytes": 100 + i * 7, "packets": i + 1,
            "ts": pd.Timestamp("2026-10-10T00:00:00") + pd.Timedelta(seconds=start + i),
        } for i in range(len(hosts))]
        return pd.DataFrame(rows)
    manifest = generate_reference(make(0), make(3600), hosts, tmp_path, ks=(5, 10, 20))
    assert (tmp_path / "neighbors.csv").exists()
    assert len(pd.read_csv(tmp_path / "neighbors.csv")) == len(hosts) * 20
    assert json.loads((tmp_path / "manifest.json").read_text())["ground_truth"] is False
