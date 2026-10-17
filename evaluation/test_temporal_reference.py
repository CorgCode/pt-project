"""Тесты эталона из другого окна (синтетические потоки, НЕ LANL)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from evaluation.temporal_reference import generate_reference, load_flows
from evaluation.evaluate import run


def flows_file(tmp_path, name, n=25, ts0=1700000000):
    hosts = [f"h{i}" for i in range(n)]
    rows = []
    for day in range(2):
        for i, host in enumerate(hosts):
            rows.append({
                "src": host,
                "dst": hosts[(i + day + 1) % n],
                "dst_port": str([53, 80, 443, 8080][i % 4]),
                "proto": "tcp" if i % 3 else "udp",
                "bytes": 100 + i * 40 + day,
                "packets": 2 + i,
                "ts": ts0 + day * 3600 + i,
            })
    file = tmp_path / name
    pd.DataFrame(rows).to_csv(file, index=False)
    return file, hosts


def test_generates_reference_and_evaluates(tmp_path):
    first, hosts = flows_file(tmp_path, "train.csv")
    second, _ = flows_file(tmp_path, "test.csv", ts0=1700200000)
    manifest = generate_reference(load_flows(first), load_flows(second),
                                  hosts, tmp_path / "reference")
    ref = pd.read_csv(tmp_path / "reference" / "features.csv")
    assert ref.host_id.tolist() == hosts
    assert len([c for c in ref if c.startswith("ref_")]) > 10
    assert np.isfinite(ref.filter(like="ref_").values).all()
    assert manifest["ground_truth"] is False
    assert manifest["provenance_verified"] is False
    assert manifest["train_end_utc"] < manifest["test_start_utc"]
    assert 0 <= manifest["reference_neighbor_stability"]["consistency@5"] <= 1
    assert json.loads((tmp_path / "reference" / "manifest.json").read_text()) == manifest

    # оценка работает без изменений на построенном отложенном эталоне
    rng = np.random.default_rng(0)
    embeddings = pd.DataFrame(rng.normal(size=(25, 4)),
                              columns=[f"emb_{i}" for i in range(4)])
    embeddings.insert(0, "host_id", hosts)
    submission = tmp_path / "embeddings.csv"
    embeddings.to_csv(submission, index=False)
    result = run(submission, tmp_path / "reference", method="synthetic_test")
    assert result["metrics"]["consistency@5_mean"] >= 0
    assert result["n_hosts"] == 25


def test_refuses_overlapping_or_touching_windows(tmp_path):
    first, hosts = flows_file(tmp_path, "train.csv")
    second, _ = flows_file(tmp_path, "test.csv", ts0=1700000000 + 20)
    with pytest.raises(ValueError, match="пересекаются"):
        generate_reference(load_flows(first), load_flows(second),
                           hosts, tmp_path / "ref")


def test_refuses_inactive_hosts(tmp_path):
    first, hosts = flows_file(tmp_path, "train.csv")
    second, _ = flows_file(tmp_path, "test.csv", n=24, ts0=1700200000)
    with pytest.raises(ValueError, match="неактивны"):
        generate_reference(load_flows(first), load_flows(second),
                           hosts, tmp_path / "ref")


def test_refuses_invalid_k(tmp_path):
    first, hosts = flows_file(tmp_path, "train.csv")
    second, _ = flows_file(tmp_path, "test.csv", ts0=1700200000)
    with pytest.raises(ValueError, match="каждое k"):
        generate_reference(load_flows(first), load_flows(second),
                           hosts, tmp_path / "ref", ks=(25,))


@pytest.mark.parametrize("col,value", [
    ("bytes", -1),
    ("packets", float("nan")),
    ("src", ""),
    ("ts", "not-a-timestamp"),
])
def test_refuses_bad_input(tmp_path, col, value):
    first, _ = flows_file(tmp_path, "train.csv")
    df = pd.read_csv(first)
    if col == "ts":
        df[col] = df[col].astype(object)  # pandas >= 3 не пускает строки в int-колонки
    df.loc[0, col] = value
    df.to_csv(first, index=False)
    with pytest.raises(ValueError):
        load_flows(first)


def test_iso_timestamps_are_supported(tmp_path):
    file, _ = flows_file(tmp_path, "train.csv")
    df = pd.read_csv(file)
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    df.to_csv(file, index=False)
    loaded = load_flows(file)
    assert pd.api.types.is_datetime64_any_dtype(loaded["ts"])
    assert loaded["ts"].notna().all()
