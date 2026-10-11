"""Contract tests for validated multi-window manifests."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from validate_benchmark import validate_manifest


def write_embedding(path: Path, hosts):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "host_id": hosts,
        "emb_0": [float(i + 1) for i in range(len(hosts))],
        "emb_1": [float(i * 2) for i in range(len(hosts))],
    }).to_csv(path, index=False)


def test_manifest_with_shared_cohort(tmp_path):
    rows = []
    for method in ("model_a", "model_b"):
        for window in ("10", "11"):
            rel = f"embeddings/{method}/{window}.csv"
            write_embedding(tmp_path / rel, ["001", "1", "02"])
            rows.append({"method": method, "window_id": window, "path": rel})
    manifest = tmp_path / "submission_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    assert validate_manifest(manifest) == 4


def test_rejects_cross_method_cohort_mismatch(tmp_path):
    write_embedding(tmp_path / "a.csv", ["001", "1"])
    write_embedding(tmp_path / "b.csv", ["001", "2"])
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([
        {"method": "a", "window_id": "10", "path": "a.csv"},
        {"method": "b", "window_id": "10", "path": "b.csv"},
    ]).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="incompatible host cohorts"):
        validate_manifest(manifest)


def test_rejects_repeated_method_window(tmp_path):
    write_embedding(tmp_path / "a.csv", ["a", "b"])
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([
        {"method": "a", "window_id": "10", "path": "a.csv"},
        {"method": "a", "window_id": "10", "path": "a.csv"},
    ]).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_manifest(manifest)


def test_rejects_missing_embedding_file(tmp_path):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{"method": "a", "window_id": "10", "path": "missing.csv"}]).to_csv(
        manifest, index=False
    )
    with pytest.raises(FileNotFoundError):
        validate_manifest(manifest)


def test_rejects_wrong_schema(tmp_path):
    write_embedding(tmp_path / "a.csv", ["a", "b"])
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{"method": "a", "window_id": "10", "filename": "a.csv"}]).to_csv(
        manifest, index=False
    )
    with pytest.raises(ValueError, match="Missing manifest columns"):
        validate_manifest(manifest)
