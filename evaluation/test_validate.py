"""Проверка формата эмбеддингов и манифеста нескольких окон."""
from __future__ import annotations

import pandas as pd
import pytest

from pathlib import Path

from evaluation.validate import validate_manifest, validate_submission
from evaluation.evaluate import _load_matrix


def test_validates_and_preserves_lexical_host_ids(tmp_path):
    file = tmp_path / "embeddings.csv"
    pd.DataFrame({
        "host_id": ["001", "1", "02"],
        "emb_0": [1.0, 2.0, 3.0],
        "emb_1": [0.0, 1.0, 1.0],
    }).to_csv(file, index=False)
    validate_submission(str(file))
    mat = _load_matrix(file, "host_id", "emb_")
    assert list(mat.index) == ["001", "1", "02"]


@pytest.mark.parametrize("invalid_id", ["", "  "])
def test_rejects_empty_or_blank_host_id(tmp_path, invalid_id):
    file = tmp_path / "embedding.csv"
    pd.DataFrame({"host_id": ["ok", invalid_id],
                  "emb_0": [1.0, 2.0]}).to_csv(file, index=False)
    with pytest.raises(ValueError, match="host_id"):
        validate_submission(str(file))


@pytest.mark.parametrize("bad_value", [float("inf"), -float("inf"), float("nan")])
def test_rejects_nonfinite_coordinates(tmp_path, bad_value):
    file = tmp_path / "embedding.csv"
    pd.DataFrame({"host_id": ["a", "b"],
                  "emb_0": [1.0, bad_value]}).to_csv(file, index=False)
    with pytest.raises(ValueError, match="NaN|конечными"):
        validate_submission(str(file))

@pytest.mark.parametrize("value", ["inf", "-inf", "garbage"])
def test_common_evaluator_refuses_invalid_numbers(tmp_path, value):
    file = tmp_path / "embeddings.csv"
    pd.DataFrame({"host_id": ["a", "b"], "emb_0": ["1.2", value]}).to_csv(file, index=False)
    with pytest.raises(ValueError, match="finite|numeric"):
        _load_matrix(file, "host_id", "emb_")


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
    with pytest.raises(ValueError, match="разные наборы хостов"):
        validate_manifest(manifest)


def test_rejects_repeated_method_window(tmp_path):
    write_embedding(tmp_path / "a.csv", ["a", "b"])
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([
        {"method": "a", "window_id": "10", "path": "a.csv"},
        {"method": "a", "window_id": "10", "path": "a.csv"},
    ]).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="повтор"):
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
    with pytest.raises(ValueError, match="нет колонок"):
        validate_manifest(manifest)
