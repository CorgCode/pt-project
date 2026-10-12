"""Leaderboard must not compare scores from different reference datasets."""
from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

from benchmark.build_leaderboard import main


def write_score(folder, name, fingerprint, n_hosts=25):
    folder.mkdir(exist_ok=True)
    payload = {
        "method": name,
        "n_hosts": n_hosts,
        "metrics": {"consistency@5_mean": 0.5},
        "reference_info": {
            "features_sha256": fingerprint,
            "kind": "heldout_temporal_behavior_proxy",
            "ground_truth": False,
        },
    }
    (folder / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def run_board(monkeypatch, folder, output):
    monkeypatch.setattr(sys, "argv", [
        "build_leaderboard.py", "--results-dir", str(folder), "--output", str(output)
    ])
    main()


def test_same_reference_is_comparable(tmp_path, monkeypatch):
    folder = tmp_path / "results"
    write_score(folder, "pca", "abc")
    write_score(folder, "random", "abc")
    output = tmp_path / "board.csv"
    run_board(monkeypatch, folder, output)
    board = pd.read_csv(output)
    assert set(board["method"]) == {"pca", "random"}


@pytest.mark.parametrize("a,b,expected", [
    ("abc", "def", "different or unknown reference"),
    ("abc", None, "no reference fingerprint"),
])
def test_different_references_rejected(tmp_path, monkeypatch, a, b, expected):
    folder = tmp_path / "results"
    write_score(folder, "pca", a)
    write_score(folder, "random", b)
    with pytest.raises(ValueError, match=expected):
        run_board(monkeypatch, folder, tmp_path / "board.csv")


def test_different_host_cohorts_rejected(tmp_path, monkeypatch):
    folder = tmp_path / "results"
    write_score(folder, "pca", "abc", n_hosts=25)
    write_score(folder, "random", "abc", n_hosts=30)
    with pytest.raises(ValueError, match="host cohorts"):
        run_board(monkeypatch, folder, tmp_path / "board.csv")

def test_both_unverified_references_rejected(tmp_path, monkeypatch):
    folder = tmp_path / "results"
    write_score(folder, "model_a", None)
    write_score(folder, "model_b", None)
    with pytest.raises(ValueError, match="no reference fingerprint"):
        run_board(monkeypatch, folder, tmp_path / "board.csv")


def _score(folder, name, **info):
    folder.mkdir(exist_ok=True)
    payload = {"method": name, "n_hosts": 25, "split": info.pop("split", "test"),
               "metrics": {"triplet_accuracy": 0.6},
               "reference_info": {"files_sha256": info.pop("files", {"pairs.csv": "x", "edges.csv": "y"}), **info}}
    (folder / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_all_reference_files_are_fingerprinted(tmp_path, monkeypatch):
    f = tmp_path / "r"
    _score(f, "a", data_kind="lanl", synthetic=False)
    _score(f, "b", data_kind="lanl", synthetic=False, files={"pairs.csv": "x", "edges.csv": "CHANGED"})
    with pytest.raises(ValueError, match="different or unknown reference"):
        run_board(monkeypatch, f, tmp_path / "board.csv")


def test_synthetic_and_real_never_share_a_leaderboard(tmp_path, monkeypatch):
    f = tmp_path / "r"
    _score(f, "a", data_kind="lanl", synthetic=False)
    _score(f, "b", data_kind="synthetic_fixture", synthetic=True)
    with pytest.raises(ValueError, match="data kinds"):
        run_board(monkeypatch, f, tmp_path / "board.csv")


def test_validation_and_test_scores_never_mixed(tmp_path, monkeypatch):
    f = tmp_path / "r"
    _score(f, "a", data_kind="lanl", synthetic=False, split="test")
    _score(f, "b", data_kind="lanl", synthetic=False, split="validation")
    with pytest.raises(ValueError, match="splits"):
        run_board(monkeypatch, f, tmp_path / "board.csv")


def test_consistent_new_style_results_are_accepted(tmp_path, monkeypatch):
    f = tmp_path / "r"
    _score(f, "a", data_kind="lanl", synthetic=False)
    _score(f, "b", data_kind="lanl", synthetic=False)
    run_board(monkeypatch, f, tmp_path / "board.csv")
    assert set(pd.read_csv(tmp_path / "board.csv").method) == {"a", "b"}
