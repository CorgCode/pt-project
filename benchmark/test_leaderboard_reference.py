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


@pytest.mark.parametrize("a,b", [
    ("abc", "def"),
    ("abc", None),
])
def test_different_references_rejected(tmp_path, monkeypatch, a, b):
    folder = tmp_path / "results"
    write_score(folder, "pca", a)
    write_score(folder, "random", b)
    with pytest.raises(ValueError, match="different or unknown reference"):
        run_board(monkeypatch, folder, tmp_path / "board.csv")


def test_different_host_cohorts_rejected(tmp_path, monkeypatch):
    folder = tmp_path / "results"
    write_score(folder, "pca", "abc", n_hosts=25)
    write_score(folder, "random", "abc", n_hosts=30)
    with pytest.raises(ValueError, match="host cohorts"):
        run_board(monkeypatch, folder, tmp_path / "board.csv")
