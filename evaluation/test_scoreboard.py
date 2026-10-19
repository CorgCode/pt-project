"""Таблица результатов не должна смешивать оценки с разных эталонов, когорт, split и типов данных."""
from __future__ import annotations

import json
import pandas as pd
import pytest

from evaluation.scoreboard import build


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
    build(folder, output)


def test_same_reference_is_comparable(tmp_path, monkeypatch):
    folder = tmp_path / "results"
    write_score(folder, "pca", "abc")
    write_score(folder, "random", "abc")
    output = tmp_path / "board.csv"
    run_board(monkeypatch, folder, output)
    board = pd.read_csv(output)
    assert set(board["method"]) == {"pca", "random"}


@pytest.mark.parametrize("a,b,expected", [
    ("abc", "def", "разных эталонах"),
    ("abc", None, "нет отпечатка эталона"),
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
    with pytest.raises(ValueError, match="когортах"):
        run_board(monkeypatch, folder, tmp_path / "board.csv")

def test_both_unverified_references_rejected(tmp_path, monkeypatch):
    folder = tmp_path / "results"
    write_score(folder, "model_a", None)
    write_score(folder, "model_b", None)
    with pytest.raises(ValueError, match="нет отпечатка эталона"):
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
    with pytest.raises(ValueError, match="разных эталонах"):
        run_board(monkeypatch, f, tmp_path / "board.csv")


def test_synthetic_and_real_never_share_a_leaderboard(tmp_path, monkeypatch):
    f = tmp_path / "r"
    _score(f, "a", data_kind="lanl", synthetic=False)
    _score(f, "b", data_kind="synthetic_fixture", synthetic=True)
    with pytest.raises(ValueError, match="типы данных"):
        run_board(monkeypatch, f, tmp_path / "board.csv")


def test_validation_and_test_scores_never_mixed(tmp_path, monkeypatch):
    f = tmp_path / "r"
    _score(f, "a", data_kind="lanl", synthetic=False, split="test")
    _score(f, "b", data_kind="lanl", synthetic=False, split="validation")
    with pytest.raises(ValueError, match="split"):
        run_board(monkeypatch, f, tmp_path / "board.csv")


def test_consistent_new_style_results_are_accepted(tmp_path, monkeypatch):
    f = tmp_path / "r"
    _score(f, "a", data_kind="lanl", synthetic=False)
    _score(f, "b", data_kind="lanl", synthetic=False)
    run_board(monkeypatch, f, tmp_path / "board.csv")
    assert set(pd.read_csv(tmp_path / "board.csv").method) == {"a", "b"}


def _fit_score(folder, name, fit, learned=True):
    folder.mkdir(exist_ok=True)
    metrics = {"triplet_accuracy": 0.6}
    if learned:
        metrics["link_learned_roc_auc"] = 0.7
    payload = {"method": name, "n_hosts": 25, "split": "test", "metrics": metrics,
               "reference_info": {"files_sha256": {"pairs.csv": "x"}, "fit_reference_files_sha256": fit,
                                  "data_kind": "synthetic_fixture", "synthetic": True}}
    (folder / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_method_without_a_learned_link_score_may_omit_the_fit_reference(tmp_path, monkeypatch):
    f = tmp_path / "r"
    _fit_score(f, "model", {"edges.csv": "v1"})
    _fit_score(f, "ceiling", None, learned=False)            # тот же эталон оценки, без обученного классификатора
    run_board(monkeypatch, f, tmp_path / "board.csv")
    assert set(pd.read_csv(tmp_path / "board.csv").method) == {"model", "ceiling"}


def test_learned_scores_fitted_on_different_references_are_not_comparable(tmp_path, monkeypatch):
    f = tmp_path / "r"
    _fit_score(f, "a", {"edges.csv": "v1"})
    _fit_score(f, "b", {"edges.csv": "v2"})
    with pytest.raises(ValueError, match="классификатор связей"):
        run_board(monkeypatch, f, tmp_path / "board.csv")


def test_markdown_has_headlines_and_chance_row(tmp_path):
    f = tmp_path / "r"
    f.mkdir()
    for name, v in (("a", 0.7), ("b", 0.6)):
        payload = {"method": name, "n_hosts": 25, "split": "test",
                   "metrics": {"triplet_accuracy": v}, "metrics_ci": {"triplet_accuracy": [v - 0.1, v + 0.1]},
                   "summary": [{"group": "Общие собеседники", "metric": "triplet_accuracy", "title": "Тройки",
                                "value": v, "ci": None, "chance": 0.5}],
                   "reference_info": {"files_sha256": {"triplets.csv": "x"}, "data_kind": "synthetic_fixture",
                                      "synthetic": True}}
        (f / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    build(f, tmp_path / "board")
    md = (tmp_path / "board.md").read_text(encoding="utf-8")
    assert md.index("`a`") < md.index("`b`")                 # лучший метод выше
    assert "chance level" in md and "0.500" in md and "[0.60–0.80]" in md


def test_overall_score_maps_chance_to_0_and_perfect_to_100():
    from evaluation.scoreboard import column_scores, overall_score
    summary = [{"metric": "auc", "title": "AUC", "chance": 0.5}, {"metric": "top1", "title": "Top-1", "chance": 0.0}]
    r = {"method": "m", "metrics": {"auc": 0.75, "top1": 0.2}, "summary": summary}
    assert column_scores(r, {"auc": "AUC", "top1": "Top-1"}) == {"auc": 50.0, "top1": 20.0}
    worse = {"method": "w", "metrics": {"auc": 0.4, "top1": 0.0}, "summary": summary}   # ниже случайного — 0
    s = overall_score([r, worse], {"auc": "AUC", "top1": "Top-1"})
    assert abs(s["m"] - 1000 ** 0.5) < 1e-9 and s["w"] == 1.0          # геометрическое: √(50·20); ноль → 1


def test_overall_score_weighs_tasks_equally_and_punishes_a_failed_task():
    from evaluation.scoreboard import overall_score, task_scores
    summary = [{"metric": "a", "group": "Связи", "chance": 0.0}, {"metric": "b", "group": "Связи", "chance": 0.0},
               {"metric": "c", "group": "Узнаваемость", "chance": 0.0}]
    cols = {"a": "A", "b": "B", "c": "C"}
    r = {"method": "m", "metrics": {"a": 0.8, "b": 0.2, "c": 0.4}, "summary": summary}
    assert task_scores(r, cols) == pytest.approx({"Связи": 40.0, "Узнаваемость": 40.0})   # √(80·20) = 40
    balanced = {"method": "bal", "metrics": {"a": 0.4, "b": 0.4, "c": 0.4}, "summary": summary}
    lopsided = {"method": "lop", "metrics": {"a": 0.9, "b": 0.9, "c": 0.01}, "summary": summary}
    s = overall_score([r, balanced, lopsided], cols)
    assert s["m"] == pytest.approx(40.0) and s["bal"] == pytest.approx(40.0)
    assert s["lop"] < s["bal"]                         # среднее было бы 63 против 40, здесь — 9.5 против 40
