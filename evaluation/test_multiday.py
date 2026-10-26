"""Несколько test-окон (другие дни): оценка на каждом, итог — среднее, таблица по дням."""
import json
import shutil

import pytest

from evaluation import submit
from pipeline.prepare import prepare
from pipeline.reference import build_reference
from pipeline.windows import PipelineConfig

CONFIG = PipelineConfig(windows={"train": (0, 21600), "validation": (21600, 32400),
                                 "test": (32400, 37800), "test_late": (37800, 43200)})


@pytest.fixture(scope="module")
def multiday(raw_fixture, tmp_path_factory):
    root = tmp_path_factory.mktemp("multiday")
    prepare(sorted(raw_fixture.glob("netflow_day-*.bz2")), CONFIG, root / "processed")
    build_reference(root / "processed", root / "reference")
    return root


@pytest.fixture
def board(tmp_path, monkeypatch, multiday):
    root = tmp_path / "scoreboard"
    shutil.copytree(multiday / "processed", root / "data" / "processed")
    shutil.copytree(multiday / "reference", root / "data" / "reference")
    for name, value in (("BOARD", root), ("DATA", root / "data"), ("SUBMISSIONS", root / "submissions"),
                        ("RESULTS", root / "results"), ("N_BOOT", 0)):
        monkeypatch.setattr(submit, name, value)
    monkeypatch.delenv("PT_DATA", raising=False)
    return root


def test_windows_are_ordered_and_named():
    assert CONFIG.ordered == ["train", "validation", "test", "test_late"]
    with pytest.raises(ValueError):
        PipelineConfig(windows={"train": (0, 10), "test": (10, 20), "holdout": (20, 30)})
    with pytest.raises(ValueError):                                     # test-окна не пересекаются
        PipelineConfig(windows={"train": (0, 10), "test": (10, 20), "test_b": (15, 30)})


def test_every_test_window_is_scored_and_averaged(board):
    assert submit.test_windows() == ["test", "test_late"]
    submit.builtin(["random", "behavioral"])
    sub = board / "submissions" / "behavioral"
    assert (sub / "test.csv").exists() and (sub / "test_late.csv").exists()
    r = submit.score(sub)
    assert r["test_windows"] == ["test", "test_late"] and r["ci_kind"] == "range over test windows"
    one, two = (r["by_window"][w]["metrics"]["consistency@10_mean"] for w in ("test", "test_late"))
    assert abs(r["metrics"]["consistency@10_mean"] - (one + two) / 2) < 1e-12
    assert r["metrics_ci"]["consistency@10_mean"] == [min(one, two), max(one, two)]
    assert {s["metric"] for s in r["summary"]} >= {"consistency@10_mean", "identity_top1"}


def test_table_shows_scores_by_day(board):
    submit.builtin(["random", "behavioral"])
    for d in sorted((board / "submissions").iterdir()):
        submit.score(d)
    submit.render()
    md = (board / "README.md").read_text(encoding="utf-8")
    assert "## По дням" in md and "от худшего до лучшего" in md and "проверяется на 2 отрезках" in md
    row = next(line for line in md.splitlines() if line.startswith("| `behavioral` |"))
    assert len(row.split("|")) == 2 + 4                                # метод, два окна, среднее
    r = json.loads((board / "results" / "behavioral.json").read_text())
    assert set(r["by_window"]) == {"test", "test_late"}


def test_score_is_the_mean_of_daily_scores(board):
    submit.builtin(["random", "behavioral"])
    for d in sorted((board / "submissions").iterdir()):
        submit.score(d)
    submit.render()
    md = (board / "README.md").read_text(encoding="utf-8")
    row = next(line for line in md.splitlines() if line.startswith("| `behavioral` |"))
    a, b, mean = (float(x.strip(" *")) for x in row.split("|")[2:5])
    assert abs((a + b) / 2 - mean) <= 0.051
    assert "день 01, 09:00–10:30; день 01, 10:30–12:00" in md
