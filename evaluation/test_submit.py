import json
import shutil

import pandas as pd
import pytest

from evaluation import submit


@pytest.fixture
def board(tmp_path, monkeypatch, processed, reference):
    root = tmp_path / "scoreboard"
    shutil.copytree(processed, root / "data" / "processed")
    shutil.copytree(reference, root / "data" / "reference")
    monkeypatch.setattr(submit, "BOARD", root)
    monkeypatch.setattr(submit, "DATA", root / "data")
    monkeypatch.setattr(submit, "SUBMISSIONS", root / "submissions")
    monkeypatch.setattr(submit, "RESULTS", root / "results")
    monkeypatch.setattr(submit, "N_BOOT", 0)
    return root


def _scores(board):
    for d in sorted((board / "submissions").iterdir()):
        submit.score(d)
    submit.render()


def test_builtin_methods_land_in_the_table_with_a_leader(board):
    submit.builtin(["random", "behavioral", "graph_spectral"])
    _scores(board)
    md = (board / "README.md").read_text(encoding="utf-8")
    assert "Лидер:" in md and "**" in md and "случайный уровень" in md
    assert "*`random`*" in md                                      # ориентир, а не участник
    r = json.loads((board / "results" / "behavioral.json").read_text())
    assert "identity_top1" in r["metrics"] and any(s["metric"] == "identity_top1" for s in r["summary"])
    assert set(pd.read_csv(board / "scoreboard.csv").method) == {"random", "behavioral", "graph_spectral"}


def test_fresh_results_are_not_recomputed_and_stale_ones_are(board):
    submit.builtin(["behavioral"])
    d = board / "submissions" / "behavioral"
    submit.score(d)
    assert submit.is_fresh(d)
    (d / "test.csv").unlink()                                      # отправка изменилась
    assert not submit.is_fresh(d)


def test_removed_submission_disappears_from_the_table(board):
    submit.builtin(["random", "behavioral"])
    _scores(board)
    shutil.rmtree(board / "submissions" / "random")
    submit.render()
    assert not (board / "results" / "random.json").exists()
    assert "| *`random`* |" not in (board / "README.md").read_text(encoding="utf-8")


def test_broken_submission_is_refused(board):
    d = board / "submissions" / "bad"
    d.mkdir(parents=True)
    pd.DataFrame({"host_id": ["a", "a"], "emb_0": [1.0, 2.0]}).to_csv(d / "embeddings.csv", index=False)
    with pytest.raises(ValueError, match="повторяющиеся host_id"):
        submit.score(d)


def test_method_template_trains_saves_and_lands_in_the_table(board, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("template_method", submit.ROOT / "scoreboard" / "template_method.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    monkeypatch.setattr(tm, "ROOT", board.parent)               # board = <tmp>/scoreboard
    tm.main()
    d = board / "submissions" / tm.NAME
    assert (d / "embeddings.csv").exists() and (d / "test.csv").exists() and (d / "SHARED_COORDINATES").exists()
    assert f"`{tm.NAME}`" in (board / "README.md").read_text(encoding="utf-8")
