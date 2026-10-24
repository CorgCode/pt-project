"""Подбор на validation учитывает узнаваемость, если у отправки есть validation.csv."""
import importlib.util
import shutil

import pytest

from evaluation import submit


@pytest.fixture
def board(tmp_path, monkeypatch, processed, reference):
    root = tmp_path / "scoreboard"
    shutil.copytree(processed, root / "data" / "processed")
    shutil.copytree(reference, root / "data" / "reference")
    for name, value in (("BOARD", root), ("DATA", root / "data"), ("SUBMISSIONS", root / "submissions"),
                        ("RESULTS", root / "results"), ("N_BOOT", 0)):
        monkeypatch.setattr(submit, name, value)
    monkeypatch.delenv("PT_DATA", raising=False)
    return root


def test_validation_score_counts_identity_when_validation_vectors_exist(board):
    submit.builtin(["behavioral"])
    d = board / "submissions" / "behavioral"
    assert (d / "validation.csv").exists()
    without, t0 = submit.validation_score(d / "embeddings.csv")
    with_id, t1 = submit.validation_score(d / "embeddings.csv", d / "validation.csv")
    assert "Identity" not in t0 and "Identity" in t1


def test_hybrid_tfidf_tune_picks_from_grid(board, monkeypatch):
    spec = importlib.util.spec_from_file_location("hybrid_tfidf", submit.ROOT / "scoreboard" / "hybrid_tfidf.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "GRID", {"dim": (8,), "behavioral": (1.0, 2.0)})
    from pipeline.data import ProcessedData
    best = mod.tune(ProcessedData(board / "data" / "processed"))
    assert best["dim"] == 8 and best["weights"]["behavioral"] in (1.0, 2.0)
