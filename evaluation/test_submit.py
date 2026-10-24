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


def _card(d):
    """Заполненная карточка метода — без неё отправку не принимают."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "README.md").write_text("# m\n\n" + "".join(f"## {h}\n\nтекст\n\n" for h in submit.CARD_SECTIONS),
                                 encoding="utf-8")


def _scores(board):
    for d in sorted((board / "submissions").iterdir()):
        submit.score(d)
    submit.render()


def test_builtin_methods_land_in_the_table_with_a_leader(board):
    submit.builtin(["random", "behavioral", "graph_spectral"])
    _scores(board)
    md = (board / "README.md").read_text(encoding="utf-8")
    assert "| 1 | [`" in md and "**" in md and "chance level" in md
    assert "*[`random`](" in md                                    # ориентир, а не участник
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
    assert "[`random`]" not in (board / "README.md").read_text(encoding="utf-8")


def test_broken_submission_is_refused(board):
    d = board / "submissions" / "bad"
    _card(d)
    pd.DataFrame({"host_id": ["a", "a"], "emb_0": [1.0, 2.0]}).to_csv(d / "embeddings.csv", index=False)
    with pytest.raises(ValueError, match="повторяющиеся host_id"):
        submit.score(d)


def test_method_template_trains_saves_and_lands_in_the_table(board, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("template_method", submit.ROOT / "scoreboard" / "template_method.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    monkeypatch.setattr(tm, "ROOT", board.parent)               # board = <tmp>/scoreboard
    d = board / "submissions" / tm.NAME
    with pytest.raises(SystemExit, match="Заполни карточку"):  # первый запуск кладёт заготовку карточки
        tm.main()
    assert "TODO" in (d / "README.md").read_text(encoding="utf-8")
    _card(d)
    tm.main()
    assert (d / "embeddings.csv").exists() and (d / "test.csv").exists() and (d / "SHARED_COORDINATES").exists()
    assert f"`{tm.NAME}`" in (board / "README.md").read_text(encoding="utf-8")


def test_submission_without_validation_hosts_is_scored_without_learned_links(board):
    import pandas as pd
    hosts = pd.read_csv(board / "data" / "reference" / "test" / "eval_hosts.csv", dtype=str)["host_id"]
    f = pd.read_csv(board / "data" / "reference" / "test" / "features.csv", dtype={"host_id": str}).set_index("host_id")
    val = board / "data" / "reference" / "validation" / "eval_hosts.csv"
    val.write_text(val.read_text() + "ghost-host-only-in-validation\n")   # хост, которого нет в эмбеддинге
    d = board / "submissions" / "ceiling"
    _card(d)
    f.loc[hosts].rename(columns=lambda c: c.replace("ref_", "emb_")).to_csv(d / "embeddings.csv")
    r = submit.score(d)
    assert "link_learned_roc_auc" not in r["metrics"] and "link_roc_auc" in r["metrics"]


def test_submission_needs_a_filled_card(board):
    submit.builtin(["behavioral"])
    d = board / "submissions" / "behavioral"
    (d / "README.md").unlink()
    with pytest.raises(ValueError, match="нет карточки"):
        submit.score(d)
    (d / "README.md").write_text((submit.ROOT / "scoreboard" / "CARD_TEMPLATE.md").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="заполни разделы Кратко"):
        submit.score(d)


def test_table_links_each_method_to_its_card(board):
    submit.builtin(["random", "behavioral"])
    _scores(board)
    md = (board / "README.md").read_text(encoding="utf-8")
    assert "[`behavioral`](" + submit.CARD_URL.format("behavioral") + ")" in md


def test_tikhon_metrics_are_columns_outside_the_score(board):
    submit.builtin(["random", "behavioral"])
    _scores(board)
    md = (board / "README.md").read_text(encoding="utf-8")
    head = next(line for line in md.splitlines() if line.startswith("| # |"))
    assert head.endswith("| Triplets* | Cosine links, AUC* |") and "в score не входят" in md


def test_builtin_skips_methods_already_trained_on_same_data(board, capsys):
    submit.builtin(["behavioral"])
    first = (board / "submissions" / "behavioral" / "embeddings.csv").stat().st_mtime_ns
    submit.builtin(["behavioral"])
    assert "пропускаю: behavioral" in capsys.readouterr().out
    assert (board / "submissions" / "behavioral" / "embeddings.csv").stat().st_mtime_ns == first
    submit.builtin(["behavioral"], force=True)
    assert (board / "submissions" / "behavioral" / "embeddings.csv").stat().st_mtime_ns != first


def test_builtin_in_two_processes_gives_same_vectors(board, tmp_path):
    submit.builtin(["behavioral", "graph_spectral"])
    one = {m: (board / "submissions" / m / "embeddings.csv").read_text() for m in ("behavioral", "graph_spectral")}
    submit.builtin(["behavioral", "graph_spectral"], jobs=2, force=True)
    for m, text in one.items():
        assert (board / "submissions" / m / "embeddings.csv").read_text() == text
