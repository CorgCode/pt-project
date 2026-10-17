"""Синтетика в формате LANL -> пайплайн -> эталоны -> общая оценка -> таблица результатов."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from evaluation.scoreboard import build as build_board
from evaluation.evaluate import run
from pipeline.data import ProcessedData
from pipeline.reference import build_reference

ALL_METRICS = ["knn_consistency", "future_link_prediction", "pairwise_roc_auc", "pairwise_pr_auc",
               "precision_at_k", "recall_at_k", "triplet_accuracy"]


def write_emb(path, df):
    df.index.name = "host_id"
    df.add_prefix("emb_").to_csv(path)
    return path


@pytest.fixture(scope="module")
def cohort(processed):
    return ProcessedData(processed).cohort


def random_emb(cohort, tmp_path, seed=0, d=8):
    return write_emb(tmp_path / f"rand{seed}.csv",
                     pd.DataFrame(np.random.default_rng(seed).normal(size=(len(cohort), d)), index=cohort))


def test_every_applicable_metric_runs_and_probe_is_unavailable(reference, cohort, tmp_path):
    r = run(random_emb(cohort, tmp_path), reference / "test", "random", n_boot=20)
    for name in ALL_METRICS:
        assert r["metric_status"][name] == "ok", name
    assert r["metric_status"]["linear_probe"].startswith("unavailable")
    assert any(u.startswith("linear_probe") for u in r["unavailable_metrics"])
    assert not any(k.startswith("probe_") for k in r["metrics"])           # зонд без меток не выдумывается
    assert r["split"] == "test" and r["reference_info"]["synthetic"] is True
    assert r["reference_info"]["data_kind"] == "synthetic_fixture"
    assert set(r["reference_info"]["files_sha256"]) >= {"pairs.csv", "edges.csv", "triplets.csv"}


def test_random_embedding_is_at_chance_on_all_metrics(reference, cohort, tmp_path):
    rs = [run(random_emb(cohort, tmp_path, s), reference / "test", "r")["metrics"] for s in range(4)]
    mean = lambda k: float(np.mean([m[k] for m in rs]))
    assert abs(mean("link_roc_auc") - 0.5) < 0.04 and abs(mean("pair_roc_auc") - 0.5) < 0.08
    assert abs(mean("triplet_accuracy") - 0.5) < 0.1
    assert 0.8 < mean("pair_pr_auc_lift") < 1.3 and mean("consistency@10_mean") < 0.1


def test_validation_reference_uses_validation_split_by_default(reference, cohort, tmp_path):
    r = run(random_emb(cohort, tmp_path), reference / "validation", "random")
    assert r["split"] == "validation" and r["reference_info"]["target_window"] == "validation"


def test_unfaithful_split_is_an_error(reference, cohort, tmp_path):
    with pytest.raises(ValueError):
        run(random_emb(cohort, tmp_path), reference / "test", "r", split="validation")


def test_embedding_missing_hosts_fails_loudly(reference, cohort, tmp_path):
    e = pd.DataFrame(np.random.default_rng(0).normal(size=(len(cohort) - 3, 4)), index=cohort[:-3])
    with pytest.raises(ValueError):
        run(write_emb(tmp_path / "short.csv", e), reference / "test", "short")


def test_row_order_does_not_change_scores(reference, cohort, tmp_path):
    e = pd.DataFrame(np.random.default_rng(1).normal(size=(len(cohort), 6)), index=cohort)
    a = run(write_emb(tmp_path / "a.csv", e), reference / "test", "m")["metrics"]
    b = run(write_emb(tmp_path / "b.csv", e.sample(frac=1, random_state=5)), reference / "test", "m")["metrics"]
    assert a.keys() == b.keys()
    assert all(a[k] == pytest.approx(b[k], abs=1e-9) for k in a)


def test_bootstrap_ci_present_and_reproducible(reference, cohort, tmp_path):
    p = random_emb(cohort, tmp_path)
    a = run(p, reference / "test", "m", n_boot=50, seed=1)
    b = run(p, reference / "test", "m", n_boot=50, seed=1)
    assert a["metrics_ci"] == b["metrics_ci"] and "link_roc_auc" in a["metrics_ci"]
    lo, hi = a["metrics_ci"]["triplet_accuracy"]
    assert lo <= a["metrics"]["triplet_accuracy"] <= hi
    assert run(p, reference / "test", "m", n_boot=0)["metrics_ci"] == {}


def test_probe_needs_declared_provenance(processed, cohort, tmp_path):
    roles = pd.DataFrame({"host_id": cohort, "target": ["a", "b"] * (len(cohort) // 2) + ["a"] * (len(cohort) % 2),
                          "split": ["train"] * (len(cohort) // 2) + ["test"] * (len(cohort) - len(cohort) // 2)})
    labels = tmp_path / "labels.csv"
    roles.to_csv(labels, index=False)
    ref = tmp_path / "ref"
    build_reference(processed, ref, labels=labels, labels_source="unit-test labels")
    e = random_emb(cohort, tmp_path)
    ok = run(e, ref / "test", "m")
    assert ok["metric_status"]["linear_probe"] == "ok" and "probe_accuracy" in ok["metrics"]
    assert ok["metric_details"]["linear_probe"]["labels_source"] == "unit-test labels"
    # тот же файл меток, но источник убран из манифеста -> отказ, а не расчёт
    mf = ref / "test" / "manifest.json"
    m = json.loads(mf.read_text())
    m["labels_source"] = None
    mf.write_text(json.dumps(m))
    refused = run(e, ref / "test", "m")
    assert refused["metric_status"]["linear_probe"].startswith("unavailable") and "probe_accuracy" not in refused["metrics"]


def test_cli_and_leaderboard_roundtrip(reference, cohort, tmp_path, monkeypatch):
    out = tmp_path / "results"
    for s, name in enumerate(["m1", "m2"]):
        p = random_emb(cohort, tmp_path, s)
        subprocess.run([sys.executable, "-m", "evaluation.evaluate", "--embeddings", str(p),
                        "--reference-dir", str(reference / "test"), "--method", name,
                        "--bootstrap", "10", "--output", str(out / f"{name}.json")],
                       check=True, capture_output=True)
    build_board(out, tmp_path / "board.csv")
    board = pd.read_csv(tmp_path / "board.csv")
    assert set(board.method) == {"m1", "m2"} and "triplet_accuracy" in board.columns


def test_leaderboard_rejects_scores_from_two_references(reference, cohort, processed, tmp_path, monkeypatch):
    other = tmp_path / "other"
    build_reference(processed, other, seed=7)               # другая выборка кандидатов
    out = tmp_path / "res"
    out.mkdir()
    p = random_emb(cohort, tmp_path)
    for name, ref in [("a", reference / "test"), ("b", other / "test")]:
        (out / f"{name}.json").write_text(json.dumps(run(p, ref, name)))
    with pytest.raises(ValueError, match="разных эталонах"):
        build_board(out, tmp_path / "x.csv")


def test_identity_cli(cohort, tmp_path):
    rng = np.random.default_rng(0)
    A = pd.DataFrame(rng.normal(size=(len(cohort), 6)), index=cohort)
    B = A + rng.normal(scale=0.05, size=A.shape)
    pa, pb = write_emb(tmp_path / "A.csv", A), write_emb(tmp_path / "B.csv", B)
    out = tmp_path / "id.json"
    subprocess.run([sys.executable, "-m", "evaluation.identity", "--embeddings-a", str(pa),
                    "--embeddings-b", str(pb), "--method", "m", "--shared-coordinate-system",
                    "--bootstrap", "10", "--output", str(out)], check=True, capture_output=True)
    res = json.loads(out.read_text())
    assert res["none"]["top1"] > 0.9 and res["procrustes"]["top1"] > 0.9
    assert res["shared_coordinate_system_declared"] is True


# ---------------------------------------------------------------- обученный классификатор связей
def test_learned_link_scorer_fit_on_earlier_window_only(reference, cohort, tmp_path):
    from evaluation.future_links import load_candidate_edges, persistence_baseline, score_learned
    emb = pd.DataFrame(np.random.default_rng(0).normal(size=(len(cohort), 6)), index=cohort)
    val = load_candidate_edges(reference / "validation" / "edges.csv")
    test = load_candidate_edges(reference / "test" / "edges.csv")
    r = score_learned(emb, val, test, fit_split="validation", eval_split="test")
    assert 0.4 < r["roc_auc"] < 0.6 and r["n_fit"] == len(val) and r["n_eval"] == len(test)
    with pytest.raises(ValueError, match="temporal leakage"):          # обучение на БОЛЕЕ ПОЗДНЕМ окне
        score_learned(emb, test, val, fit_split="test", eval_split="validation")
    with pytest.raises(ValueError, match="before the evaluation window"):
        score_learned(emb, test, test, fit_split="test", eval_split="test")
    p = persistence_baseline(test, "test")
    assert p["roc_auc"] > 0.55                                          # «связь уже была» несёт настоящий сигнал


def test_evaluator_reports_learned_link_score_and_fingerprints_fit_reference(reference, cohort, tmp_path):
    e = random_emb(cohort, tmp_path)
    plain = run(e, reference / "test", "m")
    fitted = run(e, reference / "test", "m", fit_reference_dir=reference / "validation")
    assert "link_learned_roc_auc" not in plain["metrics"] and "link_learned_roc_auc" in fitted["metrics"]
    assert fitted["reference_info"]["fit_reference_files_sha256"]["edges.csv"]
    assert fitted["metric_details"]["future_link_prediction"]["persistence_baseline"]["roc_auc"] > 0.5
