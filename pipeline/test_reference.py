import json

import numpy as np
import pandas as pd
import pytest

from evaluation.evaluate import run
from evaluation.future_links import load_candidate_edges
from pipeline.data import ProcessedData
from pipeline.reference import build_reference


def test_each_target_has_complete_reference(reference):
    for target in ("validation", "test"):
        d = reference / target
        for name in ("features.csv", "neighbors.csv", "edges.csv", "pairs.csv", "triplets.csv", "manifest.json"):
            assert (d / name).exists(), (target, name)
        assert not (d / "labels.csv").exists()               # метки не выдумываются


def test_manifest_provenance(reference, processed):
    m = json.loads((reference / "test" / "manifest.json").read_text())
    assert m["synthetic"] is True and m["target_window"] == "test" and m["source_window"] == "train"
    assert m["pipeline_manifest_sha256"] and m["cohort_sha256"]
    assert set(m["files_sha256"]) >= {"features.csv", "neighbors.csv", "edges.csv", "pairs.csv"}
    assert m["labels"] is None and m["labels_source"] is None
    assert m["split"] == "test" and m["ground_truth"] is False
    assert m["judgments"]["relevance_definition"]


def test_edges_use_train_as_source_and_target_as_future(reference):
    e = load_candidate_edges(reference / "test" / "edges.csv")
    assert e.split.eq("test").all() and "seen_in_source" in e.columns
    assert 0.0 < e.label.astype(int).mean() < 1.0
    ev = load_candidate_edges(reference / "validation" / "edges.csv")
    assert ev.split.eq("validation").all()
    assert ev.target_window.iloc[0] != e.target_window.iloc[0]


def test_pairs_and_triplets_are_valid_and_judged_only(reference):
    m = json.loads((reference / "test" / "manifest.json").read_text())["judgments"]
    p = pd.read_csv(reference / "test" / "pairs.csv", dtype=str)
    assert (p.host_a < p.host_b).all() and not p.duplicated(["host_a", "host_b"]).any()
    score = p.ref_score.astype(float)
    lab = p.label.astype(int)
    assert (score[lab == 1] >= m["tau_pos"]).all() and (score[lab == 0] == 0).all()   # неразмеченные исключены
    assert m["n_unjudged_pairs"] > 0
    t = pd.read_csv(reference / "test" / "triplets.csv", dtype=str)
    assert len(t) > 0 and (t.anchor != t.positive).all() and (t.anchor != t.negative).all()


def test_reference_hosts_equal_cohort(reference, processed):
    cohort = set(ProcessedData(processed).cohort)
    for target in ("validation", "test"):
        assert set(pd.read_csv(reference / target / "features.csv", dtype=str).host_id) == cohort
        assert set(pd.read_csv(reference / target / "neighbors.csv", dtype=str).host_id) == cohort


def test_reference_is_deterministic(processed, reference, tmp_path):
    build_reference(processed, tmp_path)
    for target in ("validation", "test"):
        a = json.loads((reference / target / "manifest.json").read_text())["files_sha256"]
        b = json.loads((tmp_path / target / "manifest.json").read_text())["files_sha256"]
        assert a == b


def test_reference_changes_with_seed(processed, reference, tmp_path):
    build_reference(processed, tmp_path, seed=1)
    a = json.loads((reference / "test" / "manifest.json").read_text())["files_sha256"]
    b = json.loads((tmp_path / "test" / "manifest.json").read_text())["files_sha256"]
    assert a["edges.csv"] != b["edges.csv"] and a["features.csv"] == b["features.csv"]


def test_labels_require_provenance_and_cohort_membership(processed, tmp_path):
    cohort = ProcessedData(processed).cohort
    lab = pd.DataFrame({"host_id": cohort[:10], "target": "x", "split": "train"})
    p = tmp_path / "labels.csv"
    lab.to_csv(p, index=False)
    with pytest.raises(ValueError, match="labels-source"):
        build_reference(processed, tmp_path / "r", labels=p)
    bad = pd.DataFrame({"host_id": ["Nope"], "target": "x", "split": "train"})
    bad.to_csv(p, index=False)
    with pytest.raises(ValueError, match="вне когорты"):
        build_reference(processed, tmp_path / "r2", labels=p, labels_source="unit test")
    lab.to_csv(p, index=False)
    build_reference(processed, tmp_path / "r3", labels=p, labels_source="unit test")
    m = json.loads((tmp_path / "r3" / "test" / "manifest.json").read_text())
    assert m["labels_source"] == "unit test" and m["labels"]["n_labelled"] == 10


def test_reference_works_with_current_evaluator(reference, processed):
    emb_path = processed.parent / "emb_tmp.csv"
    cohort = ProcessedData(processed).cohort
    pd.DataFrame(np.random.default_rng(0).normal(size=(len(cohort), 4)),
                 index=pd.Index(cohort, name="host_id")).add_prefix("emb_").to_csv(emb_path)
    res = run(emb_path, reference / "test", "random")
    assert "consistency@10_mean" in res["metrics"]
