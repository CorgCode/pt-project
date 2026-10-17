"""Тесты слоя данных на закоммиченном (синтетическом) демо."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.results import Bundle, BundleError, METRIC_GROUPS, discover

DEMO = Path(__file__).resolve().parents[1] / "demo" / "synthetic_run"


@pytest.fixture(scope="module")
def b():
    return Bundle(DEMO)


def test_demo_bundle_is_discovered_and_synthetic(b):
    assert DEMO in discover(DEMO.parents[1]) and b.synthetic and b.label == "СИНТЕТИКА"
    assert b.kind == "synthetic_fixture" and b.provenance_verified is False and "не трафик LANL" in b.badge
    assert b.targets == ["test", "validation"] and "random" in b.methods


def test_invalid_bundle_gives_clear_error(tmp_path):
    with pytest.raises(BundleError, match="run/manifest.json"):
        Bundle(tmp_path)


def test_board_has_every_method_and_nan_for_missing_metrics(b):
    t = b.board("test")
    assert set(t.index) == set(b.methods)
    assert t.loc["reference_ceiling", "consistency@10_mean"] == pytest.approx(1.0)
    assert t["probe_accuracy"].isna().all()                  # недоступное остаётся NaN, ничем не заполняется
    assert any(u.startswith("linear_probe") for u in b.unavailable())


def test_board_values_equal_the_result_files(b):
    r = b.results("test")["graph_spectral"]["metrics"]
    assert b.board("test").loc["graph_spectral", "triplet_accuracy"] == r["triplet_accuracy"]


def test_ci_brackets_estimate(b):
    lo, hi = b.ci("test", "behavioral", "pair_roc_auc")
    assert lo <= b.board("test").loc["behavioral", "pair_roc_auc"] <= hi
    assert b.ci("test", "behavioral", "no_such_metric") is None


def test_link_context_matches_audit(b):
    ctx = b.link_context()
    assert ctx["persistence_roc_auc"] > 0.5 and ctx["natural_prevalence"] < ctx["sampled_positive_rate"]


def test_projection_and_agreement(b):
    p = b.projection("behavioral")
    assert {"PC1", "PC2", "agreement"} <= set(p.columns) and p.agreement.between(0, 1).all()
    ceiling = b.per_host_agreement("reference_ceiling", 10)
    assert (ceiling == 1.0).all()                           # совпадают по построению
    assert b.per_host_agreement("random", 10).mean() < 0.1


def test_agreement_mean_equals_reported_consistency(b):
    assert b.per_host_agreement("graph_spectral", 10).mean() == pytest.approx(
        b.board("test").loc["graph_spectral", "consistency@10_mean"], abs=1e-9)


def test_neighborhood_excludes_self_and_marks_reference_overlap(b):
    host = b.embedding("behavioral").index[0]
    nb = b.neighborhood("behavioral", host, 10)
    assert host not in set(nb.host_id) and len(nb) == 10 and nb.cosine.is_monotonic_decreasing
    assert {"reference_rank", "in_reference_top_k"} <= set(nb.columns)
    ref = b.reference_list(host, 10)
    assert len(ref) == 10 and set(nb[nb.in_reference_top_k].host_id) <= set(ref.host_id)
    with pytest.raises(BundleError):
        b.neighborhood("behavioral", "nope")


def test_identity_audit_and_manifests(b):
    idn = b.identity("test")
    assert idn["random"]["procrustes"]["top1"] < 5 * idn["random"]["procrustes"]["chance_top1"]
    assert b.audit()["weight_sensitivity"]["mean_kendall_tau"] is not None
    assert b.reference_manifest("test")["synthetic"] is True and b.processed_manifest["synthetic"] is True
    assert b.cohort()["policy"] == "train_only" and b.cohort()["coverage"]["test"] == 1.0


def test_metric_groups_cover_known_keys(b):
    keys = {k for g in METRIC_GROUPS.values() for k, _ in g}
    assert {"link_learned_roc_auc", "triplet_accuracy", "consistency@10_mean"} <= keys


def test_legacy_bundle_without_a_kind_is_unverified_never_lanl(tmp_path):
    import json, shutil
    shutil.copytree(DEMO, tmp_path / "old")
    mf = tmp_path / "old" / "run" / "manifest.json"
    m = json.loads(mf.read_text())
    for k in ("data_kind", "synthetic", "provenance", "badge"):
        m.pop(k, None)
    mf.write_text(json.dumps(m))
    old = Bundle(tmp_path / "old")
    assert old.kind == "unverified" and old.synthetic is False and old.label == "НЕПРОВЕРЕННЫЕ"
    assert "за LANL не выдаётся" in old.badge


def test_scored_embedding_is_restricted_to_evaluable_hosts(b):
    assert list(b.scored_embedding("behavioral").index) == b.eval_hosts()
    assert set(b.projection("behavioral").index) == set(b.eval_hosts())


def test_pair_prevalence_keys_use_the_judged_wording(b):
    pairs = b.audit()["prevalence"]["pairs"]
    assert "natural" not in "".join(pairs)
    assert pairs["population_positive_rate"] < pairs["judged_population_prevalence"] < pairs["sampled_positive_rate"]
    assert pairs["unjudged_share"] > 0.5
    assert "pair_pr_auc_at_judged_prevalence" in {k for g in METRIC_GROUPS.values() for k, _ in g}
