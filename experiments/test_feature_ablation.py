import json

import numpy as np
import pandas as pd
import pytest

from experiments.feature_ablation import BLOCKS, _embedding, _variants, render, run_ablation
from experiments.run_experiment import run_experiment
from pipeline.data import ProcessedData


@pytest.fixture(scope="module")
def ablation(processed, reference, tmp_path_factory):
    out = tmp_path_factory.mktemp("abl")
    return run_ablation(processed, reference, out, n_boot=40, seed=0), out


def test_outputs_and_variants(ablation):
    rep, out = ablation
    assert (out / "ablation.json").exists() and (out / "ablation.md").exists()
    assert list(rep["protocol"]["variants"]) == ["full", "no_ports"] + [f"no_{b}" for b in BLOCKS if b != "ports"]
    assert rep["protocol"]["variants"]["no_ports"] == ["ports"] and rep["protocol"]["variants"]["full"] == []
    assert (out / "embeddings" / "no_ports" / "train.csv").exists()


def test_identical_protocol_for_every_variant(ablation, reference):
    rep, out = ablation
    assert rep["protocol"]["identical_for_all_variants"] is True and rep["protocol"]["split"] == "test"
    hashes = json.loads((reference / "test" / "manifest.json").read_text())["files_sha256"]
    assert rep["protocol"]["reference_files_sha256"] == hashes
    scored = {pd.read_csv(out / "embeddings" / n / "train.csv", dtype={"host_id": str}).shape[0] for n in rep["protocol"]["variants"]}
    assert len(scored) == 1                                                       # все варианты встраивают одних и тех же хостов


def test_paired_difference_equals_the_difference_of_reported_means(ablation):
    rep, _ = ablation
    p = rep["paired_full_minus_no_ports"]
    m = rep["metrics"]
    assert p["consistency@10"]["difference"] == pytest.approx(m["full"]["consistency@10_mean"] - m["no_ports"]["consistency@10_mean"], abs=1e-9)
    for key in ("pair_roc_auc", "link_roc_auc", "triplet_accuracy"):
        assert p[key]["difference"] == pytest.approx(m["full"][key] - m["no_ports"][key], abs=1e-9)
    for rec in p.values():
        lo, hi = rec["ci95"]
        assert lo <= hi and rec["ci_excludes_zero"] == (lo > 0 or hi < 0)


def test_port_free_reference_diagnostic(ablation, reference):
    d = ablation[0]["port_free_reference_diagnostic"]
    n = len(pd.read_csv(reference / "test" / "eval_hosts.csv"))
    assert "NOT comparable" in d["note"]
    for k in (5, 10, 20):
        assert d["chance"][f"consistency@{k}"] == pytest.approx(k / (n - 1))
        assert d["chance"][f"consistency@{k}"] < d["reference_self_agreement"][f"consistency@{k}"] <= 1.0
        assert d["full"][f"consistency@{k}"] > 3 * d["chance"][f"consistency@{k}"]
    assert d["paired_full_minus_no_ports@10"]["ci95"][0] <= d["paired_full_minus_no_ports@10"]["difference"] <= d["paired_full_minus_no_ports@10"]["ci95"][1]


def test_fit_uses_only_the_train_window(processed):
    from benchmark.baselines.test_portfree import Guard
    d = Guard(processed)
    d.fitting = True
    try:
        _embedding(d, ("ports",))
    finally:
        d.fitting = False


def test_ablation_is_reproducible(processed, reference, ablation, tmp_path):
    again = run_ablation(processed, reference, tmp_path, n_boot=40, seed=0)
    assert json.dumps(again, sort_keys=True) == json.dumps(ablation[0], sort_keys=True)


def test_report_states_provenance_and_the_methodological_caveats(ablation):
    rep, out = ablation
    text = (out / "ablation.md").read_text()
    assert "SYNTHETIC FIXTURE" in text and "nothing follows for LANL" in text
    for needle in ("by construction", "port-free reference diagnostic", "not a leaderboard reference", "leaderboard is unchanged",
                   "service identity", "PortNNNN"):
        assert needle in text, needle
    assert "UNVERIFIED INPUT" in render({**rep, "data_kind": "unverified", "badge": "UNVERIFIED INPUT - x"})
    assert "declared" in render({**rep, "data_kind": "lanl_declared", "badge": "LANL (declared by the operator)"}).lower()


def test_missing_reference_is_a_clear_error(processed, tmp_path):
    with pytest.raises(ValueError, match="pipeline.reference"):
        run_ablation(processed, tmp_path, tmp_path / "o", n_boot=0)


def test_runner_adds_the_variants_on_request_with_the_same_reference(processed, reference, tmp_path):
    plain = run_experiment(processed, reference, tmp_path / "a", dim=6, n_boot=0, do_audit=False)
    assert not any("portfree" in m for m in plain["methods"])
    m = run_experiment(processed, reference, tmp_path / "b", dim=6, n_boot=0, audit_configs=2, null_draws=2, port_free_baselines=True)
    assert {"behavioral_portfree", "behavioral_portfree_pca"} <= set(m["methods"])
    r = [json.loads((tmp_path / "b" / "results" / "test" / f"{n}.json").read_text()) for n in ("behavioral", "behavioral_portfree")]
    assert r[0]["reference_info"]["files_sha256"] == r[1]["reference_info"]["files_sha256"] and r[0]["n_hosts"] == r[1]["n_hosts"]
    audit = json.loads((tmp_path / "b" / "audit.json").read_text())
    assert "MINUS the port block" in audit["information_overlap"]["behavioral_portfree"]["neighbors"]
    assert "behavioral_portfree" in pd.read_csv(tmp_path / "b" / "scoreboard_test.csv").method.tolist()


def test_report_renders_without_a_bootstrap(processed, reference, tmp_path):
    rep = run_ablation(processed, reference, tmp_path, n_boot=0, leave_one_out=False)
    assert all(v["ci95"] is None and v["ci_excludes_zero"] is None for v in rep["paired_full_minus_no_ports"].values())
    text = (tmp_path / "ablation.md").read_text()
    assert "bootstrap off" in text and list(rep["protocol"]["variants"]) == ["full", "no_ports"]
