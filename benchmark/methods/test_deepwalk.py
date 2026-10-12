import json

import numpy as np
import pandas as pd
import pytest

from benchmark import methods
from benchmark.methods.conftest_helpers import TrainOnlyGuard
from benchmark.methods.deepwalk import DeepWalk
from evaluation.evaluate import run
from experiments.run_experiment import run_experiment
from pipeline.data import ProcessedData

FAST = dict(num_walks=6, walk_length=30, window=5, epochs=2)


@pytest.fixture(scope="module")
def data(processed):
    return TrainOnlyGuard(processed)


@pytest.fixture(scope="module")
def fitted(data):
    return DeepWalk(dim=8, seed=0, **FAST).fit(data)


def test_registered_and_documented():
    assert "deepwalk" in methods.available()
    m = methods.create(["deepwalk"], dim=4, seed=1, config={"deepwalk": {"num_walks": 2}})[0]
    assert m.params["num_walks"] == 2 and m.params["objective"] == "hs" and "Perozzi" in m.reference
    with pytest.raises(ValueError, match="unknown method"):
        methods.create(["nope"])
    with pytest.raises(ValueError, match="not selected"):
        methods.create(["deepwalk"], config={"node2vec": {}})
    with pytest.raises(ValueError, match="unknown parameter"):
        DeepWalk(dim=4, bogus=1)


def test_output_contract(fitted, data):
    e = fitted.embed(data, "train")
    assert list(e.index) == data.cohort and e.index.name == "host_id" and e.shape == (len(data.cohort), 8)
    assert all(c.startswith("emb_") for c in e.columns)
    assert np.isfinite(e.to_numpy()).all() and (np.linalg.norm(e.to_numpy(), axis=1) > 0).all()


def test_fit_reads_only_the_train_window(processed):
    d = TrainOnlyGuard(processed)
    d.fitting = True
    try:
        DeepWalk(dim=4, seed=0, num_walks=2, walk_length=10, window=3).fit(d)
    finally:
        d.fitting = False


def test_scored_embedding_does_not_depend_on_future_windows(processed, data):
    other = ProcessedData(processed)
    other.flows = lambda w, orig=other.flows: orig(w).assign(bytes=lambda f: f["bytes"] * 3) if w != "train" else orig(w)
    a = DeepWalk(dim=4, seed=0, num_walks=3, walk_length=15, window=3).fit(data).embed(data, "train")
    b = DeepWalk(dim=4, seed=0, num_walks=3, walk_length=15, window=3).fit(other).embed(other, "train")
    assert np.array_equal(a.to_numpy(), b.to_numpy())


def test_deterministic_and_seed_sensitive(data):
    mk = lambda s: DeepWalk(dim=4, seed=s, num_walks=3, walk_length=15, window=3).fit(data).embed(data, "train")
    assert np.array_equal(mk(1).to_numpy(), mk(1).to_numpy()) and not np.array_equal(mk(1).to_numpy(), mk(2).to_numpy())


def test_later_windows_are_fitted_separately_and_flagged(fitted, data):
    hosts = data.evaluable("test")
    e = fitted.embed(data, "test", hosts)
    assert list(e.index) == hosts and fitted.shared_coordinate_system is False
    assert "test" in fitted.info()["fit_seconds"] and fitted.info()["fit_windows"] == ["train"]


def test_host_without_edges_in_a_window_is_an_error(fitted, data):
    with pytest.raises(KeyError, match="no edge"):
        fitted.embed(data, "train", ["definitely-not-a-host"])


def test_hierarchical_softmax_and_negative_sampling_both_work(data):
    for obj in ("hs", "neg"):
        e = DeepWalk(dim=4, seed=0, objective=obj, num_walks=2, walk_length=10, window=3).fit(data).embed(data, "train")
        assert np.isfinite(e.to_numpy()).all()


def test_learns_structure_that_random_vectors_do_not(processed, reference, tmp_path, fitted, data):
    """Shared-peer pair judgments and behavioural neighbours are far above chance for a trained embedding."""
    p = tmp_path / "dw.csv"
    fitted.embed(data, "train").to_csv(p)
    r = run(p, reference / "test", "dw")["metrics"]
    rnd = tmp_path / "rnd.csv"
    pd.DataFrame(np.random.default_rng(0).normal(size=(len(data.cohort), 8)), index=pd.Index(data.cohort, name="host_id")
                 ).add_prefix("emb_").to_csv(rnd)
    rr = run(rnd, reference / "test", "rnd")["metrics"]
    assert r["pair_roc_auc"] > 0.75 and r["triplet_accuracy"] > 0.75 and r["pair_roc_auc"] > rr["pair_roc_auc"] + 0.2
    assert r["consistency@10_mean"] > 3 * rr["consistency@10_mean"]


def test_runs_through_the_experiment_runner(processed, reference, tmp_path):
    cfg = {"deepwalk": dict(num_walks=3, walk_length=15, window=3)}
    m = run_experiment(processed, reference, tmp_path / "o", dim=6, n_boot=0, do_audit=False,
                       methods=["deepwalk"], method_config=cfg)
    assert "deepwalk" in m["methods"] and "deepwalk" not in m["missing_expected_methods"]
    assert m["missing_expected_methods"] == ["node2vec", "graphsage", "struc2vec", "rolx"]
    info = m["graph_methods"]["deepwalk"]
    assert info["params"]["num_walks"] == 3 and info["fit_windows"] == ["train"] and info["fit_seconds"]["train"] > 0
    res = json.loads((tmp_path / "o" / "results" / "test" / "deepwalk.json").read_text())
    assert res["metric_status"]["triplet_accuracy"] == "ok" and res["method_info"]["reference"]
    assert (tmp_path / "o" / "identity" / "test" / "deepwalk.json").exists()
    assert "deepwalk" in (tmp_path / "o" / "report.md").read_text()


def test_audit_table_shows_the_method_declared_overlap(processed, reference, tmp_path):
    m = run_experiment(processed, reference, tmp_path / "o", dim=4, n_boot=0, audit_configs=2, null_draws=2,
                       methods=["deepwalk"], method_config={"deepwalk": dict(num_walks=2, walk_length=10, window=3)})
    audit = json.loads((tmp_path / "o" / "audit.json").read_text())
    assert audit["information_overlap"]["deepwalk"]["edges"].startswith("strong")
    assert "random" in audit["information_overlap"]                       # baselines keep their rows


def test_all_and_fast_selection(processed, reference, tmp_path):
    from benchmark import methods as M
    fast = M.create(["all"], dim=4, fast=True)
    assert {m.name for m in fast} == set(M.available()) and fast[0].params["num_walks"] == 2
    full = M.create(["deepwalk"], dim=4)[0]
    assert full.params["num_walks"] == 10                               # paper default without --fast
    over = M.create(["deepwalk"], dim=4, config={"deepwalk": {"num_walks": 7}}, fast=True)[0]
    assert over.params["num_walks"] == 7 and over.params["walk_length"] == 10      # overrides win, fast fills the rest
    m = run_experiment(processed, reference, tmp_path / "o", dim=4, n_boot=0, do_audit=False,
                       methods=["all"], fast_methods=True)
    assert set(M.available()) <= set(m["methods"])
