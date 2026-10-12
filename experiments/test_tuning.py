import json
import shutil
import subprocess
import sys

import pytest

from experiments.tune_methods import MAX_CONFIGS, tune

GRID = {"num_walks": [2, 4], "window": [2, 3]}


@pytest.fixture(scope="module")
def tuned(processed, reference):
    return tune(processed, reference, "deepwalk", GRID, dim=4, seed=0, metric="pair_roc_auc", fast=True)


def test_tries_every_combination_and_picks_by_the_declared_metric(tuned):
    assert tuned["n_configurations"] == 4 and tuned["selected_by"] == "pair_roc_auc"
    scores = {json.dumps(r["config"], sort_keys=True): r["metrics"]["pair_roc_auc"] for r in tuned["results"]}
    assert len(scores) == 4
    assert tuned["best_validation_score"] == max(scores.values())
    assert json.dumps(tuned["best_config"], sort_keys=True) in scores
    assert "optimistically biased" in tuned["warning"] and tuned["test_reference_read"] is False


def test_is_deterministic(processed, reference, tuned):
    again = tune(processed, reference, "deepwalk", GRID, dim=4, seed=0, metric="pair_roc_auc", fast=True)
    assert again["best_config"] == tuned["best_config"]
    assert [r["metrics"]["pair_roc_auc"] for r in again["results"]] == [r["metrics"]["pair_roc_auc"] for r in tuned["results"]]


def test_never_reads_the_test_reference(processed, reference, tmp_path):
    only_val = tmp_path / "ref"
    shutil.copytree(reference / "validation", only_val / "validation")          # no test directory at all
    out = tune(processed, only_val, "deepwalk", {"num_walks": [2]}, dim=4, metric="pair_roc_auc", fast=True)
    assert out["target_reference"] == "validation"
    r = subprocess.run([sys.executable, "-m", "experiments.tune_methods", "--processed-dir", str(processed),
                        "--reference-dir", str(reference), "--method", "deepwalk", "--grid", "{}", "--target", "test"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "refused" in r.stderr


def test_validation_split_is_used_for_scoring(processed, reference, tuned):
    r = tuned["results"][0]["metrics"]
    assert "pair_roc_auc" in r and "link_learned_roc_auc" not in r              # learned scorer needs an earlier fit window


def test_input_errors(processed, reference, tmp_path):
    with pytest.raises(ValueError, match="validation reference"):
        tune(processed, tmp_path, "deepwalk", {}, fast=True)
    with pytest.raises(ValueError, match="unknown method"):
        tune(processed, reference, "nope", {})
    with pytest.raises(ValueError, match="not available"):
        tune(processed, reference, "deepwalk", {"num_walks": [2]}, dim=4, metric="probe_accuracy", fast=True)
    with pytest.raises(ValueError, match="exceed the limit"):
        tune(processed, reference, "deepwalk", {"num_walks": list(range(1, MAX_CONFIGS + 2))}, fast=True)
    with pytest.raises(ValueError, match="unknown parameter"):
        tune(processed, reference, "deepwalk", {"bogus": [1]}, fast=True)


def test_empty_grid_evaluates_the_defaults_once(processed, reference):
    out = tune(processed, reference, "deepwalk", {}, dim=4, metric="triplet_accuracy", fast=True)
    assert out["n_configurations"] == 1 and out["best_config"] == {}
