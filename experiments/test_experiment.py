import json

import numpy as np
import pandas as pd
import pytest

from experiments.run_experiment import EXPECTED_EXTERNAL, run_experiment
from pipeline.data import ProcessedData


@pytest.fixture(scope="module")
def run_dir(processed, reference, tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    run_experiment(processed, reference, out, dim=8, seed=0, n_boot=20, audit_configs=3, null_draws=5)
    return out


def test_all_outputs_exist(run_dir):
    for p in ["manifest.json", "report.md", "audit.json", "scoreboard_test.csv", "scoreboard_validation.csv",
              "figures/scorecard_test.png", "figures/knn_by_k.png", "figures/identity_test.png",
              "figures/weight_sensitivity.png", "results/test/random.json", "identity/test/behavioral.json"]:
        assert (run_dir / p).exists(), p


def test_synthetic_is_labelled_everywhere(run_dir):
    m = json.loads((run_dir / "manifest.json").read_text())
    assert m["synthetic"] is True and m["data_kind"] == "synthetic_fixture"
    assert "СИНТЕТИКА" in (run_dir / "report.md").read_text()
    r = json.loads((run_dir / "results" / "test" / "behavioral.json").read_text())
    assert r["reference_info"]["synthetic"] is True


def test_missing_methods_are_listed_not_simulated(run_dir):
    m = json.loads((run_dir / "manifest.json").read_text())
    assert m["missing_expected_methods"] == EXPECTED_EXTERNAL and m["external_methods"] == []
    assert not any(x in m["methods"] for x in EXPECTED_EXTERNAL)
    assert "в этом прогоне нет" in (run_dir / "report.md").read_text()


def test_same_cohort_and_reference_for_every_method(run_dir):
    res = [json.loads(p.read_text()) for p in (run_dir / "results" / "test").glob("*.json")]
    assert len({r["n_hosts"] for r in res}) == 1
    assert len({json.dumps(r["reference_info"]["files_sha256"], sort_keys=True) for r in res}) == 1
    assert len({r["split"] for r in res}) == 1


def test_sanity_ordering(run_dir):
    board = pd.read_csv(run_dir / "scoreboard_test.csv").set_index("method")
    assert board.loc["reference_ceiling", "consistency@10_mean"] == pytest.approx(1.0)
    assert board.loc["random", "consistency@10_mean"] < 0.1 < board.loc["behavioral", "consistency@10_mean"]
    assert abs(board.loc["random", "pair_roc_auc"] - 0.5) < 0.1
    assert board.loc["graph_spectral", "link_learned_roc_auc"] > board.loc["random", "link_learned_roc_auc"] + 0.05
    assert "link_learned_roc_auc" not in pd.read_csv(run_dir / "scoreboard_validation.csv").columns


def test_identity_is_chance_for_random_and_above_for_real_methods(run_dir):
    rnd = json.loads((run_dir / "identity" / "test" / "random.json").read_text())
    beh = json.loads((run_dir / "identity" / "test" / "behavioral.json").read_text())
    gr = json.loads((run_dir / "identity" / "test" / "graph_spectral.json").read_text())
    assert rnd["procrustes"]["top1"] < 5 * rnd["procrustes"]["chance_top1"]
    assert beh["procrustes"]["top1"] > 3 * beh["procrustes"]["chance_top1"]
    assert gr["shared_coordinate_system_declared"] is False and gr["procrustes"]["top1"] > 0.05


def test_audit_content(run_dir):
    a = json.loads((run_dir / "audit.json").read_text())
    w = a["weight_sensitivity"]
    assert len(w["configs"]) == 3 and w["mean_reference_agreement"] > w["chance_agreement"]
    assert set(a["reference_stability"]["pairs"]) >= {"train->test"}
    assert a["random_null"]["n_draws"] == 5 and "link_roc_auc" in a["random_null"]["metrics"]
    assert a["prevalence"]["future_links"]["natural_prevalence"] < a["prevalence"]["future_links"]["sampled_positive_rate"]


def test_external_submission_is_evaluated_and_unwindowed_has_no_identity(processed, reference, tmp_path):
    d = ProcessedData(processed)
    sub = tmp_path / "subs" / "mymodel"
    sub.mkdir(parents=True)
    pd.DataFrame(np.random.default_rng(3).normal(size=(len(d.cohort), 5)),
                 index=pd.Index(d.cohort, name="host_id")).add_prefix("emb_").to_csv(sub / "embeddings.csv")
    out = tmp_path / "run"
    m = run_experiment(processed, reference, out, dim=4, n_boot=0, submissions_dir=tmp_path / "subs", do_audit=False)
    assert "mymodel" in m["methods"] and m["external_methods"] == ["mymodel"]
    assert (out / "results" / "test" / "mymodel.json").exists()
    assert not (out / "identity" / "test" / "mymodel.json").exists()


def test_cohort_mismatch_between_reference_and_processed_data_is_refused(processed, reference, tmp_path):
    other = tmp_path / "ref"
    import shutil
    shutil.copytree(reference, other)
    mf = other / "test" / "manifest.json"
    m = json.loads(mf.read_text())
    m["cohort_sha256"] = "0" * 64
    mf.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="по другой когорте"):
        run_experiment(processed, other, tmp_path / "o", n_boot=0, do_audit=False)


def test_end_to_end_fixture_one_command(tmp_path):
    from experiments.end_to_end import end_to_end
    m = end_to_end(tmp_path, fixture=True, n_hosts=120, dim=6, n_boot=0, pairs_per_anchor=10, seed=2)
    assert m["synthetic"] is True and m["n_hosts"] > 20
    assert (tmp_path / "run" / "report.md").exists() and (tmp_path / "reference" / "test" / "pairs.csv").exists()
    with pytest.raises(ValueError, match="--fixture"):
        end_to_end(tmp_path / "x")


def test_demo_bundle_is_synthetic_and_has_no_absolute_paths(tmp_path):
    from experiments.make_demo_bundle import build
    out = build(tmp_path / "b", n_hosts=100, n_boot=0)
    assert "НЕ данные LANL" in (out / "README.md").read_text(encoding="utf-8")
    m = json.loads((out / "run" / "manifest.json").read_text())
    assert m["synthetic"] is True
    blob = "".join(p.read_text() for p in out.rglob("*.json")) + (out / "run" / "report.md").read_text()
    assert str(tmp_path) not in blob and "/private/" not in blob and "/Users/" not in blob
    assert (out / "reference" / "test" / "neighbors.csv").exists() and (out / "run" / "embeddings" / "random" / "train.csv").exists()
