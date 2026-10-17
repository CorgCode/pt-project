"""Смысл доли положительных у пар по общим собеседникам (ревью #50).

Доля, посчитанная по размеченным парам манифеста, — условная (пара размечена: J >= tau или J == 0).
Её нельзя называть долей «по всем парам», а большая доля неразмеченных должна менять долю по всем парам,
но не условные величины.
"""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from evaluation.evaluate import run
from evaluation.retrieval import pair_auc
from evaluation.uncertainty import reweighted_average_precision
from pipeline.data import ProcessedData


@pytest.fixture(scope="module")
def emb_path(processed, tmp_path_factory):
    hosts = ProcessedData(processed).cohort
    p = tmp_path_factory.mktemp("e") / "e.csv"
    pd.DataFrame(np.random.default_rng(0).normal(size=(len(hosts), 6)),
                 index=pd.Index(hosts, name="host_id")).add_prefix("emb_").to_csv(p)
    return p


def with_judgments(reference, tmp_path, **changes):
    """Скопировать test-эталон и переписать счётчики в манифесте (файл пар не трогаем)."""
    dst = tmp_path / "ref"
    shutil.copytree(reference / "test", dst)
    m = json.loads((dst / "manifest.json").read_text())
    m["judgments"].update(changes)
    (dst / "manifest.json").write_text(json.dumps(m))
    return dst


def test_no_pair_metric_is_called_natural(reference, emb_path):
    r = run(emb_path, reference / "test", "m")
    assert not any("natural" in k and k.startswith("pair") for k in r["metrics"])
    assert "pair_pr_auc_at_judged_prevalence" in r["metrics"]
    d = r["metric_details"]["pairwise_auc"]
    assert {"judged_population_prevalence", "population_positive_rate", "unjudged_share"} <= set(d)
    assert "размеченных" in d["interpretation"] and "не среди всех пар" in d["interpretation"]


def test_judged_prevalence_is_conditional_on_judged_pairs(reference, emb_path):
    j = json.loads((reference / "test" / "manifest.json").read_text())["judgments"]
    d = run(emb_path, reference / "test", "m")["metric_details"]["pairwise_auc"]
    assert d["judged_population_prevalence"] == pytest.approx(
        j["n_positive_pairs"] / (j["n_positive_pairs"] + j["n_negative_pairs"]))
    assert d["population_positive_rate"] == pytest.approx(j["n_positive_pairs"] / j["n_possible_pairs"])
    assert d["unjudged_share"] == pytest.approx(j["n_unjudged_pairs"] / j["n_possible_pairs"])
    assert d["population_positive_rate"] < d["judged_population_prevalence"]       # неразмеченных много


def test_large_unjudged_share_moves_population_rate_not_judged_quantities(reference, emb_path, tmp_path):
    j = json.loads((reference / "test" / "manifest.json").read_text())["judgments"]
    base = run(emb_path, reference / "test", "m")
    inflated = j["n_unjudged_pairs"] * 50
    ref2 = with_judgments(reference, tmp_path, n_unjudged_pairs=inflated,
                          n_possible_pairs=j["n_positive_pairs"] + j["n_negative_pairs"] + inflated)
    big = run(emb_path, ref2, "m")
    b, g = base["metric_details"]["pairwise_auc"], big["metric_details"]["pairwise_auc"]
    assert g["unjudged_share"] > b["unjudged_share"]
    assert g["population_positive_rate"] < b["population_positive_rate"] / 5     # доля по всем парам падает
    assert g["judged_population_prevalence"] == b["judged_population_prevalence"]    # условные величины не меняются
    assert big["metrics"]["pair_pr_auc_at_judged_prevalence"] == base["metrics"]["pair_pr_auc_at_judged_prevalence"]


def test_reweighting_target_is_the_judged_prevalence_not_the_population_rate(reference, emb_path):
    from evaluation.retrieval import load_pairs, _scored
    from evaluation.io import load_embeddings
    emb = load_embeddings(emb_path)
    emb = emb.loc[pd.read_csv(reference / "test" / "eval_hosts.csv", dtype=str).host_id]
    pairs = load_pairs(reference / "test" / "pairs.csv")
    j = json.loads((reference / "test" / "manifest.json").read_text())["judgments"]
    judged = j["n_positive_pairs"] / (j["n_positive_pairs"] + j["n_negative_pairs"])
    out = pair_auc(emb, pairs, "test", judged_prevalence=judged)
    p = _scored(emb, pairs, "test")
    assert out["pr_auc_at_judged_prevalence"] == pytest.approx(
        reweighted_average_precision(p.label.to_numpy(), p.score.to_numpy(), judged))
    population = j["n_positive_pairs"] / j["n_possible_pairs"]
    assert out["pr_auc_at_judged_prevalence"] != pytest.approx(
        reweighted_average_precision(p.label.to_numpy(), p.score.to_numpy(), population))


def test_pair_auc_has_no_natural_prevalence_argument():
    with pytest.raises(TypeError):
        pair_auc(pd.DataFrame(), pd.DataFrame(), natural_prevalence=0.1)


def test_link_natural_prevalence_is_over_all_cohort_pairs(reference, emb_path, processed):
    """Метки связей полные (пара либо общалась, либо нет), поэтому настоящая доля определена."""
    r = run(emb_path, reference / "test", "m")
    n = r["n_hosts"]
    ed = pd.read_csv(reference / "test" / "edges.csv")
    expected = (ed.label == 1).sum() / (n * (n - 1) / 2)
    assert r["metric_details"]["future_link_prediction"]["natural_prevalence"] == pytest.approx(expected)


# ---------------------------------------------------------------- оцениваемые хосты (когорта train_only)
def test_evaluator_scores_exactly_the_evaluable_hosts(reference, emb_path, processed):
    r = run(emb_path, reference / "test", "m")
    ev = pd.read_csv(reference / "test" / "eval_hosts.csv").host_id.tolist()
    assert r["n_hosts"] == len(ev) and r["n_embedded_hosts"] >= len(ev)
    assert r["reference_info"]["cohort_policy"] == "train_only" and r["reference_info"]["n_evaluable"] == len(ev)


def test_embedding_missing_an_evaluable_host_is_refused(reference, emb_path, tmp_path):
    df = pd.read_csv(emb_path, dtype={"host_id": str})
    ev = pd.read_csv(reference / "test" / "eval_hosts.csv", dtype=str).host_id.tolist()
    short = df[df.host_id != ev[0]]
    p = tmp_path / "short.csv"
    short.to_csv(p, index=False)
    with pytest.raises(ValueError, match="оцениваемых хостов"):
        run(p, reference / "test", "m")


def test_extra_non_evaluable_hosts_in_the_embedding_do_not_change_scores(reference, emb_path, tmp_path):
    df = pd.read_csv(emb_path, dtype={"host_id": str})
    extra = pd.concat([df, df.iloc[:3].assign(host_id=["zz1", "zz2", "zz3"])])
    p = tmp_path / "extra.csv"
    extra.to_csv(p, index=False)
    a, b = run(emb_path, reference / "test", "m"), run(p, reference / "test", "m")
    assert b["n_hosts"] == a["n_hosts"] and b["n_embedded_hosts"] == a["n_embedded_hosts"] + 3
    assert all(a["metrics"][k] == pytest.approx(b["metrics"][k], abs=1e-9) for k in a["metrics"])


def test_the_scored_host_list_is_part_of_the_reference_fingerprint(reference, emb_path, tmp_path):
    """eval_hosts.csv решает, кого оценивать, поэтому его изменение меняет отпечаток."""
    r = run(emb_path, reference / "test", "m")
    assert "eval_hosts.csv" in r["reference_info"]["files_sha256"]
    other = tmp_path / "ref"
    shutil.copytree(reference / "test", other)
    hosts = pd.read_csv(other / "eval_hosts.csv", dtype=str)
    hosts.iloc[:-1].to_csv(other / "eval_hosts.csv", index=False)               # на одного хоста меньше
    try:
        r2 = run(emb_path, other, "m")
    except ValueError:
        return                                                                  # несогласованный эталон отклоняется сразу
    assert r2["reference_info"]["files_sha256"]["eval_hosts.csv"] != r["reference_info"]["files_sha256"]["eval_hosts.csv"]
