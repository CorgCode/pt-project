"""Регрессионные тесты протокола запуска экспериментов (ревью #51)."""
import bz2
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from methods.baselines.temporal import default_baselines
from experiments.run_experiment import run_experiment
from pipeline import fixtures
from pipeline.data import ProcessedData
from pipeline.prepare import prepare
from pipeline.provenance import SYNTHETIC_MARKER
from pipeline.reference import build_reference
from pipeline.windows import PipelineConfig

CFG = PipelineConfig(windows={"train": (0, 21600), "validation": (21600, 32400), "test": (32400, 43200)})
FAST = dict(dim=6, n_boot=0, audit_configs=2, null_draws=3)


def rewrite(src_dir, dst_dir, fn, keep_marker=True):
    """Скопировать синтетику, пропустив каждую сырую строку через fn(fields) -> fields | None."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(src_dir.glob("netflow_day-*.bz2")):
        with bz2.open(f, "rt") as fh:
            lines = [l.rstrip("\n").split(",") for l in fh if l.strip()]
        out = [x for x in (fn(l) for l in lines) if x is not None]
        with bz2.open(dst_dir / f.name, "wt") as fh:
            fh.write("\n".join(",".join(x) for x in out) + "\n")
    if keep_marker:
        shutil.copy(src_dir / SYNTHETIC_MARKER, dst_dir / SYNTHETIC_MARKER)
    return dst_dir


def pipeline_for(raw, tmp, **prep):
    prepare(sorted(raw.glob("netflow_day-*.bz2")), prep.pop("config", CFG), tmp / "processed", **prep)
    build_reference(tmp / "processed", tmp / "reference", pairs_per_anchor=10)
    return tmp / "processed", tmp / "reference"


@pytest.fixture(scope="module")
def base_raw(tmp_path_factory):
    d = tmp_path_factory.mktemp("base_raw")
    fixtures.generate(d, n_hosts=130, seed=5)
    return d


# ---------------------------------------------------------------- выбывание хостов при когорте train_only
@pytest.fixture(scope="module")
def attrition(base_raw, tmp_path_factory):
    silent = {f"Comp{100000 + 7 * i:06d}" for i in range(0, 130, 6)}              # ~17% пропадают в test-окне
    raw = rewrite(base_raw, tmp_path_factory.mktemp("attr_raw"),
                  lambda f: None if float(f[0]) >= 32400 and (f[2] in silent or f[3] in silent) else f)
    tmp = tmp_path_factory.mktemp("attr")
    processed, reference = pipeline_for(raw, tmp)
    out = tmp / "run"
    manifest = run_experiment(processed, reference, out, **FAST)
    return dict(silent=silent, processed=processed, reference=reference, out=out, manifest=manifest)


def test_vanished_hosts_are_cohort_members_but_not_evaluable(attrition):
    d = ProcessedData(attrition["processed"])
    assert attrition["silent"] <= set(d.cohort)
    assert not attrition["silent"] & set(d.evaluable("test")) and set(d.evaluable("validation")) >= attrition["silent"]


def test_every_method_is_scored_on_exactly_the_evaluable_hosts(attrition):
    d = ProcessedData(attrition["processed"])
    for name in attrition["manifest"]["methods"]:
        r = json.loads((attrition["out"] / "results" / "test" / f"{name}.json").read_text())
        assert r["n_hosts"] == len(d.evaluable("test")) < r["n_embedded_hosts"] or name == "reference_ceiling"
        assert r["reference_info"]["cohort_policy"] == "train_only"
        assert r["reference_info"]["evaluable_coverage"] == pytest.approx(len(d.evaluable("test")) / len(d.cohort))


def test_identity_uses_only_hosts_active_in_both_windows(attrition):
    d = ProcessedData(attrition["processed"])
    res = json.loads((attrition["out"] / "identity" / "test" / "behavioral.json").read_text())
    assert res["procrustes"]["n_eval_hosts"] + res["procrustes"]["n_alignment_hosts"] == len(d.evaluable("test"))


def test_report_discloses_cohort_policy_and_coverage(attrition):
    text = (attrition["out"] / "report.md").read_text()
    assert "политика когорты `train_only`" in text and "ТОЛЬКО по train-окну" in text
    cov = attrition["manifest"]["cohort"]["coverage"]
    assert cov["test"] < cov["validation"] == 1.0 and f"({cov['test']:.0%})" in text


def test_all_windows_policy_changes_the_population_and_says_so(base_raw, tmp_path, attrition):
    silent = attrition["silent"]
    raw = rewrite(base_raw, tmp_path / "raw", lambda f: None if float(f[0]) >= 32400 and (f[2] in silent or f[3] in silent) else f)
    cfg = PipelineConfig(windows=CFG.windows, cohort_policy="all_windows")
    processed, reference = pipeline_for(raw, tmp_path, config=cfg)
    m = run_experiment(processed, reference, tmp_path / "run", **FAST)
    assert m["cohort"]["policy"] == "all_windows" and m["cohort"]["n_cohort"] < attrition["manifest"]["cohort"]["n_cohort"]
    assert "стабильных хостов" in (tmp_path / "run" / "report.md").read_text()


# ---------------------------------------------------------------- утечка: оцениваемые эмбеддинги зависят только от train
def test_scored_embeddings_do_not_depend_on_future_windows(base_raw, tmp_path):
    a, _ = pipeline_for(base_raw, tmp_path / "a")
    shifted = rewrite(base_raw, tmp_path / "raw_b",                   # меняем ТОЛЬКО потоки после train-окна
                      lambda f: f if float(f[0]) < 21600 else [*f[:6], "9999", f[7], f[8], str(int(float(f[9])) * 7 + 3), f[10]])
    b, _ = pipeline_for(shifted, tmp_path / "b")
    da, db = ProcessedData(a), ProcessedData(b)
    assert da.training_flows().equals(db.training_flows()) and da.cohort == db.cohort
    assert not da.flows("test").equals(db.flows("test"))
    for ba, bb in zip(default_baselines(8, 0), default_baselines(8, 0)):
        ea, eb = ba.fit(da).embed(da, "train"), bb.fit(db).embed(db, "train")
        np.testing.assert_allclose(ea.to_numpy(), eb.to_numpy(), err_msg=ba.name)


# ---------------------------------------------------------------- воспроизводимость
def _canon(path, root):
    t = path.read_text(encoding="utf-8").replace(str(root), "<run>")
    return json.loads(t) if path.suffix == ".json" else t


def test_same_seed_gives_identical_artifacts(processed, reference, tmp_path):
    outs = []
    for k in ("one", "two"):
        out = tmp_path / k
        run_experiment(processed, reference, out, dim=6, seed=3, n_boot=15, audit_configs=2, null_draws=3)
        outs.append(out)
    names = [p.relative_to(outs[0]) for p in sorted(outs[0].rglob("*")) if p.suffix in (".json", ".csv", ".md")]
    assert any(n.name == "report.md" for n in names) and len(names) > 20
    for n in names:
        a, b = _canon(outs[0] / n, outs[0]), _canon(outs[1] / n, outs[1])
        if n.name == "manifest.json" and n.parent.name == "":
            for d in (a, b):
                d["environment"].pop("git_sha", None)
        assert a == b, n


def test_different_seed_changes_random_baseline_but_not_deterministic_ones(processed, reference, tmp_path):
    for k, seed in (("s0", 0), ("s1", 1)):
        run_experiment(processed, reference, tmp_path / k, dim=6, seed=seed, n_boot=0, do_audit=False)
    load = lambda k, m: json.loads((tmp_path / k / "results" / "test" / f"{m}.json").read_text())["metrics"]
    assert load("s0", "random")["triplet_accuracy"] != load("s1", "random")["triplet_accuracy"]
    assert load("s0", "behavioral")["triplet_accuracy"] == load("s1", "behavioral")["triplet_accuracy"]


# ---------------------------------------------------------------- отчёты никогда не путают синтетику / непроверенное / заявленное
def relabelled(base_raw, tmp, seed=1):
    """Тот же трафик, ID устройств заменены случайными, маркер убран: подписи генератора нет."""
    rng = np.random.default_rng(seed)
    ids = sorted({f[2] for f in _lines(base_raw)} | {f[3] for f in _lines(base_raw)})
    new = dict(zip(ids, (f"Comp{n}" for n in rng.choice(np.arange(100000, 999999), len(ids), replace=False))))
    return rewrite(base_raw, tmp, lambda f: [*f[:2], new[f[2]], new[f[3]], *f[4:]], keep_marker=False)


def _lines(raw):
    for f in sorted(raw.glob("netflow_day-*.bz2")):
        with bz2.open(f, "rt") as fh:
            for l in fh:
                if l.strip():
                    yield l.rstrip("\n").split(",")


@pytest.fixture(scope="module")
def other_kinds(base_raw, tmp_path_factory):
    raw = relabelled(base_raw, tmp_path_factory.mktemp("rl_raw"))
    res = {}
    for kind, kw in (("unverified", {}), ("lanl_declared", dict(data_kind="lanl", source_note="unit test, not real LANL"))):
        tmp = tmp_path_factory.mktemp(kind)
        processed, reference = pipeline_for(raw, tmp, **kw)
        m = run_experiment(processed, reference, tmp / "run", **FAST)
        res[kind] = (m, (tmp / "run" / "report.md").read_text())
    return res


def test_unverified_run_is_never_called_lanl(other_kinds):
    m, text = other_kinds["unverified"]
    assert m["data_kind"] == "unverified" and m["synthetic"] is False and m["real_lanl_declared"] is False
    assert "НЕПРОВЕРЕННЫЕ ДАННЫЕ" in text and "за LANL не выдаётся" in text
    assert "LANL (заявлено" not in text and "СИНТЕТИКА" not in text


def test_declared_lanl_run_says_declared_not_verified(other_kinds):
    m, text = other_kinds["lanl_declared"]
    assert m["data_kind"] == "lanl_declared" and m["real_lanl_declared"] is True and m["provenance_verified"] is False
    assert "LANL (заявлено оператором; происхождение пайплайном не проверено)" in text
    assert "unit test, not real LANL" in text and "СИНТЕТИКА" not in text
    assert "не утверждения о других данных" in text


def test_synthetic_run_report_says_nothing_follows_for_lanl(attrition):
    text = (attrition["out"] / "report.md").read_text()
    assert "СИНТЕТИКА" in text and "о LANL ни одно число" in text


def test_every_result_file_carries_the_kind(other_kinds, attrition):
    for out, kind in ((attrition["out"], "synthetic_fixture"),):
        for p in (out / "results" / "test").glob("*.json"):
            assert json.loads(p.read_text())["reference_info"]["data_kind"] == kind


# ---------------------------------------------------------------- внешние методы и утечка
def submission(tmp, processed, name="mymodel", provenance=None):
    d = ProcessedData(processed)
    sub = tmp / "subs" / name
    sub.mkdir(parents=True)
    pd.DataFrame(np.random.default_rng(3).normal(size=(len(d.cohort), 4)),
                 index=pd.Index(d.cohort, name="host_id")).add_prefix("emb_").to_csv(sub / "embeddings.csv")
    if provenance is not None:
        (sub / "provenance.json").write_text(json.dumps(provenance))
    return tmp / "subs"


def test_submission_trained_on_a_later_window_is_refused(processed, reference, tmp_path):
    subs = submission(tmp_path, processed, provenance={"trained_on_windows": ["train", "test"]})
    with pytest.raises(ValueError, match="утечка по времени"):
        run_experiment(processed, reference, tmp_path / "o", submissions_dir=subs, do_audit=False, dim=4, n_boot=0)


def test_undeclared_training_windows_are_flagged_not_trusted(processed, reference, tmp_path):
    m = run_experiment(processed, reference, tmp_path / "o", submissions_dir=submission(tmp_path, processed),
                       do_audit=False, dim=4, n_boot=0)
    assert m["external_provenance"]["mymodel"]["trained_on_windows"] is None
    assert "окна обучения не заявлены" in (tmp_path / "o" / "report.md").read_text()


def test_declared_train_only_submission_is_recorded_as_declared(processed, reference, tmp_path):
    m = run_experiment(processed, reference, tmp_path / "o", do_audit=False, dim=4, n_boot=0,
                       submissions_dir=submission(tmp_path, processed, provenance={"trained_on_windows": ["train"]}))
    assert m["external_provenance"]["mymodel"] == {"trained_on_windows": ["train"], "verified": False}


def test_malformed_provenance_file_is_an_error(processed, reference, tmp_path):
    subs = submission(tmp_path, processed, provenance={"windows": "train"})
    with pytest.raises(ValueError, match="trained_on_windows"):
        run_experiment(processed, reference, tmp_path / "o", submissions_dir=subs, do_audit=False, dim=4, n_boot=0)
