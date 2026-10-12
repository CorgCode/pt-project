"""Cohort selection: no future information in the default policy; the stable-host policy is explicit."""
import bz2
import json

import numpy as np
import pandas as pd
import pytest

from pipeline.data import ProcessedData
from pipeline.lanl_netflow import PipelineError
from pipeline.prepare import prepare
from pipeline.reference import build_reference
from pipeline.windows import PipelineConfig

W = {"train": (0, 1000), "validation": (1000, 2000), "test": (2000, 3000)}
STABLE = [f"S{i:02d}" for i in range(40)]            # active in every window
EARLY = [f"E{i:02d}" for i in range(12)]            # train only
MID = [f"M{i:02d}" for i in range(12)]              # train + validation, gone in test
PEERS = [f"P{i:02d}" for i in range(15)]


def rows(hosts, t0, per_host=6, seed=0, heavy=1):
    rng = np.random.default_rng(seed)
    out = []
    for i, h in enumerate(hosts):
        for k in range(per_host * heavy):
            peer = PEERS[int(rng.integers(len(PEERS)))] if k % 2 else STABLE[(i + k + 1) % len(STABLE)]
            out.append(f"{t0 + 20 + 7 * k + i},1,{h},{peer},6,Port{1000 + i},{443 if i % 2 else 80},2,2,{100 + i},50")
    return out


def write(tmp, test_hosts, name="netflow_day-02.bz2"):
    lines = (rows(STABLE + EARLY + MID, 0, heavy=2, seed=1) + rows(STABLE + MID, 1000, seed=2)
             + rows(test_hosts, 2000, seed=3))
    p = tmp / name
    tmp.mkdir(parents=True, exist_ok=True)
    with bz2.open(p, "wt") as f:
        f.write("\n".join(lines) + "\n")
    return p


def cfg(policy, **kw):
    return PipelineConfig(windows=W, min_events_per_window=3, min_cohort_size=25, cohort_policy=policy, **kw)


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    return write(tmp_path_factory.mktemp("raw"), STABLE)


def test_train_only_cohort_and_coverage(raw, tmp_path):
    m = prepare([raw], cfg("train_only"), tmp_path / "o")
    d = ProcessedData(tmp_path / "o")
    assert d.cohort_policy == "train_only"
    assert set(d.cohort) >= set(STABLE + EARLY + MID)
    assert set(EARLY) <= set(d.cohort) and not set(EARLY) & set(d.evaluable("validation"))
    assert set(MID) <= set(d.evaluable("validation")) and not set(MID) & set(d.evaluable("test"))
    assert set(STABLE) <= set(d.evaluable("test"))
    sel = m["cohort"]["selection"]
    assert sel["n_evaluable"]["test"] < sel["n_evaluable"]["validation"] < sel["n_cohort"]
    assert sel["coverage"]["test"] == pytest.approx(sel["n_evaluable"]["test"] / sel["n_cohort"])
    assert sel["n_hosts_active_in_every_window"] == len(set(d.evaluable("test")) & set(d.evaluable("validation")))
    assert "TRAIN window only" in sel["note"]


def test_all_windows_policy_is_explicitly_the_transductive_stable_host_benchmark(raw, tmp_path):
    m = prepare([raw], cfg("all_windows"), tmp_path / "o")
    d = ProcessedData(tmp_path / "o")
    assert set(d.cohort) == set(d.evaluable("test")) == set(d.evaluable("validation"))
    assert not set(EARLY + MID) & set(d.cohort)
    note = m["cohort"]["selection"]["note"]
    assert "transductive" in note and "EVERY window" in note and "survival" in note


def test_train_only_cohort_does_not_depend_on_future_windows(tmp_path):
    a = write(tmp_path / "a", STABLE)                        # everybody active in test
    b = write(tmp_path / "b", STABLE[:30])                   # 10 hosts silent in test
    ma = prepare([a], cfg("train_only"), tmp_path / "oa")
    mb = prepare([b], cfg("train_only"), tmp_path / "ob")
    assert ProcessedData(tmp_path / "oa").cohort == ProcessedData(tmp_path / "ob").cohort       # same population
    assert ma["cohort"]["n_hosts"] == mb["cohort"]["n_hosts"]
    assert mb["cohort"]["evaluable"]["test"] < ma["cohort"]["evaluable"]["test"]               # coverage shows it
    # ... whereas the legacy policy lets future activity change who is in the cohort
    la = prepare([a], cfg("all_windows"), tmp_path / "la")
    lb = prepare([b], cfg("all_windows"), tmp_path / "lb")
    assert la["cohort"]["n_hosts"] != lb["cohort"]["n_hosts"]


def test_default_policy_is_train_only():
    assert PipelineConfig(windows=W).cohort_policy == "train_only"
    with pytest.raises(ValueError, match="cohort_policy"):
        PipelineConfig(windows=W, cohort_policy="whatever")


def test_selection_diagnostic_reports_survivorship(raw, tmp_path):
    m = prepare([raw], cfg("train_only"), tmp_path / "o")
    med = m["cohort"]["selection"]["median_train_events"]
    assert med["active_in_every_window"] is not None and med["not_active_in_every_window"] is not None
    assert all(isinstance(v, float) and v >= 3 for v in med.values())


def test_too_few_evaluable_hosts_in_a_window_is_an_error(tmp_path):
    p = write(tmp_path / "r", STABLE[:10])                   # only 10 hosts active in test
    with pytest.raises(PipelineError, match="evaluable in test"):
        prepare([p], cfg("train_only"), tmp_path / "o")


def test_references_are_built_on_evaluable_hosts_with_coverage(raw, tmp_path):
    prepare([raw], cfg("train_only"), tmp_path / "o")
    d = ProcessedData(tmp_path / "o")
    build_reference(tmp_path / "o", tmp_path / "r", ks=(2, 5))
    for t in ("validation", "test"):
        ev = pd.read_csv(tmp_path / "r" / t / "eval_hosts.csv").host_id.astype(str).tolist()
        assert ev == d.evaluable(t)
        assert set(pd.read_csv(tmp_path / "r" / t / "features.csv", dtype=str).host_id) == set(ev)
        assert set(pd.read_csv(tmp_path / "r" / t / "neighbors.csv", dtype=str).host_id) == set(ev)
        m = json.loads((tmp_path / "r" / t / "manifest.json").read_text())
        assert m["cohort_policy"] == "train_only" and m["n_evaluable"] == len(ev)
        assert m["coverage"] == pytest.approx(len(ev) / len(d.cohort)) and m["cohort_selection"]["note"]
    assert len(d.evaluable("test")) < len(d.evaluable("validation")) < len(d.cohort)
