import bz2
import json

import numpy as np
import pandas as pd
import pytest

from pipeline import fixtures
from pipeline.conftest import CONFIG
from pipeline.data import ProcessedData
from pipeline.lanl_netflow import PipelineError
from pipeline.prepare import describe, detect_kind, prepare


def test_outputs_and_manifest(processed):
    m = json.loads((processed / "manifest.json").read_text())
    assert m["synthetic"] is True and m["data_kind"] == "synthetic_fixture"
    assert m["provenance_verified"] is False and m["real_lanl_declared"] is False
    assert m["source_format"] == "lanl-uhn-2017-netflow"
    assert m["canonical_mapping"]["bytes"] == "SrcBytes"
    assert all(len(i["sha256"]) == 64 for i in m["inputs"])
    for w in ("train", "validation", "test"):
        assert (processed / "flows" / f"{w}.csv").exists()
        assert (processed / "graph" / f"edges_{w}.csv").exists()


def test_flows_stay_inside_their_windows(processed):
    d = ProcessedData(processed)
    for w, (lo, hi) in CONFIG.windows.items():
        f = d.flows(w)
        assert f.ts.min() >= lo and f.ts.max() < hi
        assert ((f.ts + f.duration) < hi).all()          # nothing straddles the boundary


def test_no_temporal_leakage_between_windows(processed):
    d = ProcessedData(processed)
    assert d.flows("train").ts.max() < d.flows("validation").ts.min()
    assert d.flows("validation").ts.max() < d.flows("test").ts.min()
    assert d.manifest["leakage_checks"]["max_time_train_lt_min_time_test"] is True


def test_cohort_is_train_active_and_evaluable_hosts_are_active_in_their_window(processed):
    d = ProcessedData(processed)
    cohort = pd.read_csv(processed / "cohort.csv")
    assert (cohort["events_train"] >= CONFIG.min_events_per_window).all()
    assert cohort.host_id.is_unique and cohort.host_id.tolist() == sorted(cohort.host_id)
    assert d.cohort_policy == "train_only"
    for w in ("validation", "test"):
        ev = cohort[cohort[f"evaluable_{w}"]]
        assert (ev[f"events_{w}"] >= CONFIG.min_events_per_window).all()
        assert d.evaluable(w) == ev.host_id.tolist()
        f = d.flows(w)
        assert set(d.evaluable(w)) <= set(f.src) | set(f.dst)
    assert d.evaluable("train") == d.cohort


def test_graph_is_undirected_cohort_only_no_self_loops(processed):
    d = ProcessedData(processed)
    members = set(d.cohort)
    for w in d.windows:
        g = d.graph(w)
        assert (g.host_a < g.host_b).all() and not g.duplicated(["host_a", "host_b"]).any()
        assert set(g.host_a) <= members and set(g.host_b) <= members
        assert (g.flows > 0).all()


def test_training_inputs_are_train_window_only(processed):
    d = ProcessedData(processed)
    assert d.training_flows().ts.max() < CONFIG.windows["train"][1]
    assert d.training_graph().equals(d.graph("train"))


def test_deterministic_rerun(raw_fixture, tmp_path):
    files = sorted(raw_fixture.glob("netflow_day-*.bz2"))
    a = prepare(files, CONFIG, tmp_path / "a")
    b = prepare(files, CONFIG, tmp_path / "b", chunksize=3000)    # different chunking
    assert a["cohort"]["sha256"] == b["cohort"]["sha256"]
    for w in ("train", "validation", "test"):
        da = pd.read_csv(tmp_path / "a" / "flows" / f"{w}.csv").sort_values(list(pd.read_csv(tmp_path / "a" / "flows" / f"{w}.csv").columns)).reset_index(drop=True)
        db = pd.read_csv(tmp_path / "b" / "flows" / f"{w}.csv").sort_values(list(da.columns)).reset_index(drop=True)
        pd.testing.assert_frame_equal(da, db)


def test_refuses_to_overwrite(raw_fixture, processed):
    with pytest.raises(PipelineError, match="overwrite"):
        prepare(sorted(raw_fixture.glob("*.bz2")), CONFIG, processed)


def test_synthetic_marker_cannot_be_relabelled_as_lanl(raw_fixture):
    files = sorted(raw_fixture.glob("*.bz2"))
    assert detect_kind(files, None) == "synthetic_fixture"
    with pytest.raises(PipelineError, match="SYNTHETIC_FIXTURE"):
        detect_kind(files, "lanl", "someone said so")


def test_unmarked_files_are_unverified_never_lanl(tmp_path):
    p = tmp_path / "x.bz2"
    p.write_bytes(b"")
    assert detect_kind([p], None) == "unverified"


def test_empty_window_has_clear_error(raw_fixture, tmp_path):
    from pipeline.windows import PipelineConfig
    cfg = PipelineConfig(windows={"train": (0, 21600), "test": (900000, 900100)})
    with pytest.raises(PipelineError, match="contains no flows"):
        prepare(sorted(raw_fixture.glob("*.bz2")), cfg, tmp_path / "o")


def test_cohort_too_small(raw_fixture, tmp_path):
    from pipeline.windows import PipelineConfig
    cfg = PipelineConfig(windows=CONFIG.windows, min_events_per_window=10_000)
    with pytest.raises(PipelineError, match="Cohort too small"):
        prepare(sorted(raw_fixture.glob("*.bz2")), cfg, tmp_path / "o")


def test_straddling_flows_dropped_and_counted(tmp_path):
    rows = [f"{t},{dur},Comp{h:03d},Comp{(h + 1) % 30:03d},6,Port1,80,1,1,10,5"
            for h in range(30) for t, dur in [(10, 1), (20, 1), (100, 1), (110, 1), (299, 50), (350, 1)]]
    raw = tmp_path / "raw"
    raw.mkdir()
    with bz2.open(raw / "netflow_day-01.bz2", "wt") as f:
        f.write("\n".join(rows) + "\n")
    from pipeline.windows import PipelineConfig
    cfg = PipelineConfig(windows={"train": (0, 300), "test": (300, 400)}, min_events_per_window=1)
    m = prepare([raw / "netflow_day-01.bz2"], cfg, tmp_path / "o")
    assert m["counts"]["straddling_dropped_train"] == 30      # t=299, duration 50 crosses 300
    assert m["windows"]["train"]["n_flows"] == 120       # 4 per host


def test_describe_reports_time_range(raw_fixture):
    out = describe(sorted(raw_fixture.glob("*.bz2")))
    assert len(out["files"]) == 3 and out["files"][0]["min_time"] >= 0
    assert out["files"][0]["n_hosts"] > 0
