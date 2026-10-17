"""Регрессия: непроверенные и синтетические данные никогда не выдаются за LANL."""
import bz2
import json
import shutil

import numpy as np
import pytest

from pipeline.lanl_netflow import PipelineError
from pipeline.prepare import prepare
from pipeline.provenance import BADGE, KIND_LANL_DECLARED, KIND_SYNTHETIC, KIND_UNVERIFIED, classify, fixture_signature
from pipeline.reference import build_reference
from pipeline.windows import PipelineConfig

CFG = PipelineConfig(windows={"train": (0, 300), "test": (400, 700)}, min_events_per_window=1)


def write_netflow(directory, name="netflow_day-02.bz2", n_hosts=40, seed=0, ids=None):
    """Маленький файл NetFlow, где ID устройств НЕ идут прогрессией генератора."""
    rng = np.random.default_rng(seed)
    nums = rng.choice(np.arange(100000, 999999), n_hosts, replace=False) if ids is None else ids
    hosts = [f"Comp{n}" for n in nums]
    rows = []
    for t0 in (0, 450):
        for i, h in enumerate(hosts):
            for k in range(3):
                rows.append(f"{t0 + 10 + k * 20 + i % 7},1,{h},{hosts[(i + 1 + k) % n_hosts]},6,Port{40000 + i},443,2,2,100,50")
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / name
    with bz2.open(p, "wt") as f:
        f.write("\n".join(rows) + "\n")
    return p


def test_unmarked_non_lanl_input_is_unverified_by_default(tmp_path):
    p = write_netflow(tmp_path / "raw")
    m = prepare([p], CFG, tmp_path / "out")
    assert m["data_kind"] == KIND_UNVERIFIED == m["provenance"]["kind"]
    assert m["synthetic"] is False and m["real_lanl_declared"] is False and m["provenance_verified"] is False
    assert m["provenance"]["basis"] == "no_declaration"


def test_lanl_requires_explicit_declaration_and_source_note(tmp_path):
    p = write_netflow(tmp_path / "raw")
    with pytest.raises(PipelineError, match="--source-note"):
        classify([p], "lanl")
    with pytest.raises(PipelineError, match="--source-note"):
        classify([p], "lanl", "   ")
    c = classify([p], "lanl", "downloaded from the LANL form, 2026-10-01")
    assert c["kind"] == KIND_LANL_DECLARED and c["verified"] is False
    assert c["source_note"].startswith("downloaded") and c["basis"] == "operator_declaration"


def test_lanl_declaration_requires_lanl_file_names(tmp_path):
    p = write_netflow(tmp_path / "raw", name="my_traffic.bz2")
    with pytest.raises(PipelineError, match="netflow_day-NN"):
        classify([p], "lanl", "from somewhere")
    assert classify([p])["kind"] == KIND_UNVERIFIED          # без заявления — просто unverified


def test_declared_lanl_is_stamped_declared_not_verified_in_every_artifact(tmp_path):
    p = write_netflow(tmp_path / "raw")
    m = prepare([p], CFG, tmp_path / "out", data_kind="lanl", source_note="LANL form, 2026-10-01")
    assert m["data_kind"] == KIND_LANL_DECLARED and m["real_lanl_declared"] is True
    assert m["provenance_verified"] is False and "not proof" in m["provenance"]["verification"]


def test_fixture_with_marker_removed_is_still_recognised_as_synthetic(raw_fixture, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for f in raw_fixture.glob("netflow_day-*.bz2"):
        shutil.copy(f, raw / f.name)                           # маркер НЕ копируется
    files = sorted(raw.glob("*.bz2"))
    c = classify(files)
    assert c["kind"] == KIND_SYNTHETIC and c["basis"] == "fixture_signature"
    with pytest.raises(PipelineError, match="signature"):
        classify(files, "lanl", "pretending")


def test_marker_plus_lanl_claim_is_refused(raw_fixture):
    with pytest.raises(PipelineError, match="SYNTHETIC_FIXTURE"):
        classify(sorted(raw_fixture.glob("*.bz2")), "lanl", "pretending")


def test_fixture_signature_function():
    assert fixture_signature({f"Comp{100000 + 7 * i:06d}" for i in range(50)})
    rng = np.random.default_rng(1)
    assert not fixture_signature({f"Comp{n}" for n in rng.choice(np.arange(100000, 999999), 50, replace=False)})
    assert not fixture_signature({f"Comp{100000 + 7 * i:06d}" for i in range(5)})       # слишком мало, чтобы делать вывод
    assert not fixture_signature({"Mail", "Comp100007"})


def test_provenance_flows_into_reference_manifests(tmp_path):
    p = write_netflow(tmp_path / "raw", n_hosts=60)
    cfg2 = PipelineConfig(windows={"train": (0, 300), "test": (400, 700)}, min_events_per_window=1)
    prepare([p], cfg2, tmp_path / "out")
    build_reference(tmp_path / "out", tmp_path / "ref", ks=(2, 5))
    ref = json.loads((tmp_path / "ref" / "test" / "manifest.json").read_text())
    assert ref["data_kind"] == KIND_UNVERIFIED and ref["synthetic"] is False
    assert ref["real_lanl_declared"] is False and ref["provenance_verified"] is False
    assert ref["provenance"]["basis"] == "no_declaration"


def test_badges_state_what_is_and_is_not_known():
    assert "SYNTHETIC" in BADGE[KIND_SYNTHETIC]
    assert "UNVERIFIED" in BADGE[KIND_UNVERIFIED] and "not presented as LANL" in BADGE[KIND_UNVERIFIED]
    assert "declared" in BADGE[KIND_LANL_DECLARED] and "not verified" in BADGE[KIND_LANL_DECLARED]
