"""temporal_encoder: учится только на train, детерминирован, peers/services совпадают с hybrid_tfidf."""
import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest

from pipeline.data import ProcessedData

pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scoreboard" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def te():
    return _load("temporal_encoder")


def test_fit_reads_only_train_and_is_deterministic(te, processed, monkeypatch):
    data = ProcessedData(processed)
    flows = data.flows

    def train_only(window):
        assert window == "train", f"fit прочитал окно {window}"
        return flows(window)
    monkeypatch.setattr(data, "flows", train_only)
    a = te.TemporalEncoder(epochs=2, pca_dim=16).fit(data)
    b = te.TemporalEncoder(epochs=2, pca_dim=16).fit(data)
    monkeypatch.setattr(data, "flows", flows)
    hosts = data.evaluable("test")
    ea, eb = a.embed(data, "test", hosts), b.embed(data, "test", hosts)
    assert ea.shape == (len(hosts), 16) and np.isfinite(ea.to_numpy()).all()
    np.testing.assert_allclose(ea.to_numpy(), eb.to_numpy(), atol=1e-5)


def test_without_pca_has_all_blocks(te, processed):
    data = ProcessedData(processed)
    m = te.TemporalEncoder(epochs=1, pca_dim=None).fit(data)
    x = m.embed(data, "validation", data.evaluable("validation"))
    behavioral = m.beh.embed(data, "train", data.cohort).shape[1]
    sizes = [behavioral, m.models["peers"][1].n_components, m.models["services"][1].n_components,
             3 * m.models["services"][1].n_components, 128]
    assert x.shape[1] == sum(sizes)


def test_peers_and_services_match_hybrid_tfidf(te, processed):
    ht = _load("hybrid_tfidf")
    data = ProcessedData(processed)
    m = te.TemporalEncoder(epochs=1).fit(data)
    flows = data.training_flows()
    for kind in ("peers", "services"):
        ref = ht.TfidfSVD(kind, dim=te.SVD_DIM).fit(flows, data.cohort)
        assert list(ref.vocab) == list(m.vocab[kind])
        got = m._train[kind]
        want = ref.embed(flows, data.cohort)
        nonzero = np.linalg.norm(got, axis=1) > 0
        np.testing.assert_allclose(got[nonzero], want[nonzero], atol=1e-8)


def test_more_slices_and_tuning_pick_from_grid(te, processed, reference, tmp_path, monkeypatch):
    from evaluation import submit
    root = tmp_path / "scoreboard"
    shutil.copytree(processed, root / "data" / "processed")
    shutil.copytree(reference, root / "data" / "reference")
    for name, value in (("BOARD", root), ("DATA", root / "data"), ("SUBMISSIONS", root / "submissions"),
                        ("RESULTS", root / "results"), ("N_BOOT", 0)):
        monkeypatch.setattr(submit, name, value)
    data = ProcessedData(processed)
    m = te.TemporalEncoder(epochs=1, pca_dim=8, splits=4).fit(data)
    assert m.embed(data, "test", data.evaluable("test")).shape == (len(data.evaluable("test")), 8)
    monkeypatch.setattr(te, "EPOCHS", 1)
    grid = {"splits": (2,), "behavioral": (2.0,), "neural": (0.5, 1.0), "diffusion": (0.0,)}
    best, rows = te.tune(data, grid, pca_dim=8)
    assert len(rows) == 2 and best.recipe["neural"] in (0.5, 1.0) and best.recipe["diffusion"] == 0.0
