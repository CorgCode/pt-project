"""temporal_encoder с encoder="cca": без torch, оси подобраны по двум половинам train."""
import importlib.util
from pathlib import Path

import numpy as np

from pipeline.data import ProcessedData

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("temporal_encoder", ROOT / "scoreboard" / "temporal_encoder.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cca_encoder_needs_no_torch_and_keeps_host_identity(processed, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "torch", None)          # import torch упадёт, если CCA его тронет
    te = _load()
    data = ProcessedData(processed)
    m = te.TemporalEncoder(pca_dim=16, encoder="cca").fit(data)
    x = m.embed(data, "validation", data.evaluable("validation"))
    assert x.shape[1] == 16 and np.isfinite(x.to_numpy()).all()
    assert (np.diff(m.corr) <= 1e-12).all() and m.corr.max() <= 1 + 1e-9
