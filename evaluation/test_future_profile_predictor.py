import importlib.util
from pathlib import Path
import numpy as np
import pytest
from pipeline.data import ProcessedData

pytest.importorskip('torch')
spec = importlib.util.spec_from_file_location('future_profile_predictor', Path(__file__).resolve().parents[1] / 'scoreboard/future_profile_predictor.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_train_only_and_deterministic(processed, monkeypatch):
    data = ProcessedData(processed)
    original = data.flows
    def only_train(window):
        assert window == 'train'
        return original(window)
    monkeypatch.setattr(data, 'flows', only_train)
    a = m.FutureProfilePredictor(epochs=1, pca_dim=8).fit(data)
    b = m.FutureProfilePredictor(epochs=1, pca_dim=8).fit(data)
    train = a.embed(data, 'train').to_numpy()
    a._parts_cache.clear()
    np.testing.assert_allclose(train, a.embed(data, 'train').to_numpy(), atol=1e-8)
    monkeypatch.setattr(data, 'flows', original)
    hosts = data.evaluable('validation')
    x = a.embed(data, 'validation', hosts)
    assert np.isfinite(x.to_numpy()).all() and x.shape == (len(hosts), 8)
    np.testing.assert_allclose(x, b.embed(data, 'validation', hosts), atol=1e-6)
    np.testing.assert_allclose(x.to_numpy()[::-1], a.embed(data, 'validation', hosts[::-1]).to_numpy(), atol=1e-8)
    assert len(a.loss_history) == 1 and np.isfinite(a.loss_history).all()
