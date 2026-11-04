import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pytest

from pipeline.data import ProcessedData
spec = importlib.util.spec_from_file_location('temporal_stable_hybrid',
    Path(__file__).resolve().parents[1] / 'scoreboard' / 'temporal_stable_hybrid.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
TemporalStableHybrid, stable_projection = module.TemporalStableHybrid, module.stable_projection


def test_projection_prefers_stable_direction():
    rng = np.random.default_rng(4)
    signal = rng.normal(size=500)
    x = np.stack([np.column_stack([signal, rng.normal(size=500)]) for _ in range(4)])
    mean, projection = stable_projection(x, np.full((4, 500), 10), dim=1)
    assert abs(projection[0, 0]) > 4 * abs(projection[1, 0])
    assert np.isfinite(mean).all()


def test_projection_requires_temporal_pairs():
    activity = np.array([[5, 0], [0, 5]])
    with pytest.raises(ValueError, match='двух срезах'):
        stable_projection(np.ones((2, 2, 3)), activity)


def test_fit_train_only_shared_coordinates_and_host_order(processed, monkeypatch, tmp_path):
    data = ProcessedData(processed)
    flows = data.flows
    def train_only(window):
        assert window == 'train'
        return flows(window)
    monkeypatch.setattr(data, 'flows', train_only)
    model = TemporalStableHybrid(dim=4, stable_dim=3, splits=2, min_flows=1).fit(data)
    monkeypatch.setattr(data, 'flows', flows)
    train = model.embed(data, 'train').to_numpy()
    model.cache.clear()
    np.testing.assert_allclose(model.embed(data, 'train').to_numpy(), train, atol=1e-10)
    hosts = data.evaluable('validation')
    result = model.embed(data, 'validation', hosts)
    assert result.shape == (len(hosts), model.embed(data, 'train').shape[1])
    assert np.isfinite(result.to_numpy()).all()
    np.testing.assert_allclose(model.embed(data, 'validation', hosts[::-1]).to_numpy(), result.to_numpy()[::-1])
    path = tmp_path / 'model.joblib'
    joblib.dump(model.__dict__, path)
    loaded = TemporalStableHybrid.load(path)
    loaded.cache.clear()
    np.testing.assert_allclose(loaded.embed(data, 'validation', hosts).to_numpy(), result.to_numpy())
    portable = tmp_path / 'model.npz'
    model.export(portable)
    restored = TemporalStableHybrid.load(portable)
    np.testing.assert_allclose(restored.embed(data, 'train').to_numpy(), train, atol=1e-10)
    np.testing.assert_allclose(restored.embed(data, 'validation', hosts).to_numpy(), result.to_numpy(), atol=1e-10)
    with pytest.raises(ValueError, match='когорте'):
        model.embed(data, 'validation', ['missing'])
