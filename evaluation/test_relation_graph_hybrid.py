import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline.data import ProcessedData

spec = importlib.util.spec_from_file_location('relation_graph_hybrid', Path(__file__).resolve().parents[1] / 'scoreboard/relation_graph_hybrid.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_relation_direction_and_service():
    index = pd.MultiIndex.from_tuples([('a', 'o:b|6/80'), ('a', 'o:c|17/53'), ('b', 'o:b|6/80')], names=['h', 't'])
    counts = pd.Series([3, 7, 100], index=index)
    out, inn = m.relation_adjacency(counts, ['a', 'b', 'c'], '6/80')
    np.testing.assert_array_equal(out.toarray(), [[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    assert inn[1, 0] == 1
    other, _ = m.relation_adjacency(counts, ['a', 'b', 'c'], excluded=['6/80'])
    assert other[0, 2] == 1 and other[0, 1] == 0


def test_train_only_and_fixed_coordinates(processed, monkeypatch):
    data = ProcessedData(processed)
    original = data.flows
    def only_train(window):
        assert window == 'train'
        return original(window)
    monkeypatch.setattr(data, 'flows', only_train)
    model = m.RelationGraphHybrid(pca_dim=8, top_services=2).fit(data)
    before = model.embed(data, 'train').to_numpy()
    model._parts_cache.clear()
    np.testing.assert_allclose(before, model.embed(data, 'train').to_numpy(), atol=1e-8)
    relations = list(model.relations)
    monkeypatch.setattr(data, 'flows', original)
    hosts = data.evaluable('validation')
    result = model.embed(data, 'validation', hosts).to_numpy()
    assert result.shape == (len(hosts), 8) and np.isfinite(result).all()
    assert model.relations == relations
    np.testing.assert_allclose(result[::-1], model.embed(data, 'validation', hosts[::-1]).to_numpy(), atol=1e-8)
