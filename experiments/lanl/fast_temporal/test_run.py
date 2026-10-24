"""Проверки границ эксперимента и сохранности исходных моделей."""
import ast
import importlib.util
import json
from pathlib import Path

import pytest

from experiments.lanl.fast_temporal.run import PACKAGE, check_dataset, materialize


def test_materialize_preserves_existing_work(tmp_path):
    work = tmp_path / 'work'
    materialize(work)
    sentinel = work / 'model.pt'
    sentinel.write_bytes(b'existing model')
    with pytest.raises(ValueError, match='пустую папку'):
        materialize(work)
    assert sentinel.read_bytes() == b'existing model'
    assert (work / 'hybrid_v6_fast/model.py').exists()


def test_templates_compile_and_use_configured_paths(tmp_path, monkeypatch):
    work = tmp_path / 'work'
    materialize(work)
    monkeypatch.setenv('PT_REPO_ROOT', str(PACKAGE.parents[2]))
    monkeypatch.setenv('PT_RUN_ROOT', str(work))
    monkeypatch.setenv('PT_LANL_DIR', str(tmp_path / 'dataset'))
    for path in work.rglob('*.py'):
        source = path.read_text()
        ast.parse(source, filename=str(path))
        assert '/Volumes/KINGSTON' not in source
        assert '/Users/daniilgladyshev' not in source
    spec = importlib.util.spec_from_file_location('test_feature_core', work / 'hybrid_v4/train.py')
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    assert core.DATA == tmp_path / 'dataset'
    assert core.PREVIOUS == work / 'base_embeddings'


def test_reject_changed_temporal_protocol(tmp_path):
    processed = tmp_path / 'processed'
    processed.mkdir()
    (processed / 'manifest.json').write_text(json.dumps({'config': {
        'windows': {'train': [126000, 133200], 'validation': [129600, 131400], 'test': [131400, 133200]},
        'cohort_policy': 'train_only', 'min_events_per_window': 300}}))
    with pytest.raises(ValueError, match='окна или когорта'):
        check_dataset(tmp_path)


def test_disabled_gate_preserves_embedding_and_enabled_gate_is_bounded():
    torch = pytest.importorskip('torch')
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    path = PACKAGE / 'templates/hybrid_v6_fast/model.py'
    spec = importlib.util.spec_from_file_location('test_fast_gate', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    x = np.random.default_rng(7).normal(size=(32, sum(module.SIZES)))
    assert np.array_equal(module.apply_gate(x, None, None, 0), x)
    scaler = StandardScaler().fit(module.reliability(x))
    gate = module.Gate()
    with torch.no_grad():
        gate.net[-1].bias[:] = torch.tensor([-100., -2., 0., 2., 100.])
    y = module.apply_gate(x, gate, scaler, 1)
    ratio = y / x
    assert np.isfinite(y).all()
    assert ratio.min() >= 0.5 - 1e-12 and ratio.max() <= 1.5 + 1e-12
