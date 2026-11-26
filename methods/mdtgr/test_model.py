"""Модель, композиция блоков, сохранение и обучение на маленьком синтетическом наборе."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from methods.mdtgr.dataset import ShardSet, window_tensors
from methods.mdtgr.infer import export, load_checkpoint, write_vectors
from methods.mdtgr.model import MDTGR
from methods.mdtgr.train import OFFICIAL_TEST_START, TrainConfig, train


@pytest.fixture(scope="module")
def shards(mdtgr_shards):
    return mdtgr_shards


@pytest.fixture(scope="module")
def model(shards):
    torch.manual_seed(0)
    return MDTGR(shards.n_hosts, shards.n_partners, shards.n_services,
                 int(shards.manifest["behavior_width"]), dim=16, code_dim=8, bag_dim=8,
                 service_dim=4, hidden=32, dropout=0.0)


def test_encode_shapes_and_norms(shards, model):
    tensors = window_tensors(shards.shard("validation"))
    window, code = model.encode(tensors)
    assert window.shape == (shards.n_hosts, 16)
    assert code.shape == (shards.n_hosts, 8)
    assert torch.allclose(window.norm(dim=1), torch.ones(shards.n_hosts), atol=1e-5)
    assert torch.allclose(code.norm(dim=1), torch.ones(shards.n_hosts), atol=1e-5)
    assert torch.isfinite(window).all() and torch.isfinite(code).all()


def test_code_block_is_shared_between_windows(shards, model):
    """Персональный код не зависит от окна — это и есть узнаваемость через время."""
    a = model.encode(window_tensors(shards.shard("train")))[1]
    b = model.encode(window_tensors(shards.shard("validation")))[1]
    assert torch.equal(a, b)


def test_compose_weights_control_blocks(shards, model):
    tensors = window_tensors(shards.shard("validation"))
    window, code = model.encode(tensors)
    anchor = tensors.anchor

    one = model.compose(anchor, code, window, (1.0, 0.0, 0.0))
    width = int(shards.manifest["behavior_width"])
    assert one.shape == (shards.n_hosts, width + 8 + 16)
    assert torch.count_nonzero(one[:, width + 8:]) == 0
    assert torch.count_nonzero(one[:, width:width + 8]) == 0

    full = model.compose(anchor, code, window, (1.0, 1.0, 1.0))
    # общий масштаб вектора не влияет на косинус, поэтому перебор весов — это перебор пропорций
    twice = model.compose(anchor, code, window, (2.0, 2.0, 2.0))
    cos = F.cosine_similarity(full, twice, dim=1)
    assert torch.allclose(cos, torch.ones_like(cos), atol=1e-5)


def test_state_roundtrip_is_exact(shards, model, tmp_path):
    state = {"model": model.state()}
    path = tmp_path / "model.pt"
    torch.save(state, path)
    restored = load_checkpoint(path)
    tensors = window_tensors(shards.shard("validation"))
    with torch.no_grad():
        a, ca = model.encode(tensors)
        b, cb = restored.encode(tensors)
    assert torch.allclose(a, b, atol=1e-6)
    assert torch.equal(ca, cb)


def test_export_writes_expected_files(shards, model, tmp_path):
    written = export(shards, model, tmp_path, (1.0, 1.0, 1.0))
    train = pd.read_csv(written["train"])
    assert train["host_id"].tolist() == shards.hosts
    assert all(c.startswith("emb_") for c in train.columns[1:])

    val = pd.read_csv(written["validation"])
    assert val["host_id"].tolist() == shards.eval_hosts("validation")
    assert val.shape[0] == int(shards.shard("validation").evaluable.sum())
    assert np.isfinite(val.iloc[:, 1:].to_numpy(dtype=float)).all()
    assert train.shape[1] == val.shape[1]

    with pytest.raises(ValueError):
        write_vectors(tmp_path / "bad.csv", shards.hosts[:2], np.zeros((3, 4)))
    with pytest.raises(ValueError):
        bad = np.zeros((2, 3))
        bad[0, 0] = np.nan
        write_vectors(tmp_path / "nan.csv", shards.hosts[:2], bad)


def test_train_guard_rejects_official_test_window(shards, tmp_path):
    copied = ShardSet(shards.root)
    copied.manifest["window_times"]["test"] = [OFFICIAL_TEST_START, OFFICIAL_TEST_START + 3600]
    with pytest.raises(ValueError, match="официальный test"):
        train(copied, TrainConfig(episodes=["test"], epochs=1, dim=8, code_dim=4, bag_dim=4,
                                  service_dim=4, hidden=16), tmp_path / "run")


def test_tune_weights_writes_files_and_best(shards, model, tmp_path, monkeypatch):
    """Перебор весов: пишет оба файла на каждый вариант и выбирает лучший по validation_score."""
    from evaluation import submit
    from methods.mdtgr import infer

    calls = []

    def fake_validation_score(emb_file, validation_file=None):
        emb = pd.read_csv(emb_file)
        later = pd.read_csv(validation_file)
        calls.append((len(emb), len(later)))
        return float(len(later)), {"Поведение": 1.0}

    monkeypatch.setattr(submit, "validation_score", fake_validation_score)
    result = infer.tune_weights(shards, model, "validation", grid=[1.0, 2.0], anchor_grid=[1.0],
                                window_grid=[1.0], work_dir=tmp_path)

    assert len(calls) == 2
    assert all(n_emb == shards.n_hosts for n_emb, _ in calls)
    assert all(n_later == int(shards.shard("validation").evaluable.sum()) for _, n_later in calls)
    assert len(result["table"]) == 2
    assert (tmp_path / "weights.json").exists()
    saved = json.loads((tmp_path / "weights.json").read_text(encoding="utf-8"))
    assert tuple(saved["best"]["weights"]) == tuple(result["best"]["weights"])


def test_train_smoke(shards, tmp_path):
    cfg = TrainConfig(episodes=["test"], dev="validation", epochs=2, anchors=64, lr=3e-3,
                      eval_every=1, dim=8, code_dim=4, bag_dim=4, service_dim=4, hidden=16,
                      link_weight=0.0, log=False)
    result = train(shards, cfg, tmp_path / "run")
    assert tmp_path.joinpath("run", "best.pt").exists()
    assert tmp_path.joinpath("run", "last.pt").exists()
    history = json.loads((tmp_path / "run" / "history.json").read_text(encoding="utf-8"))
    assert [r["epoch"] for r in history] == [1, 2]
    assert all(np.isfinite(r["partner"]) for r in history)
    assert history[-1]["dev"]["n_hosts"] >= 0
