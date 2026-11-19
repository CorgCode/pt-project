"""Лоссы: listwise повторяет формулу метрики и не зависит от монотонных преобразований."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from methods.mdtgr.losses import (partner_listwise_loss, partner_scores, sample_negatives,
                                  temporal_infonce)


def _scores(rows: list[list[float]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32)


def test_listwise_prefers_higher_target():
    targets = torch.tensor([[False, False, True], [True, False, False]])
    blocked = torch.zeros_like(targets)
    good = partner_listwise_loss(
        _scores([[0.1, 0.2, 3.0], [4.0, 1.0, 0.5]]), targets, blocked)
    bad = partner_listwise_loss(
        _scores([[0.1, 0.2, -3.0], [-4.0, 1.0, 0.5]]), targets, blocked)
    assert float(good) < float(bad)


def test_listwise_ignores_blocked_scores():
    """Запрещённый кандидат не влияет на лосс, каким бы большим ни был его счёт."""
    targets = torch.tensor([[False, False, True]])
    blocked = torch.tensor([[False, True, False]])
    a = partner_listwise_loss(_scores([[0.0, 100.0, 2.0]]), targets, blocked)
    b = partner_listwise_loss(_scores([[0.0, -100.0, 2.0]]), targets, blocked)
    assert float(a) == pytest.approx(float(b), abs=1e-7)


def test_listwise_skips_rows_without_targets():
    targets = torch.tensor([[False, False, False]])
    blocked = torch.zeros_like(targets)
    loss = partner_listwise_loss(_scores([[1.0, 2.0, 3.0]]), targets, blocked)
    assert float(loss) == 0.0


def test_listwise_positive_when_target_is_not_first():
    """Цель не лучшая — лосс положителен при любом монотонном сжатии счётчиков."""
    scores = _scores([[0.5, 1.0, 2.0, 4.0]])
    targets = torch.tensor([[False, False, False, True]])
    blocked = torch.zeros_like(targets)
    raw = partner_listwise_loss(scores, targets, blocked, transform="none")
    compressed = partner_listwise_loss(scores, targets, blocked, transform="log1p")
    assert float(raw) > 0 and float(compressed) > 0


def test_partner_scores_masks_self():
    """Сам хост не входит в свои соседи: при пуле из одного хоста счётчики нулевые."""
    from scipy import sparse

    m = sparse.csr_matrix(np.ones((1, 4), dtype=np.float32))
    coo = m.T.tocoo()
    m_t = torch.sparse_coo_tensor(
        torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64)),
        torch.from_numpy(coo.data.astype(np.float32)), coo.shape).coalesce()
    z = torch.nn.functional.normalize(torch.randn(1, 8), dim=1)
    scores = partner_scores(z, torch.tensor([0]), m_t, k=10)
    assert scores.shape == (1, 4)
    assert torch.count_nonzero(scores) == 0


def test_temporal_infonce_and_negatives():
    torch.manual_seed(0)
    a = torch.nn.functional.normalize(torch.randn(20, 8), dim=1)
    mask = torch.ones(20, dtype=torch.bool)
    same = temporal_infonce(a, a.clone(), mask)
    other = temporal_infonce(a, torch.nn.functional.normalize(torch.randn(20, 8), dim=1), mask)
    assert float(same) < float(other)
    assert float(temporal_infonce(a, a, torch.zeros(20, dtype=torch.bool))) == 0.0

    pairs = sample_negatives(100, 500, torch.Generator().manual_seed(1), "cpu")
    assert pairs.shape == (500, 2)
    assert bool((pairs[:, 0] != pairs[:, 1]).all())
