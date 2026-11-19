"""Локальная метрика «новые собеседники» должна совпадать с эталонной."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.partners import load_partners, partner_ranking
from methods.mdtgr.dataset import host_columns, pool_matrices
from methods.mdtgr.evaluate import new_partner_metrics, new_targets
from methods.mdtgr.losses import rows_to_mask


@pytest.fixture(scope="module")
def shards(mdtgr_shards):
    return mdtgr_shards


@pytest.fixture(scope="module")
def window_matrices(shards):
    window = "validation"
    shard, source = shards.shard(window), shards.shard("train")
    m, t, keep = pool_matrices(shard, source)
    host_col = host_columns(shards.partners, shards.hosts, keep, shard.pool())
    return shard, m, t, host_col


def test_new_targets_exclude_old_partners(shards, window_matrices):
    _, m, t, _ = window_matrices
    new = new_targets(t, m)
    assert (new - new.multiply(t > 0)).nnz == 0        # новые цели — подмножество контактов дня
    assert new.multiply(m > 0).nnz == 0                # и не пересекаются со старыми собеседниками


def test_metric_matches_reference_implementation(shards, reference, window_matrices):
    shard, m, t, host_col = window_matrices
    rng = np.random.default_rng(7)
    emb = rng.normal(size=(m.shape[0], 12))

    ours = new_partner_metrics(emb, m, t, host_col)
    frame = pd.DataFrame(emb, index=pd.Index(shards.eval_hosts("validation"), name="host_id"))
    ref = partner_ranking(frame, load_partners(reference / "validation" / "partners.csv"),
                          n_boot=0)["new"]["embedding"]

    assert ours["n_hosts"] == ref["n_hosts"]
    assert ours["mrr"] == pytest.approx(ref["mrr"], abs=1e-12)
    assert ours["recall@10"] == pytest.approx(ref["recall_at_n"], abs=1e-12)


def test_metric_is_scale_invariant(shards, window_matrices):
    """Косинус не зависит от длины вектора: MRR не должен меняться от масштаба."""
    _, m, t, host_col = window_matrices
    rng = np.random.default_rng(3)
    emb = rng.normal(size=(m.shape[0], 16))
    base = new_partner_metrics(emb, m, t, host_col)
    scaled = new_partner_metrics(emb * np.linspace(0.01, 100.0, len(emb))[:, None], m, t, host_col)
    assert scaled["mrr"] == pytest.approx(base["mrr"], abs=1e-9)
    assert scaled["n_hosts"] == base["n_hosts"]


def test_rows_to_mask():
    from scipy import sparse

    mat = sparse.csr_matrix(np.array([[0.0, 2.0, 0.0], [1.0, 0.0, 3.0]]))
    mask = rows_to_mask(mat, np.array([0, 1]), 4)
    assert mask.tolist() == [[False, True, False, False], [True, False, True, False]]
