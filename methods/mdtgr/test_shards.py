"""Срезы окон: словари только по train и совпадение матриц с эталоном метрики."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from methods.mdtgr.dataset import host_columns, pool_matrices


@pytest.fixture(scope="module")
def shards(mdtgr_shards):
    return mdtgr_shards


def _reference_matrix(partners: pd.DataFrame, window: str, order: dict[str, int], universe: pd.Index):
    """Матрица «хост × кандидат» из partners.csv эталона — как в evaluation.partners."""
    p = partners[(partners.window == window) & partners.host_id.isin(order) & partners.partner.isin(universe)]
    rows = p.host_id.map(order).to_numpy()
    cols = universe.get_indexer(p.partner)
    return sparse.csr_matrix((p["count"].to_numpy(float), (rows, cols)),
                             shape=(len(order), len(universe)))


def test_manifest_and_hosts(shards, processed):
    hosts = pd.read_csv(processed / "cohort.csv", dtype=str)["host_id"].tolist()
    assert shards.hosts == hosts
    assert shards.n_hosts == len(hosts)
    assert shards.windows == ["train", "validation", "test"]
    for window in shards.windows:
        info = shards.manifest["shards"][window]
        assert info["n_evaluable"] == int(shards.shard(window).evaluable.sum())
        assert info["behavior_width"] == shards.manifest["behavior_width"]
        assert len(shards.manifest["window_times"][window]) == 2


def test_vocab_built_from_train_only(processed, shards):
    """Имена, впервые появившиеся в позднем окне, не должны попасть в словарь."""
    train = pd.read_csv(processed / "flows" / "train.csv", usecols=["src", "dst"], dtype=str)
    test = pd.read_csv(processed / "flows" / "test.csv", usecols=["src", "dst"], dtype=str)
    seen_train = set(train.src) | set(train.dst)
    assert set(shards.partners) == seen_train

    late_only = (set(test.src) | set(test.dst)) - seen_train
    assert not (late_only & set(shards.partners))


def test_contacts_are_undirected_without_loops(processed, shards):
    window = "validation"
    shard = shards.shard(window)
    contacts = shard.contacts.tocsr()
    assert (contacts != contacts.T).nnz == 0, "контакты должны быть ненаправленными"
    assert contacts.diagonal().sum() == 0, "петли в контакты не входят"

    flows = pd.read_csv(processed / "flows" / f"{window}.csv", usecols=["src", "dst"], dtype=str)
    flows = flows[flows.src != flows.dst]
    names = shards.hosts
    index = {h: i for i, h in enumerate(names)}
    rows = flows.src.map(index)
    cols = flows.dst.map(index)
    keep = rows.notna() & cols.notna()
    both = pd.concat([pd.DataFrame({"h": rows[keep], "p": cols[keep]}),
                      pd.DataFrame({"h": cols[keep], "p": rows[keep]})])
    counts = both.groupby(["h", "p"]).size()
    partner_pos = {p: i for i, p in enumerate(shards.partners)}

    # в когорте контакты — в точности неориентированные счётчики потоков
    cohort = contacts[:, [partner_pos[h] for h in names]]
    expected = sparse.csr_matrix(
        (counts.to_numpy(float), (counts.index.get_level_values(0).to_numpy(),
                                  counts.index.get_level_values(1).to_numpy())),
        shape=(len(names), len(names)))
    assert (cohort != expected).nnz == 0


def test_pool_and_metric_matrices_match_reference(shards, reference):
    """M — контакты pool × кандидаты, T — контакты дня; сравниваем с partners.csv эталона."""
    window = "validation"
    shard, source = shards.shard(window), shards.shard("train")
    ref = pd.read_csv(reference / window / "partners.csv", dtype={"host_id": str, "partner": str})
    eval_hosts = pd.read_csv(reference / window / "eval_hosts.csv", dtype=str)["host_id"].tolist()

    ours_hosts = shards.eval_hosts(window)
    assert set(ours_hosts) == set(eval_hosts)
    order = {h: i for i, h in enumerate(ours_hosts)}

    m, t, keep = pool_matrices(shard, source)
    universe = pd.Index(sorted(ref[ref.window == "source"].partner.unique()))
    names = [shards.partners[i] for i in np.asarray(keep)]
    perm = universe.get_indexer(pd.Index(names))
    ok = perm >= 0
    assert ok.sum() == len(names), "все кандидаты source-окна должны быть в universe эталона"

    ref_m = _reference_matrix(ref, "source", order, universe)[:, perm[ok]]
    ref_t = _reference_matrix(ref, "target", order, universe)[:, perm[ok]]
    assert abs(m[:, ok] - ref_m).max() == 0
    assert abs(t[:, ok] - ref_t).max() == 0
    assert m.nnz == ref_m.nnz and t.nnz == ref_t.nnz
    assert len(keep) == m.shape[1] == t.shape[1]


def test_host_columns_point_at_self(shards):
    """Позиция «сам хост среди кандидатов» должна указывать на его собственный столбец."""
    window = "validation"
    shard, source = shards.shard(window), shards.shard("train")
    m, _, keep = pool_matrices(shard, source)
    host_col = host_columns(shards.partners, shards.hosts, keep, shard.pool())
    assert host_col.shape == (m.shape[0],)
    for row, col in enumerate(host_col):
        if col < 0:
            continue
        host = shards.hosts[shard.pool()[row]]
        assert shards.partners[int(keep[col])] == host
