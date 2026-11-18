"""Похожие хосты: косинус, границы k, стабильные ничьи, консенсус."""
from __future__ import annotations

import numpy as np
import pytest

from hostquery.errors import HostQueryError
from hostquery.similar import check_k, consensus, neighbors, similar_to
from hostquery.store import Dataset


def tiny(vectors, ids=None) -> Dataset:
    x = np.asarray(vectors, dtype=float)
    return Dataset(ids=ids or [f"h{i}" for i in range(len(x))], vectors=x,
                   norms=np.linalg.norm(x, axis=1), kind="vectors",
                   stats={"hosts": len(x), "dimensions": x.shape[1]},
                   feature_names=[None] * x.shape[1], feature_kinds=["coord"] * x.shape[1])


def cosine(data: Dataset, first: str, second: str) -> float:
    i, j = data.position(first), data.position(second)
    return float(data.unit()[i] @ data.unit()[j])


def test_neighbors_exclude_self_are_sorted_and_clipped(demo):
    host = demo.ids[3]
    result = neighbors(demo, host, 10)
    ids = [item["id"] for item in result["neighbors"]]
    assert len(ids) == 10 and host not in ids and len(set(ids)) == 10
    assert [item["rank"] for item in result["neighbors"]] == list(range(1, 11))
    values = [item["similarity"] for item in result["neighbors"]]
    assert values == sorted(values, reverse=True)
    assert all(-1.0 <= value <= 1.0 for value in values)
    assert result["k"] == 10 and result["host"] == host
    for item in result["neighbors"]:
        assert item["similarity"] == pytest.approx(cosine(demo, host, item["id"]), abs=1e-12)


def test_ties_keep_dataset_order(demo):
    data = tiny([[1, 0]] * 4, ids=["a", "b", "c", "d"])
    result = neighbors(data, "a", 3)
    assert [item["id"] for item in result["neighbors"]] == ["b", "c", "d"]
    assert [item["similarity"] for item in result["neighbors"]] == [1.0, 1.0, 1.0]


def test_labels_are_zero_based_and_attached(demo):
    labels = [index % 4 for index in range(demo.n)]
    result = neighbors(demo, demo.ids[0], 5, labels=labels)
    position = {host: index for index, host in enumerate(demo.ids)}
    for item in result["neighbors"]:
        assert item["group"] == labels[position[item["id"]]]


def test_reference_ranks_are_marked(demo, demo_path):
    from hostquery.store import RunStore

    store = RunStore(bundle=demo_path)
    host = demo.ids[0]
    reference = store.reference_ranks(host)
    result = neighbors(demo, host, 5, reference=reference)
    marked = [item for item in result["neighbors"] if "reference_rank" in item]
    assert marked
    for item in marked:
        assert item["in_reference_top_k"] is (item["reference_rank"] <= 5)


def test_k_validation(demo):
    assert check_k(demo, 1) == 1 and check_k(demo, demo.n - 1) == demo.n - 1
    for bad in (0, demo.n, -1):
        with pytest.raises(HostQueryError, match="k должно быть"):
            check_k(demo, bad)
    with pytest.raises(HostQueryError, match="целым"):
        check_k(demo, True)
    single = Dataset(ids=["a"], vectors=np.ones((1, 1)), norms=np.ones(1),
                     kind="vectors", stats={}, feature_names=[None], feature_kinds=["x"])
    with pytest.raises(HostQueryError, match="хотя бы два хоста"):
        check_k(single, 1)


def test_unknown_host_has_hint(demo):
    with pytest.raises(HostQueryError) as error:
        neighbors(demo, "no-such-host", 5)
    assert "неизвестный хост" in error.value.message
    assert "hosts" in (error.value.hint or "")


def test_similar_to_vector_is_id_free(demo):
    """Поиск по вектору не знает ID: если передать вектор хоста, он сам будет первым."""
    host = demo.ids[7]
    vector = demo.vectors[demo.position(host)]
    by_vector = similar_to(demo, vector, 10)
    by_host = neighbors(demo, host, 10)
    assert by_vector["neighbors"][0]["id"] == host
    assert by_vector["neighbors"][0]["similarity"] == pytest.approx(1.0)
    assert [item["id"] for item in by_vector["neighbors"][1:]] == \
        [item["id"] for item in by_host["neighbors"][:9]]
    for left, right in zip(by_vector["neighbors"][1:], by_host["neighbors"][:9]):
        assert left["similarity"] == pytest.approx(right["similarity"], abs=1e-12)


def test_similar_to_rejects_bad_vectors(demo):
    with pytest.raises(HostQueryError, match="координат"):
        similar_to(demo, [1.0, 2.0], 3)
    with pytest.raises(HostQueryError, match="NaN"):
        similar_to(demo, [np.nan] * demo.dim, 3)
    with pytest.raises(HostQueryError, match="нулевой"):
        similar_to(demo, [0.0] * demo.dim, 3)


def test_consensus_averages_ranks_and_counts_votes(demo):
    result = consensus([("first", demo), ("second", demo)], demo.ids[0], 5)
    assert result["n_methods"] == 2 and result["warnings"] == []
    assert len(result["neighbors"]) == 5
    for item in result["neighbors"]:
        assert item["votes"] == 2 and item["rank_spread"] == 0
        assert item["methods"] == ["first", "second"]


def test_consensus_skips_method_without_host_and_warns(demo):
    host = demo.ids[0]
    keep = [index for index, value in enumerate(demo.ids) if value != host]
    subset = Dataset(ids=[demo.ids[i] for i in keep], vectors=demo.vectors[keep],
                     norms=demo.norms[keep], kind="vectors", stats=demo.stats,
                     feature_names=demo.feature_names, feature_kinds=demo.feature_kinds)
    result = consensus([("full", demo), ("subset", subset)], host, 3)
    assert any("пропущен" in warning for warning in result["warnings"])
    assert result["n_methods"] == 2
    with pytest.raises(HostQueryError, match="ни в одном"):
        consensus([("subset", subset)], host, 3)
    with pytest.raises(HostQueryError, match="хотя бы один метод"):
        consensus([], host, 3)
