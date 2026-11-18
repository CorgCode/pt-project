"""Exploratory-нетипичность: среднее сходство с k ближайшими, без порога."""
from __future__ import annotations

import numpy as np
import pytest

from hostquery.anomaly import outliers
from hostquery.errors import HostQueryError
from hostquery.store import Dataset


def make(vectors, ids=None) -> Dataset:
    x = np.asarray(vectors, dtype=float)
    return Dataset(ids=ids or [f"h{i}" for i in range(len(x))], vectors=x,
                   norms=np.linalg.norm(x, axis=1), kind="vectors",
                   stats={"hosts": len(x), "dimensions": x.shape[1]},
                   feature_names=[None] * x.shape[1], feature_kinds=["coord"] * x.shape[1])


def test_outlier_ranks_first_and_scores_are_explained():
    data = make([[1, 0], [0.99, 0.01], [0.98, 0.02], [0.97, 0.03], [0.96, 0.04], [0, 1]],
                ids=["a", "b", "c", "d", "e", "z"])
    result = outliers(data, 3)
    assert result["k"] == 3
    assert result["items"][0]["id"] == "z"
    assert result["items"][0]["score"] > 0.9
    assert result["items"][0]["mean_similarity"] < 0.05
    assert result["items"][1]["score"] < 0.1
    scores = [item["score"] for item in result["items"]]
    assert scores == sorted(scores, reverse=True)
    assert all(item["score"] == pytest.approx(1.0 - item["mean_similarity"]) for item in result["items"])


def test_identical_hosts_have_zero_novelty_and_stable_order():
    data = make([[1, 2]] * 5)
    result = outliers(data, 10)
    assert result["k"] == 4
    assert all(item["score"] == pytest.approx(0.0) for item in result["items"])
    assert [item["id"] for item in result["items"]] == ["h0", "h1", "h2", "h3", "h4"]


def test_labels_are_attached(demo):
    labels = [index % 3 for index in range(demo.n)]
    result = outliers(demo, 5, labels=labels)
    for index, item in enumerate(result["items"]):
        position = demo.ids.index(item["id"])
        assert item["group"] == labels[position]


def test_warning_says_it_is_not_an_attack_score():
    result = outliers(make([[1, 0], [0, 1], [1, 1]]), 1)
    assert any("не вероятность атаки" in warning for warning in result["warnings"])
    assert any("#21" in warning for warning in result["warnings"])


def test_k_validation():
    data = make([[1, 0], [0, 1], [1, 1]])
    with pytest.raises(HostQueryError, match="целым"):
        outliers(data, 0)
    with pytest.raises(HostQueryError, match="целым"):
        outliers(data, True)
    with pytest.raises(HostQueryError, match="три хоста"):
        outliers(make([[1, 0], [0, 1]]), 1)


def test_small_k_uses_only_nearest_neighbours():
    data = make([[1, 0], [0.9, 0.1], [0, 1]], ids=["a", "b", "z"])
    one = next(item for item in outliers(data, 1)["items"] if item["id"] == "z")
    two = next(item for item in outliers(data, 2)["items"] if item["id"] == "z")
    assert one["score"] < two["score"]
