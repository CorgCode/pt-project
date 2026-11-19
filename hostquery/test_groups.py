"""Группы: детерминированный spherical k-means и 2D-проекция без sklearn."""
from __future__ import annotations

import numpy as np
import pytest

from hostquery import groups
from hostquery.errors import HostQueryError
from hostquery.store import Dataset


def make(vectors, ids=None) -> Dataset:
    x = np.asarray(vectors, dtype=float)
    return Dataset(ids=ids or [f"h{i}" for i in range(len(x))], vectors=x,
                   norms=np.linalg.norm(x, axis=1), kind="vectors",
                   stats={"hosts": len(x), "dimensions": x.shape[1]},
                   feature_names=[None] * x.shape[1], feature_kinds=["coord"] * x.shape[1])


def test_fit_is_deterministic_and_covers_every_host(demo):
    first = groups.fit(demo, 4)
    second = groups.fit(demo, 4)
    assert first["labels"] == second["labels"]
    assert first["centers"] == second["centers"]
    assert first["projection"] == second["projection"]
    assert first["k"] == 4 and sum(first["sizes"]) == demo.n
    assert len(first["labels"]) == demo.n
    assert set(first["labels"]) <= set(range(4))
    assert [item["hostId"] for item in first["projection"]] == demo.ids
    assert all(np.isfinite([item["x"], item["y"]]).all() for item in first["projection"])


def test_k_is_clamped_and_single_group_is_all_zero():
    data = make([[1, 0], [0, 1], [-1, 0]])
    result = groups.fit(data, 10)
    assert result["k"] == 3 and result["sizes"] == [1, 1, 1]
    single = groups.fit(data, 1)
    assert single["k"] == 1 and single["labels"] == [0, 0, 0] and single["sizes"] == [3]


def test_farthest_point_init_and_labels_match_core_js():
    data = make([[1, 0], [0, 1], [-1, 0]])
    result = groups.fit(data, 2)
    assert result["labels"] == [0, 0, 1]
    members = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    expected = np.mean(members, axis=0)
    expected /= np.linalg.norm(expected)
    assert np.allclose(result["centers"][0], expected)


def test_assign_returns_training_labels(demo):
    result = groups.fit(demo, 4)
    assigned = groups.assign_unit(demo.unit(), result["centers"])
    assert assigned.tolist() == result["labels"]


def test_projection_of_identical_vectors_is_zero():
    data = make([[2.0, 3.0]] * 5)
    projection = groups.project(data)
    assert all(item["x"] == 0.0 and item["y"] == 0.0 for item in projection)


def test_projection_is_stable_under_positive_scale(demo):
    scaled = Dataset(ids=demo.ids, vectors=demo.vectors * 7.5,
                     norms=demo.norms * 7.5, kind=demo.kind, stats=demo.stats,
                     feature_names=demo.feature_names, feature_kinds=demo.feature_kinds)
    first, second = groups.project(demo), groups.project(scaled)
    assert [item["hostId"] for item in first] == [item["hostId"] for item in second]
    for left, right in zip(first, second):
        assert right["x"] == pytest.approx(left["x"] * 7.5, abs=1e-9)
        assert right["y"] == pytest.approx(left["y"] * 7.5, abs=1e-9)


def test_profile_lists_members_and_centroid_extremes(demo):
    result = groups.fit(demo, 4)
    group = int(np.bincount(result["labels"]).argmax())
    profile = groups.profile(demo, result["labels"], group, top=5)
    expected = [host for host, label in zip(demo.ids, result["labels"]) if label == group]
    assert profile["members"] == expected and profile["size"] == len(expected)
    assert profile["group"] == group and "не роль" in profile["note"]
    assert len(profile["top_features"]) == 5
    values = [abs(item["centroid"]) for item in profile["top_features"]]
    assert values == sorted(values, reverse=True)
    assert np.linalg.norm(profile["centroid"]) == pytest.approx(1.0)
    assert -1.0 <= profile["mean_similarity"] <= 1.0
    for item in profile["top_features"]:
        assert item["name"] is None and item["kind"] == "embedding_coordinate"


def test_profile_rejects_bad_labels(demo):
    labels = [0] * demo.n
    with pytest.raises(HostQueryError, match="не совпадают"):
        groups.profile(demo, labels[:-1], 0)


def test_profile_describes_empty_group_without_failing():
    data = make([[1, 1], [1, 1], [1, 1]])
    result = groups.fit(data, 3)
    assert result["k"] == 3 and result["sizes"] == [3, 0, 0]
    profile = groups.profile(data, result["labels"], 1)
    assert profile["group"] == 1 and profile["size"] == 0 and profile["empty"] is True
    assert profile["members"] == [] and profile["top_features"] == []
    assert profile["centroid"] is None and profile["mean_similarity"] is None
    assert "пуста" in profile["note"] and "#21" in profile["note"]
    assert groups.profile(data, result["labels"], 0)["empty"] is False


def test_fit_unit_validates_input():
    with pytest.raises(HostQueryError, match="целым"):
        groups.fit_unit(np.ones((3, 2)), True)
    with pytest.raises(HostQueryError, match="непустая"):
        groups.fit_unit(np.empty((0, 2)))
    with pytest.raises(HostQueryError, match="матрица"):
        groups.fit_unit(np.ones(3))


def test_one_coordinate_cannot_be_projected():
    with pytest.raises(HostQueryError, match="2D-проекции"):
        groups.project(make([[1], [2]]))
