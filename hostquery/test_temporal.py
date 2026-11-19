"""Изменения между окнами: общая шкала, группы первого окна, контакты."""
from __future__ import annotations

import numpy as np
import pytest

from hostquery.errors import HostQueryError
from hostquery.store import Dataset, dataset_from_csv
from hostquery.temporal import _prepared, compare

BEFORE = """src,dst,bytes,packets,proto,dst_port
a,b,100,1,6,443
a,c,100,1,6,443
b,a,100,1,6,443
b,c,100,1,6,443
c,a,100,1,17,53
c,b,100,1,17,53
"""

AFTER = """src,dst,bytes,packets,proto,dst_port
a,b,120,1,6,443
a,c,120,1,6,443
a,d,500,5,6,80
b,a,120,1,6,443
b,c,120,1,6,443
c,a,120,1,17,53
c,b,120,1,17,53
e,b,50,1,6,22
"""

AFTER_WITHOUT_C = """src,dst,bytes,packets,proto,dst_port
a,b,120,1,6,443
a,d,500,5,6,80
b,a,120,1,6,443
"""


def flows(text: str) -> Dataset:
    return dataset_from_csv(text)


def vectors(ids, values) -> Dataset:
    x = np.asarray(values, dtype=float)
    return Dataset(ids=list(ids), vectors=x, norms=np.linalg.norm(x, axis=1),
                   kind="vectors", stats={"hosts": len(ids), "dimensions": x.shape[1]},
                   feature_names=[None] * x.shape[1], feature_kinds=["coord"] * x.shape[1])


def test_identical_windows_have_similarity_one_and_same_groups():
    data = flows(BEFORE)
    result = compare(data, data, k=2)
    assert result["kind"] == "flows"
    assert result["newHosts"] == [] and result["missingHosts"] == []
    assert result["n_common"] == 3
    for item in result["changes"]:
        assert item["similarity"] == pytest.approx(1.0)
        assert item["beforeGroup"] == item["afterGroup"]
        assert item["added"] == [] and item["removed"] == []
        assert item["before"]["id"] == item["id"] == item["after"]["id"]


def test_changes_are_sorted_most_changed_first():
    result = compare(flows(BEFORE), flows(AFTER), k=2)
    keys = [(item["similarity"], item["id"]) for item in result["changes"]]
    assert keys == sorted(keys)
    assert result["changes"][0]["similarity"] == min(item["similarity"] for item in result["changes"])


def test_new_contacts_and_new_hosts_are_reported():
    result = compare(flows(BEFORE), flows(AFTER), k=2)
    by_id = {item["id"]: item for item in result["changes"]}
    assert "d" in by_id["a"]["added"]
    assert "d" not in by_id["a"]["removed"]
    assert "d" in by_id["a"]["afterContacts"]
    assert "d" not in by_id["a"]["beforeContacts"]
    assert set(result["newHosts"]) == {"d", "e"}
    assert result["missingHosts"] == []
    assert by_id["a"]["before"]["activity"]["outgoing"] == 2
    assert by_id["a"]["after"]["activity"]["outgoing"] == 3


def test_missing_hosts_are_reported_separately():
    result = compare(flows(BEFORE), flows(AFTER_WITHOUT_C), k=2)
    assert result["missingHosts"] == ["c"]
    assert set(result["newHosts"]) == {"d"}


def test_vector_windows_carry_no_contacts_and_warn():
    before = vectors(["a", "b", "c"], [[1, 0], [0, 1], [1, 1]])
    after = vectors(["a", "b", "c"], [[0.9, 0.1], [0.1, 0.9], [1, 0.8]])
    result = compare(before, after, k=2)
    assert result["kind"] == "vectors"
    assert any("записей обмена" in warning for warning in result["warnings"])
    for item in result["changes"]:
        assert "added" not in item and "removed" not in item
        assert set(item["before"]) == {"id", "v", "norm"}


def test_groups_are_learned_on_first_window_even_with_empty_clusters():
    identical = flows("src,dst\nx,y\ny,z\nz,x\n")
    result = compare(identical, identical, k=3)
    assert result["k"] == 3
    for item in result["changes"]:
        assert item["beforeGroup"] == 0
        assert item["afterGroup"] == 0


def test_shared_scale_is_computed_on_both_windows():
    before = flows(BEFORE)
    after = flows(AFTER)
    left, left_norms, right, right_norms = _prepared(before, after)
    joined = np.maximum(before.log_values.max(axis=0), after.log_values.max(axis=0))
    scales = np.maximum(1.0, joined)
    assert np.allclose(left, before.log_values / scales)
    assert np.allclose(right, after.log_values / scales)
    assert np.allclose(left_norms, np.linalg.norm(left, axis=1))
    assert np.allclose(right_norms, np.linalg.norm(right, axis=1))
    assert np.all(left >= 0) and np.all(right >= 0)


def test_procrustes_undoes_rotation():
    before = vectors(["a", "b", "c"], [[1, 0], [0, 1], [1, 1]])
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    values = before.vectors @ rotation
    after = vectors(["a", "b", "c"], values)
    plain = compare(before, after, k=2)
    aligned = compare(before, after, k=2, alignment="procrustes")
    assert max(abs(item["similarity"]) for item in plain["changes"]) < 0.01
    assert min(item["similarity"] for item in aligned["changes"]) > 0.999
    assert any("Procrustes" in warning for warning in aligned["warnings"])


def test_mismatched_windows_are_rejected():
    with pytest.raises(HostQueryError, match="разного формата"):
        compare(flows(BEFORE), vectors(["a", "b"], [[1, 0], [0, 1]]))
    with pytest.raises(HostQueryError, match="размерности"):
        compare(vectors(["a", "b"], [[1, 0], [0, 1]]),
                vectors(["a", "b"], [[1, 0, 0], [0, 1, 0]]))
    with pytest.raises(HostQueryError, match="общих хостов"):
        compare(vectors(["a", "b"], [[1, 0], [0, 1]]),
                vectors(["c", "d"], [[1, 0], [0, 1]]))
    with pytest.raises(HostQueryError, match="alignment"):
        compare(vectors(["a", "b"], [[1, 0], [0, 1]]),
                vectors(["a", "b"], [[1, 0], [0, 1]]), alignment="spin")
