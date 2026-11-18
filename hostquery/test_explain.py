"""Объяснение пары: вклад координат и проверяемый контекст контактов."""
from __future__ import annotations

import pytest

from hostquery.errors import HostQueryError
from hostquery.explain import _contacts, pair
from hostquery.store import dataset_from_csv

FLOWS = """src,dst,bytes,packets,proto,dst_port
h1,h2,10,1,6,443
h1,h3,20,2,17,53
h1,h4,30,3,6,80
h2,h1,40,4,6,443
h2,h4,50,5,6,80
h3,h1,60,6,6,443
"""


def cosine(data, first, second):
    i, j = data.position(first), data.position(second)
    return float(data.unit()[i] @ data.unit()[j])


def test_pair_on_embedding_has_no_invented_names(demo):
    first, second = demo.ids[0], demo.ids[1]
    result = pair(demo, first, second)
    assert result["cosine"] == pytest.approx(cosine(demo, first, second))
    assert result["a"] == first and result["b"] == second
    assert len(result["coordinates"]) == 8
    assert {item["kind"] for item in result["coordinates"]} == {"embedding_coordinate"}
    assert {item["name"] for item in result["coordinates"]} == {None}
    assert result["shared_contacts"] is None
    assert any("нет имён" in warning for warning in result["warnings"])
    assert any("не вероятность" in warning for warning in result["warnings"])


def test_pair_coordinates_and_contributions_are_consistent(demo):
    first, second = demo.ids[2], demo.ids[3]
    result = pair(demo, first, second, top=10)
    assert len(result["coordinates"]) == 10
    for item in result["coordinates"]:
        assert item["product"] == pytest.approx(item["a"] * item["b"])
        assert item["delta"] == pytest.approx(abs(item["a"] - item["b"]))
    contributions = [abs(item["contribution"]) for item in result["top_contributions"]]
    assert len(result["top_contributions"]) == 3
    assert contributions == sorted(contributions, reverse=True)


def test_pair_on_flows_shows_activity_and_shared_contacts():
    data = dataset_from_csv(FLOWS)
    result = pair(data, "h1", "h2")
    assert [item["name"] for item in result["coordinates"]] == [
        "Исходящие записи", "Входящие записи", "Байты отправлено", "Байты получено",
        "Пакеты отправлено", "Пакеты получено", "Разных контактов", "Всего записей",
    ]
    assert {"h4"} == set(result["shared_contacts"])
    assert "проверяемый контекст" in result["shared_contacts_note"]
    assert any("хешированные" in warning for warning in result["warnings"])
    assert result["cosine"] == pytest.approx(cosine(data, "h1", "h2"))


def test_contacts_exclude_self_and_are_unique():
    data = dataset_from_csv(FLOWS)
    contacts = _contacts(data)
    assert contacts["h1"] == ["h2", "h3", "h4"]
    assert contacts["h4"] == ["h1", "h2"]
    assert all(host not in contacts[host] for host in contacts)


def test_pair_is_symmetric_and_rejects_same_or_unknown_hosts(demo):
    first, second = demo.ids[4], demo.ids[5]
    forward, backward = pair(demo, first, second), pair(demo, second, first)
    assert forward["cosine"] == pytest.approx(backward["cosine"])
    with pytest.raises(HostQueryError, match="два разных"):
        pair(demo, first, first)
    with pytest.raises(HostQueryError, match="неизвестный хост"):
        pair(demo, first, "НЕТ-ТАКОГО")
