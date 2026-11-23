"""Объяснение похожести пары: вклад координат и проверяемый контекст.

Имена координат эмбеддинга неизвестны, поэтому для них возвращается
`embedding_coordinate`, а не придуманное «похожи по DNS». Для трафика имена есть,
но двенадцать сервисных блоков — хеши, и это тоже сказано в ответе.
"""
from __future__ import annotations

import numpy as np

from hostquery.errors import HostQueryError
from hostquery.store import Dataset

FLOW_VISIBLE = 8


def pair(dataset: Dataset, first: str, second: str, *, top: int = 8) -> dict:
    if first == second:
        raise HostQueryError("нужны два разных хоста")
    i, j = dataset.position(first), dataset.position(second)
    unit = dataset.unit()
    a, b = unit[i], unit[j]
    cosine = float(np.clip(a @ b, -1.0, 1.0))

    if dataset.kind == "flows":
        indices = list(range(min(FLOW_VISIBLE, dataset.dim)))
    else:
        indices = [int(index) for index in np.argsort(-(np.abs(a) + np.abs(b)), kind="stable")[:top]]

    coordinates = [{
        "index": int(index),
        "name": dataset.feature_names[index],
        "kind": dataset.feature_kinds[index],
        "a": float(a[index]),
        "b": float(b[index]),
        "delta": float(abs(a[index] - b[index])),
        "product": float(a[index] * b[index]),
    } for index in indices]

    out = {
        "a": first,
        "b": second,
        "cosine": cosine,
        "coordinates": coordinates,
        "top_contributions": sorted(
            ({"index": item["index"], "name": item["name"], "contribution": item["product"]}
             for item in coordinates),
            key=lambda item: -abs(item["contribution"]))[:3],
    }

    warnings = ["косинус — близость направлений профилей, не вероятность и не доказательство роли"]
    if dataset.kind == "flows":
        shared = _shared_contacts(dataset, first, second)
        out["shared_contacts"] = shared
        out["shared_contacts_note"] = ("общий собеседник — проверяемый контекст, "
                                       "а не объяснение всей близости")
        if dataset.traffic is not None:
            out["traffic_comparison"] = {
                direction: _compare_traffic(dataset.traffic[first][direction],
                                            dataset.traffic[second][direction])
                for direction in ("outgoing", "incoming")
            }
        warnings.append("12 сервисных признаков — хешированные блоки, не протоколы")
    else:
        out["shared_contacts"] = None
        warnings.append("у координат эмбеддинга нет имён: сравнение идёт по номерам осей")
    out["warnings"] = warnings
    return out


def _compare_traffic(first: dict, second: dict) -> dict:
    """Compare raw observations for the same direction on both hosts."""
    result = {key: {"a": first[key], "b": second[key]}
              for key in ("flows", "bytes", "packets") if key in first and key in second}
    for key in ("ports", "protocols"):
        if first.get(key) and second.get(key):
            result[f"shared_{key}"] = sorted(first[key] & second[key])
    return result


def _contacts(dataset: Dataset) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {host: [] for host in dataset.ids}
    for edge in dataset.connections or []:
        src, dst = str(edge["src"]), str(edge["dst"])
        if src == dst:
            continue
        if dst not in found.setdefault(src, []):
            found[src].append(dst)
        if src not in found.setdefault(dst, []):
            found[dst].append(src)
    return found


def _shared_contacts(dataset: Dataset, first: str, second: str) -> list[str]:
    contacts = _contacts(dataset)
    other = set(contacts.get(second, []))
    return [host for host in contacts.get(first, []) if host in other]
