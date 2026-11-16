"""Похожие хосты: косинус, top-k, консенсус по методам.

Правила те же, что у `Bundle.neighborhood` и `PTLab`: self исключён, ничьи решаются
порядком хостов в наборе (stable sort), косинус не превращается в вероятность.
"""
from __future__ import annotations

import numpy as np

from hostquery.errors import HostQueryError
from hostquery.store import Dataset


def check_k(dataset: Dataset, k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, int):
        raise HostQueryError("k должно быть целым числом")
    if dataset.n < 2:
        raise HostQueryError("для поиска соседей нужно хотя бы два хоста")
    if not 1 <= k <= dataset.n - 1:
        raise HostQueryError(f"k должно быть от 1 до {dataset.n - 1}, получено {k}")
    return k


def _ranked(dataset: Dataset, scores: np.ndarray, k: int, *, skip: int | None = None,
            labels: list[int] | None = None, reference: dict[str, int] | None = None) -> list[dict]:
    order = [int(i) for i in np.argsort(-scores, kind="stable") if i != skip][:k]
    items = []
    for rank, index in enumerate(order, start=1):
        item = {
            "id": dataset.ids[index],
            "rank": rank,
            "similarity": float(np.clip(scores[index], -1.0, 1.0)),
        }
        if labels is not None:
            item["group"] = int(labels[index])
        if reference:
            ref_rank = reference.get(dataset.ids[index])
            if ref_rank is not None:
                item["reference_rank"] = ref_rank
                item["in_reference_top_k"] = ref_rank <= k
        items.append(item)
    return items


def neighbors(dataset: Dataset, host: str, k: int = 10, *, labels: list[int] | None = None,
              reference: dict[str, int] | None = None) -> dict:
    """top-k по косинусу; `reference` — ранги эталона для пометки совпадений."""
    check_k(dataset, k)
    position = dataset.position(host)
    unit = dataset.unit()
    scores = unit @ unit[position]
    return {"host": host, "k": k, "neighbors": _ranked(dataset, scores, k, skip=position,
                                                       labels=labels, reference=reference)}


def similar_to(dataset: Dataset, vector, k: int = 10, *, labels: list[int] | None = None) -> dict:
    """Поиск по произвольному вектору той же размерности: нового хоста в наборе нет."""
    check_k(dataset, k)
    query = np.asarray(vector, dtype=float).reshape(-1)
    if query.size != dataset.dim:
        raise HostQueryError(f"у вектора {query.size} координат, а в наборе {dataset.dim}")
    if not np.isfinite(query).all():
        raise HostQueryError("в векторе есть NaN или бесконечность")
    norm = float(np.linalg.norm(query))
    if norm <= 0:
        raise HostQueryError("нулевой вектор: косинус не определён")
    scores = dataset.unit() @ (query / norm)
    return {"k": k, "neighbors": _ranked(dataset, scores, k, labels=labels)}


def consensus(items: list[tuple[str, Dataset]], host: str, k: int = 10) -> dict:
    """Средний ранг по нескольким методам; разброс рангов показывается отдельно."""
    if not items:
        raise HostQueryError("для консенсуса нужен хотя бы один метод")
    ranks: dict[str, list[int]] = {}
    sims: dict[str, list[float]] = {}
    methods: dict[str, list[str]] = {}
    warnings: list[str] = []
    for name, dataset in items:
        if host not in dataset.ids:
            warnings.append(f"метод {name}: хоста {host} нет в наборе — пропущен")
            continue
        for item in neighbors(dataset, host, k)["neighbors"]:
            ranks.setdefault(item["id"], []).append(item["rank"])
            sims.setdefault(item["id"], []).append(item["similarity"])
            methods.setdefault(item["id"], []).append(name)
    if not ranks:
        raise HostQueryError(f"хоста {host!r} нет ни в одном из выбранных методов")
    rows = []
    for candidate, values in ranks.items():
        rows.append({
            "id": candidate,
            "mean_rank": float(np.mean(values)),
            "votes": len(values),
            "rank_spread": int(max(values) - min(values)),
            "mean_similarity": float(np.mean(sims[candidate])),
            "methods": sorted(methods[candidate]),
        })
    rows.sort(key=lambda row: (row["mean_rank"], -row["votes"], row["id"]))
    return {"host": host, "k": k, "n_methods": len(items),
            "neighbors": rows[:k], "warnings": warnings}
