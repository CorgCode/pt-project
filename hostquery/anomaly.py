"""Exploratory-аномалии: насколько хост далёк от своих k ближайших соседей.

Это не детектор атак и не метрика бенчмарка: score показывает нетипичность
относительно выбранного метода и сохранённого набора. Порог и сравнение методов — #21.
"""
from __future__ import annotations

import numpy as np

from hostquery.errors import HostQueryError
from hostquery.store import Dataset


def outliers(dataset: Dataset, k: int = 10, *, labels: list[int] | None = None) -> dict:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise HostQueryError("k должно быть целым >= 1")
    if dataset.n < 3:
        raise HostQueryError("для оценки нетипичности нужно хотя бы три хоста")
    k = min(k, dataset.n - 1)
    unit = dataset.unit()
    scores = unit @ unit.T
    np.fill_diagonal(scores, -np.inf)
    nearest = -np.sort(-scores, axis=1)[:, :k]
    mean_similarity = nearest.mean(axis=1)
    items = []
    for index, host in enumerate(dataset.ids):
        item = {
            "id": host,
            "score": float(1.0 - mean_similarity[index]),
            "mean_similarity": float(mean_similarity[index]),
        }
        if labels is not None:
            item["group"] = int(labels[index])
        items.append(item)
    items.sort(key=lambda item: (-item["score"], item["id"]))
    return {
        "k": k,
        "items": items,
        "warnings": [
            "score — исследовательская нетипичность относительно выбранного метода, "
            "а не вероятность атаки; порог и сравнение методов — #21",
        ],
    }
