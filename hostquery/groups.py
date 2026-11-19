"""Группы хостов: детерминированный spherical k-means и 2D-проекция.

Алгоритм портирован из `noomo-ui/lab/core.js` один в один (farthest-point init,
20 итераций, стабильный выбор при равенстве). Поэтому числа совпадают с лабораторией,
а не «примерно похожи»: sklearn KMeans и PCA дали бы другой поворот и другие метки.

Группа — это номер кластера, а не роль хоста. Качество разбиения оценивает #21.
"""
from __future__ import annotations

import math

import numpy as np

from hostquery.errors import HostQueryError
from hostquery.store import Dataset

ITERATIONS = 20
PROJECTION_ITERATIONS = 18


def _axis(matrix: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
    dimensions = matrix.shape[1]
    vector = np.array([math.sin(j * 1.7 + (1.1 if previous is not None else 0.4))
                       for j in range(dimensions)], dtype=float)
    for _ in range(PROJECTION_ITERATIONS):
        nxt = matrix.T @ (matrix @ vector)
        if previous is not None:
            nxt = nxt - float(nxt @ previous) * previous
        norm = float(np.linalg.norm(nxt))
        if norm < 1e-12:
            return np.zeros(dimensions)
        vector = nxt / norm
    return vector


def project(dataset: Dataset) -> list[dict]:
    """Две оси степени двойки: степенной метод по центрированным векторам."""
    if dataset.dim < 2:
        raise HostQueryError("у вектора одна координата — 2D-проекции нет")
    matrix = dataset.vectors - dataset.vectors.mean(axis=0)
    first = _axis(matrix, None)
    second = _axis(matrix, first)
    return [{"x": float(row @ first), "y": float(row @ second), "hostId": host}
            for row, host in zip(matrix, dataset.ids)]


def _unit(dataset: Dataset) -> np.ndarray:
    return dataset.unit()


def fit_unit(rows: np.ndarray, k: int = 3) -> dict:
    """Ядро алгоритма: уже L2-нормированные строки, метки 0-based и центроиды."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise HostQueryError("число групп должно быть целым >= 1")
    rows = np.asarray(rows, dtype=float)
    if rows.ndim != 2 or rows.shape[0] == 0:
        raise HostQueryError("для групп нужна непустая матрица host × признак")
    k = min(k, len(rows))
    centers = [rows[0].copy()]
    while len(centers) < k:
        best, chosen = -math.inf, 0
        stacked = np.vstack(centers)
        for index, vector in enumerate(rows):
            distance = 1.0 - float(np.max(stacked @ vector))
            if distance > best:
                best, chosen = distance, index
        centers.append(rows[chosen].copy())
    matrix = np.vstack(centers)
    labels = np.zeros(len(rows), dtype=int)
    for iteration in range(ITERATIONS):
        nxt = np.argmax(rows @ matrix.T, axis=1)
        changed = not np.array_equal(nxt, labels)
        labels = nxt
        for group in range(k):
            members = rows[labels == group]
            if not members.size:
                continue
            total = members.sum(axis=0)
            norm = float(np.linalg.norm(total))
            if norm > 1e-12:
                matrix[group] = total / norm
        if not changed and iteration > 0:
            break
    sizes = [int((labels == group).sum()) for group in range(k)]
    return {
        "k": k,
        "labels": [int(label) for label in labels],
        "sizes": sizes,
        "centers": [[float(x) for x in center] for center in matrix],
    }


def fit(dataset: Dataset, k: int = 3) -> dict:
    """Метки групп 0-based, размеры, центроиды и 2D-проекция. Детерминирован."""
    result = fit_unit(dataset.unit(), k)
    result["projection"] = project(dataset)
    return result


def assign_unit(rows: np.ndarray, centers) -> np.ndarray:
    """Разложить готовые векторы по центрам — так сравниваются два окна."""
    return np.argmax(np.asarray(rows, dtype=float) @ np.asarray(centers, dtype=float).T, axis=1)


def assign(dataset: Dataset, centers) -> np.ndarray:
    return assign_unit(dataset.unit(), centers)


def profile(dataset: Dataset, labels: list[int], group: int, *, top: int = 5) -> dict:
    """Кто в группе и на каких координатах её центроид сильнее всего отличается от нуля."""
    if len(labels) != dataset.n:
        raise HostQueryError("метки групп не совпадают по длине с набором")
    members = [host for host, label in zip(dataset.ids, labels) if label == group]
    if not members:
        # Совпадающие профили — допустимый вход: k-means может оставить кластер
        # без хостов. Не роняем весь ответ из-за одной пустой группы, а явно
        # описываем её, сохраняя исходную нумерацию меток.
        return {
            "group": group,
            "size": 0,
            "members": [],
            "mean_similarity": None,
            "centroid": None,
            "top_features": [],
            "empty": True,
            "note": "группа пуста: в наборе нет хостов с этой меткой; "
                    "качество разбиения оценивает #21",
        }
    unit = dataset.unit()
    position = {host: index for index, host in enumerate(dataset.ids)}
    rows = unit[[position[host] for host in members]]
    centroid = rows.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm > 0:
        centroid = centroid / norm
    order = np.argsort(-np.abs(centroid), kind="stable")[:top]
    top_features = []
    for index in order:
        index = int(index)
        top_features.append({
            "index": index,
            "name": dataset.feature_names[index],
            "kind": dataset.feature_kinds[index],
            "centroid": float(centroid[index]),
        })
    return {
        "group": group,
        "size": len(members),
        "members": members,
        "mean_similarity": float(np.mean(rows @ centroid)),
        "centroid": [float(x) for x in centroid],
        "top_features": top_features,
        "empty": False,
        "note": "группа — номер кластера, а не роль хоста",
    }
