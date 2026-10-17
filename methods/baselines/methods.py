"""Калибровочные бейзлайны по готовому features.csv: (ref, dim, seed) -> массив (n_hosts, d).

ref — таблица хосты x ref_* из reference/features.csv, порядок строк сохраняется.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD


def _cap(dim: int, limit: int, name: str) -> int:
    if dim > limit:
        warnings.warn(f"{name}: dim={dim} exceeds the maximum {limit}; using {limit}")
        return limit
    return dim


def random_embedding(ref: pd.DataFrame, dim: int, seed: int) -> np.ndarray:
    """Случайный шум — нижняя граница, которую обязан обогнать любой метод."""
    return np.random.default_rng(seed).standard_normal((len(ref), dim))


def pca_embedding(ref: pd.DataFrame, dim: int, seed: int) -> np.ndarray:
    """PCA эталонных признаков (с центрированием)."""
    dim = _cap(dim, min(ref.shape), "pca")
    return PCA(n_components=dim, random_state=seed).fit_transform(ref.to_numpy(float))


def svd_embedding(ref: pd.DataFrame, dim: int, seed: int) -> np.ndarray:
    """TruncatedSVD эталонных признаков (без центрирования)."""
    dim = _cap(dim, min(ref.shape[0] - 1, ref.shape[1]), "svd")
    return TruncatedSVD(n_components=dim, random_state=seed).fit_transform(ref.to_numpy(float))


def reference_identity(ref: pd.DataFrame, dim: int, seed: int) -> np.ndarray:
    """Эталонные признаки как есть — потолок, «утекает» намеренно (dim и seed не используются)."""
    return ref.to_numpy(float).copy()


METHODS = {
    "random": random_embedding,
    "pca": pca_embedding,
    "svd": svd_embedding,
    "reference_identity": reference_identity,
}
