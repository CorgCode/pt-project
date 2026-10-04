"""Baseline embedding methods. Each takes (ref, dim, seed) and returns an (n_hosts, d) array.

`ref` is the hosts x ref_* DataFrame from reference/features.csv. Row order follows `ref`.
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
    """Gaussian noise: the floor every real method must beat."""
    return np.random.default_rng(seed).standard_normal((len(ref), dim))


def pca_embedding(ref: pd.DataFrame, dim: int, seed: int) -> np.ndarray:
    """PCA of the reference features (centered, linear, reduced)."""
    dim = _cap(dim, min(ref.shape), "pca")
    return PCA(n_components=dim, random_state=seed).fit_transform(ref.to_numpy(float))


def svd_embedding(ref: pd.DataFrame, dim: int, seed: int) -> np.ndarray:
    """TruncatedSVD of the reference features (no centering)."""
    dim = _cap(dim, min(ref.shape[0] - 1, ref.shape[1]), "svd")
    return TruncatedSVD(n_components=dim, random_state=seed).fit_transform(ref.to_numpy(float))


def reference_identity(ref: pd.DataFrame, dim: int, seed: int) -> np.ndarray:
    """Reference features unchanged: ceiling baseline (leaks by design; `dim`/`seed` unused)."""
    return ref.to_numpy(float).copy()


METHODS = {
    "random": random_embedding,
    "pca": pca_embedding,
    "svd": svd_embedding,
    "reference_identity": reference_identity,
}
