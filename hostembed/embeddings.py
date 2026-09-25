"""Построение векторных представлений хостов.

Реализованы три подхода (их сравнение — часть задачи проекта):
  * ``features`` — стандартизация поведенческих признаков + PCA;
  * ``graph``    — представление узла в графе коммуникаций (аппроксимация
                   DeepWalk через PPMI + усечённое SVD, метод NetMF);
  * ``combined`` — конкатенация стандартизованных ``features`` и ``graph``.

Признаковый эмбеддер разделён на fit/transform, чтобы получать представления
разных временны́х окон в ОДНОМ пространстве (нужно для оценки дрейфа хоста).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return x / norm


class FeatureEmbedder:
    """Стандартизация признаков + PCA. Обучается один раз, применяется к окнам."""

    def __init__(self, n_components: int = 16, seed: int = 42):
        self.n_components = n_components
        self.seed = seed
        self.scaler = StandardScaler()
        self.pca: PCA | None = None
        self.columns: list[str] | None = None

    def fit(self, feat_df: pd.DataFrame) -> "FeatureEmbedder":
        self.columns = list(feat_df.columns)
        xs = self.scaler.fit_transform(feat_df.values)
        k = int(min(self.n_components, xs.shape[1], max(1, xs.shape[0] - 1)))
        self.pca = PCA(n_components=k, random_state=self.seed)
        self.pca.fit(xs)
        return self

    def transform(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        assert self.pca is not None and self.columns is not None, "call fit() first"
        aligned = feat_df.reindex(columns=self.columns).fillna(0.0)
        xs = self.scaler.transform(aligned.values)
        emb = self.pca.transform(xs)
        cols = [f"f{i}" for i in range(emb.shape[1])]
        return pd.DataFrame(emb, index=feat_df.index, columns=cols)

    def fit_transform(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(feat_df).transform(feat_df)


def build_adjacency(df: pd.DataFrame, nodes: list[str]) -> np.ndarray:
    """Симметричная взвешенная матрица смежности (вес = число потоков)."""
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    a = np.zeros((n, n), dtype=np.float64)
    src = df["src_device"].map(idx)
    dst = df["dst_device"].map(idx)
    mask = src.notna() & dst.notna()
    for i, j in zip(src[mask].astype(int), dst[mask].astype(int)):
        if i == j:
            continue
        a[i, j] += 1.0
        a[j, i] += 1.0
    return a


def graph_embedding(
    df: pd.DataFrame,
    nodes: list[str],
    dim: int = 16,
    walk_len: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """DeepWalk-представление узлов через факторизацию PPMI-матрицы (NetMF).

    M = vol/T * (sum_{r=1..T} P^r) D^-1,  PPMI = max(log M, 0),  emb = SVD(PPMI).
    """
    a = build_adjacency(df, nodes)
    n = a.shape[0]
    deg = a.sum(axis=1)
    nonzero = deg > 0
    d_inv = np.zeros(n)
    d_inv[nonzero] = 1.0 / deg[nonzero]
    p = a * d_inv[:, None]                      # строковая нормировка = D^-1 A
    vol = a.sum()
    if vol == 0:
        return pd.DataFrame(np.zeros((n, dim)), index=nodes,
                            columns=[f"g{i}" for i in range(dim)])

    s = np.zeros_like(p)
    pk = np.eye(n)
    for _ in range(walk_len):
        pk = pk @ p
        s += pk
    m = (vol / walk_len) * (s * d_inv[None, :])
    with np.errstate(divide="ignore"):
        ppmi = np.log(np.where(m > 0, m, 1.0))
    ppmi = np.maximum(ppmi, 0.0)

    k = int(min(dim, max(1, min(ppmi.shape) - 1)))
    svd = TruncatedSVD(n_components=k, random_state=seed)
    u = svd.fit_transform(ppmi)                 # = U * S
    sv = np.sqrt(np.clip(svd.singular_values_, 0, None))
    emb = u / np.where(sv == 0, 1.0, sv)        # ~ U * sqrt(S)
    emb = l2_normalize(emb)
    cols = [f"g{i}" for i in range(emb.shape[1])]
    return pd.DataFrame(emb, index=nodes, columns=cols)


def combined_embedding(feat_emb: pd.DataFrame, graph_emb: pd.DataFrame) -> pd.DataFrame:
    """Стандартизовать оба представления и склеить (выровнено по хостам)."""
    idx = feat_emb.index
    g = graph_emb.reindex(idx).fillna(0.0)
    fz = StandardScaler().fit_transform(feat_emb.values)
    gz = StandardScaler().fit_transform(g.values)
    mat = np.hstack([fz, gz])
    cols = [f"c{i}" for i in range(mat.shape[1])]
    return pd.DataFrame(mat, index=idx, columns=cols)
