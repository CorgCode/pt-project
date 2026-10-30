"""Два замороженных блока → подбор веса → PCA в размерность исходного лидера."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD


class LinearNeighborhood:
    """Линейный графовый блок прежнего опыта: [X | 0.5·A·X] → SVD-64."""

    def __init__(self, dim=64, seed=0):
        self.dim, self.seed = dim, seed

    @staticmethod
    def aggregate(features, counts):
        x = np.asarray(features)
        if counts.shape != (len(x), len(x)) or not np.isfinite(x).all():
            raise ValueError("Несовместимые размеры или нечисловые признаки")
        c = (counts + counts.T).tocsr().astype(np.float32)
        c.setdiag(0)
        c.eliminate_zeros()
        if (c.data < 0).any() or not np.isfinite(c.data).all():
            raise ValueError("Счётчики связей должны быть конечными и неотрицательными")
        c.data = np.log1p(c.data)
        # Повторяем нормировку ранее измеренного линейного baseline.
        c = sp.diags(1/np.maximum(np.asarray(c.sum(1)).ravel(), 1)) @ c
        return np.hstack([x, .5*(c @ x)])

    def fit(self, features, counts, cohort):
        joined = self.aggregate(features, counts)
        cohort = np.asarray(cohort, dtype=int)
        self.input_dim_ = np.asarray(features).shape[1]
        d = min(self.dim, joined.shape[1]-1, len(cohort)-1)
        if d < 1:
            raise ValueError("Недостаточно train-хостов или признаков для SVD")
        self.svd_ = TruncatedSVD(d, random_state=self.seed).fit(joined[cohort])
        return self

    def transform(self, features, counts):
        if not hasattr(self, "svd_"):
            raise RuntimeError("Сначала fit линейного блока на train")
        if np.asarray(features).shape[1] != self.input_dim_:
            raise ValueError("Изменилось число входных признаков графа")
        z = self.svd_.transform(self.aggregate(features, counts))
        # Для пустого профиля прежний baseline записывал один общий ненулевой вектор.
        z[np.linalg.norm(z, axis=1) == 0, 0] = 1
        return z


def _check(frame, name):
    if not isinstance(frame, pd.DataFrame) or frame.empty or not len(frame.columns):
        raise ValueError(f"{name}: нужна непустая таблица векторов")
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError(f"{name}: повторяются host_id или координаты")
    if any(not isinstance(h, str) or not h.strip() for h in frame.index):
        raise ValueError(f"{name}: host_id должны быть непустыми строками")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{name}: нечисловые или бесконечные координаты")


class LeaderGraphHybrid:
    """Равный средний масштаб блоков; вес 0 точно сохраняет исходные векторы."""

    def __init__(self, weight=0., seed=29):
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("weight должен быть конечным и неотрицательным")
        self.weight, self.seed = float(weight), seed

    def _graph(self, leader, graph):
        if graph is None:
            raise ValueError("Для положительного веса нужен графовый блок")
        _check(graph, "graph")
        missing = leader.index.difference(graph.index)
        if len(missing):
            raise ValueError(f"В graph нет {len(missing)} хостов лидера: {missing[:3].tolist()}")
        if set(graph.columns) != set(self.graph_columns_):
            raise ValueError("Изменились координаты графового блока")
        return graph.loc[leader.index, self.graph_columns_].to_numpy(float)

    def fit(self, leader, graph=None):
        _check(leader, "leader")
        self.leader_columns_ = leader.columns.tolist()
        self.dim_ = len(self.leader_columns_)
        self.pca_ = None
        if not self.weight:
            return self
        if len(leader) <= self.dim_:
            raise ValueError("Для PCA нужно больше train-хостов, чем выходных координат")
        if graph is None:
            raise ValueError("Для положительного веса нужен графовый блок")
        self.graph_columns_ = graph.columns.tolist()
        base = leader.to_numpy(float)
        extra = self._graph(leader, graph)
        self.leader_scale_ = float(np.sqrt(np.mean(np.sum(base**2, axis=1))))
        self.graph_mean_ = extra.mean(0)
        self.graph_scale_ = float(np.sqrt(np.mean(np.sum((extra-self.graph_mean_)**2, axis=1))))
        if min(self.leader_scale_, self.graph_scale_) < 1e-12:
            raise ValueError("У одного из блоков нет ненулевого масштаба")
        joined = self._join(base, extra)
        self.pca_ = PCA(n_components=self.dim_, svd_solver="full", random_state=self.seed).fit(joined)
        return self

    def _join(self, leader, graph):
        return np.hstack([leader/self.leader_scale_, self.weight*(graph-self.graph_mean_)/self.graph_scale_])

    def transform(self, leader, graph=None):
        if not hasattr(self, "leader_columns_"):
            raise RuntimeError("Сначала fit на train")
        _check(leader, "leader")
        if set(leader.columns) != set(self.leader_columns_):
            raise ValueError("Изменились координаты лидера")
        leader = leader.loc[:, self.leader_columns_]
        if not self.weight:
            return leader.copy()
        joined = self._join(leader.to_numpy(float), self._graph(leader, graph))
        z = self.pca_.transform(joined)
        return pd.DataFrame(z, index=leader.index, columns=self.leader_columns_)
