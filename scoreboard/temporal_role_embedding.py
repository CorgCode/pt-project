
"""Temporal Role Embedding v1.

Самостоятельная CPU-модель сетевых эмбеддингов.
Не использует HybridTfidf и тестовые метки.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, normalize


class TemporalRoleEmbedding:
    name = "temporal_role_embedding"
    shared_coordinate_system = True

    def __init__(
        self,
        role_dim=64,
        behavior_weight=2.0,
        role_weight=1.0,
        seed=42,
    ):
        self.role_dim = role_dim
        self.behavior_weight = behavior_weight
        self.role_weight = role_weight
        self.seed = seed

    def _behavior(self, flows, hosts):
        hosts = pd.Index(hosts)

        src = flows["src"].astype(str)
        dst = flows["dst"].astype(str)

        edges = pd.DataFrame({
            "src": src.to_numpy(),
            "dst": dst.to_numpy(),
        }).drop_duplicates()

        service = (
            flows["proto"].astype(str)
            + "/"
            + flows["dst_port"].astype(str)
        )

        def counts(series):
            return (
                series.value_counts()
                .reindex(hosts, fill_value=0)
                .to_numpy(dtype=float)
            )

        def unique_count(group, column):
            return (
                group[column].nunique()
                .reindex(hosts, fill_value=0)
                .to_numpy(dtype=float)
            )

        outgoing = counts(src)
        incoming = counts(dst)

        out_degree = unique_count(
            edges.groupby("src"), "dst"
        )
        in_degree = unique_count(
            edges.groupby("dst"), "src"
        )

        service_table = pd.DataFrame({
            "src": src.to_numpy(),
            "dst": dst.to_numpy(),
            "service": service.to_numpy(),
        })

        out_services = unique_count(
            service_table.groupby("src"), "service"
        )
        in_services = unique_count(
            service_table.groupby("dst"), "service"
        )

        total = outgoing + incoming

        features = np.column_stack([
            outgoing,
            incoming,
            out_degree,
            in_degree,
            out_services,
            in_services,
            total,
            out_degree + in_degree,
            outgoing / np.maximum(total, 1),
            incoming / np.maximum(total, 1),
        ])

        # Сжимаем большие значения активности.
        features[:, :8] = np.log1p(features[:, :8])

        return features.astype(np.float32)

    def _graph(self, flows):
        """Направленная матрица: исходящие + входящие связи."""
        src = flows["src"].astype(str).to_numpy()
        dst = flows["dst"].astype(str).to_numpy()

        src_idx = self.host_index.get_indexer(src)
        dst_idx = self.host_index.get_indexer(dst)

        valid = (src_idx >= 0) & (dst_idx >= 0)

        src_idx = src_idx[valid]
        dst_idx = dst_idx[valid]

        n = len(self.hosts)

        # Одна строка на хост.
        # Первые n столбцов — исходящие связи.
        # Следующие n — входящие.
        rows = np.concatenate([src_idx, dst_idx])
        cols = np.concatenate([
            dst_idx,
            src_idx + n,
        ])

        matrix = sp.coo_matrix(
            (
                np.ones(len(rows), dtype=np.float32),
                (rows, cols),
            ),
            shape=(n, 2 * n),
        ).tocsr()

        # Частые взаимодействия не должны
        # полностью подавлять редкие.
        matrix.data = np.log1p(matrix.data)

        return normalize(matrix, norm="l2")

    def fit(self, data):
        self.hosts = list(data.cohort)
        self.host_index = pd.Index(self.hosts)

        flows = data.training_flows()

        behavior = self._behavior(flows, self.hosts)
        self.behavior_scaler = StandardScaler().fit(
            behavior
        )

        graph = self._graph(flows)

        k = min(
            self.role_dim,
            graph.shape[0] - 1,
            graph.shape[1] - 1,
        )

        if k < 1:
            raise ValueError(
                "Недостаточно хостов для обучения SVD"
            )

        self.svd = TruncatedSVD(
            n_components=k,
            random_state=self.seed,
        )

        roles = self.svd.fit_transform(graph)

        self.role_scaler = StandardScaler().fit(roles)

        return self

    def embed(self, data, window, hosts=None):
        hosts = list(
            hosts if hosts is not None
            else data.cohort
        )

        flows = (
            data.training_flows()
            if window == "train"
            else data.flows(window)
        )

        behavior = self._behavior(flows, hosts)
        behavior = self.behavior_scaler.transform(
            behavior
        )

        graph = self._graph(flows)
        roles = self.svd.transform(graph)
        roles = self.role_scaler.transform(roles)

        indices = self.host_index.get_indexer(hosts)

        if (indices < 0).any():
            raise ValueError(
                "Обнаружены хосты вне обучающей когорты"
            )

        roles = roles[indices]

        # Нормируем вклад каждой ветви
        # по статистикам обучающего окна.
        result = np.hstack([
            behavior * self.behavior_weight,
            roles * self.role_weight,
        ])

        if not np.isfinite(result).all():
            raise ValueError(
                "В эмбеддингах обнаружены NaN/Inf"
            )

        return pd.DataFrame(
            result,
            index=pd.Index(hosts, name="host_id"),
            columns=[
                f"emb_{i}"
                for i in range(result.shape[1])
            ],
        )
