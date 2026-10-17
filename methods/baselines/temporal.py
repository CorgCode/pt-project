"""Бейзлайны по временному протоколу: обучение ТОЛЬКО на train-окне.

У каждого бейзлайна
    fit(data)                    читает только data.training_flows() / training_graph() / cohort
    embed(data, window, hosts)   -> DataFrame host_id x emb_* по потокам ЭТОГО окна (по умолчанию — вся когорта)
и признак shared_coordinate_system: True, если все окна в одной системе координат (преобразование
подобрано на train и не меняется), False, если каждое окно обучается отдельно — тогда сравнение
между окнами требует выравнивания (evaluation.identity).

Оценивается embed(data, "train"); поздние окна нужны только для эталонов и узнаваемости.
Поведенческие бейзлайны используют те же признаки, что и эталон соседей (но другое окно), графовый —
структуру собеседников, как эталоны связей и пар. Поэтому это точки калибровки, а не независимое доказательство.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.decomposition import PCA, TruncatedSVD

from evaluation.knn_consistency import build_reference_features
from pipeline.data import ProcessedData


def _frame(X: np.ndarray, hosts: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(np.asarray(X, dtype=float), index=pd.Index(hosts, name="host_id"))
    return df.rename(columns=lambda c: f"emb_{c}")


class Baseline:
    name = "baseline"
    shared_coordinate_system = False
    description = ""

    def fit(self, data: ProcessedData) -> "Baseline":
        return self

    def embed(self, data: ProcessedData, window: str, hosts: list[str] | None = None) -> pd.DataFrame:
        raise NotImplementedError

    def info(self) -> dict:
        return {"name": self.name, "description": self.description,
                "fit_windows": ["train"] if self.name != "random" else [],
                "shared_coordinate_system": self.shared_coordinate_system}


class RandomBaseline(Baseline):
    name = "random"
    description = "seeded Gaussian noise; independent draw per window"

    def __init__(self, dim: int = 16, seed: int = 0):
        self.dim, self.seed = dim, seed

    def embed(self, data, window, hosts=None):
        hosts = list(hosts) if hosts is not None else data.cohort
        offset = sum(ord(c) for c in window)                 # свой поток случайных чисел на окно
        rng = np.random.default_rng(self.seed * 1000 + offset)
        full = rng.standard_normal((len(data.cohort), self.dim))     # строка на каждого хоста когорты
        return _frame(full[[data.cohort.index(h) for h in hosts]], hosts)


class BehavioralBaseline(Baseline):
    name = "behavioral"
    shared_coordinate_system = True
    description = ("standardised behavioural profile (volume, degree, direction, ports, protocols, "
                   "hour-of-day) fitted on train; same transform applied to other windows")

    exclude_blocks: tuple[str, ...] = ()

    def __init__(self, exclude_blocks: tuple[str, ...] | None = None):
        if exclude_blocks is not None:
            self.exclude_blocks = tuple(exclude_blocks)
        self._fitted = None
        self._train_feats = None

    def fit(self, data):
        self._train_feats, self._fitted = build_reference_features(
            data.training_flows(), hosts=data.cohort, return_fitted=True, exclude_blocks=self.exclude_blocks)
        return self

    def info(self) -> dict:
        out = super().info()
        out["excluded_feature_blocks"] = list(self.exclude_blocks)
        return out

    def _features(self, data, window, hosts=None) -> pd.DataFrame:
        hosts = list(hosts) if hosts is not None else data.cohort
        if window == "train":
            return self._train_feats.loc[hosts]
        return build_reference_features(data.flows(window), hosts=hosts, fitted=self._fitted,
                                        exclude_blocks=self.exclude_blocks)

    def embed(self, data, window, hosts=None):
        hosts = list(hosts) if hosts is not None else data.cohort
        return _frame(self._features(data, window, hosts).to_numpy(), hosts)


class BehavioralPCA(BehavioralBaseline):
    name = "behavioral_pca"
    description = "PCA (fit on train features) of the behavioural profile"

    def __init__(self, dim: int = 16, seed: int = 0, exclude_blocks: tuple[str, ...] | None = None):
        super().__init__(exclude_blocks)
        self.dim, self.seed = dim, seed

    def fit(self, data):
        super().fit(data)
        d = min(self.dim, *self._train_feats.shape)
        self._model = PCA(n_components=d, random_state=self.seed).fit(self._train_feats.to_numpy())
        return self

    def embed(self, data, window, hosts=None):
        hosts = list(hosts) if hosts is not None else data.cohort
        return _frame(self._model.transform(self._features(data, window, hosts).to_numpy()), hosts)


class BehavioralPortFree(BehavioralBaseline):
    """Поведенческий профиль БЕЗ единого признака по портам (#43)."""
    name = "behavioral_portfree"
    exclude_blocks = ("ports",)
    description = ("standardised behavioural profile (volume, degree, direction, protocols, hour-of-day) with the "
                   "port block removed entirely; fitted on train, same transform applied to other windows")


class BehavioralPortFreePCA(BehavioralPCA):
    name = "behavioral_portfree_pca"
    exclude_blocks = ("ports",)
    description = "PCA (fit on train features) of the port-free behavioural profile"


class BehavioralSVD(BehavioralBaseline):
    name = "behavioral_svd"
    description = "TruncatedSVD (fit on train features, no centering) of the behavioural profile"

    def __init__(self, dim: int = 16, seed: int = 0):
        super().__init__()
        self.dim, self.seed = dim, seed

    def fit(self, data):
        super().fit(data)
        d = min(self.dim, self._train_feats.shape[1] - 1, self._train_feats.shape[0] - 1)
        self._model = TruncatedSVD(n_components=d, random_state=self.seed).fit(self._train_feats.to_numpy())
        return self

    def embed(self, data, window, hosts=None):
        hosts = list(hosts) if hosts is not None else data.cohort
        return _frame(self._model.transform(self._features(data, window, hosts).to_numpy()), hosts)


class GraphSpectral(Baseline):
    name = "graph_spectral"
    description = ("spectral embedding U*sqrt(S) of D^-1/2 A D^-1/2 (A = log1p flow counts) of the "
                   "window's FULL communication graph (cohort hosts and their peers); each window "
                   "is fitted separately, so coordinates are NOT shared across windows")

    def __init__(self, dim: int = 16, seed: int = 0):
        self.dim, self.seed = dim, seed

    def embed(self, data, window, hosts=None):
        hosts = list(hosts) if hosts is not None else data.cohort
        flows = data.training_flows() if window == "train" else data.flows(window)
        flows = flows[flows["src"] != flows["dst"]]
        universe = pd.Index(sorted(set(flows["src"].astype(str)) | set(flows["dst"].astype(str)) | set(hosts)))
        i, j = universe.get_indexer(flows["src"].astype(str)), universe.get_indexer(flows["dst"].astype(str))
        a = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(len(universe),) * 2).tocsr()
        a = a + a.T
        a.data = np.log1p(a.data)
        deg = np.asarray(a.sum(axis=1)).ravel()
        inv = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
        norm = sp.diags(inv) @ a @ sp.diags(inv)
        k = min(self.dim, len(universe) - 2)
        v0 = np.random.default_rng(self.seed).standard_normal(len(universe))
        u, s, _ = svds(norm.astype(float), k=k, v0=v0)
        order = np.argsort(-s)
        emb = (u[:, order] * np.sqrt(s[order]))[universe.get_indexer(hosts)]
        if (np.linalg.norm(emb, axis=1) == 0).any():
            raise ValueError("graph_spectral: у части хостов нулевой эмбеддинг (изолированы в графе)")
        return _frame(emb, hosts)


def default_baselines(dim: int = 16, seed: int = 0, port_free: bool = False) -> list[Baseline]:
    """Стандартные бейзлайны; port_free=True добавляет варианты без портов (#43)."""
    out = [RandomBaseline(dim, seed), BehavioralBaseline(), BehavioralPCA(dim, seed),
           BehavioralSVD(dim, seed), GraphSpectral(dim, seed)]
    if port_free:
        out += [BehavioralPortFree(), BehavioralPortFreePCA(dim, seed)]
    return out
