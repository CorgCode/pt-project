"""Baselines that follow the temporal protocol: FIT on the train window only.

Each baseline has
    fit(data)          reads only data.training_flows() / training_graph() / cohort
    embed(data, window) -> DataFrame[host_id x emb_*] for the cohort, from THAT window's flows
and declares `shared_coordinate_system`: True if embeddings of different windows live in one
coordinate system (fixed, train-fitted transform), False if every window is fitted separately
(then cross-window comparison needs alignment, see evaluation.identity_retrieval).

Scored embedding = embed(data, "train") (the source window); the later windows are used for
reference building and identity retrieval only.

Information overlap with the references (see docs/METHODOLOGY.md): behavioural baselines use the
same feature definitions as the behavioural-neighbour reference (different, later window);
the graph baseline uses peer structure like the shared-peer / future-link references (later
window). Overlap is by design for baselines and is why they are calibration points, not
independent evidence.
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

    def embed(self, data: ProcessedData, window: str) -> pd.DataFrame:
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

    def embed(self, data, window):
        offset = sum(ord(c) for c in window)                 # stable per-window stream
        rng = np.random.default_rng(self.seed * 1000 + offset)
        return _frame(rng.standard_normal((len(data.cohort), self.dim)), data.cohort)


class BehavioralBaseline(Baseline):
    name = "behavioral"
    shared_coordinate_system = True
    description = ("standardised behavioural profile (volume, degree, direction, ports, protocols, "
                   "hour-of-day) fitted on train; same transform applied to other windows")

    def __init__(self):
        self._fitted = None
        self._train_feats = None

    def fit(self, data):
        self._train_feats, self._fitted = build_reference_features(
            data.training_flows(), hosts=data.cohort, return_fitted=True)
        return self

    def _features(self, data, window) -> pd.DataFrame:
        if window == "train":
            return self._train_feats
        return build_reference_features(data.flows(window), hosts=data.cohort, fitted=self._fitted)

    def embed(self, data, window):
        return _frame(self._features(data, window).to_numpy(), data.cohort)


class BehavioralPCA(BehavioralBaseline):
    name = "behavioral_pca"
    description = "PCA (fit on train features) of the behavioural profile"

    def __init__(self, dim: int = 16, seed: int = 0):
        super().__init__()
        self.dim, self.seed = dim, seed

    def fit(self, data):
        super().fit(data)
        d = min(self.dim, *self._train_feats.shape)
        self._model = PCA(n_components=d, random_state=self.seed).fit(self._train_feats.to_numpy())
        return self

    def embed(self, data, window):
        return _frame(self._model.transform(self._features(data, window).to_numpy()), data.cohort)


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

    def embed(self, data, window):
        return _frame(self._model.transform(self._features(data, window).to_numpy()), data.cohort)


class GraphSpectral(Baseline):
    name = "graph_spectral"
    description = ("spectral embedding U*sqrt(S) of D^-1/2 A D^-1/2 (A = log1p flow counts) of the "
                   "window's FULL communication graph (cohort hosts and their peers); each window "
                   "is fitted separately, so coordinates are NOT shared across windows")

    def __init__(self, dim: int = 16, seed: int = 0):
        self.dim, self.seed = dim, seed

    def embed(self, data, window):
        flows = data.training_flows() if window == "train" else data.flows(window)
        flows = flows[flows["src"] != flows["dst"]]
        universe = pd.Index(sorted(set(flows["src"].astype(str)) | set(flows["dst"].astype(str)) | set(data.cohort)))
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
        emb = (u[:, order] * np.sqrt(s[order]))[universe.get_indexer(data.cohort)]
        if (np.linalg.norm(emb, axis=1) == 0).any():
            raise ValueError("graph_spectral: some cohort hosts have a zero embedding (isolated in the graph)")
        return _frame(emb, data.cohort)


def default_baselines(dim: int = 16, seed: int = 0) -> list[Baseline]:
    return [RandomBaseline(dim, seed), BehavioralBaseline(), BehavioralPCA(dim, seed),
            BehavioralSVD(dim, seed), GraphSpectral(dim, seed)]
