"""GraphSAGE без учителя с усреднением соседей (Hamilton, Ying, Leskovec, NeurIPS 2017).

Два слоя; для вершины v с выборкой соседей S(v) (равномерно, с повторами, размеры samples = (S1, S2)):

    h1_v = relu( W1 [ x_v || mean_{u in S1(v)} x_u ] ),         n1_v = h1_v / |h1_v|
    h2_v =       W2 [ n1_v || mean_{u in S2(v)} n1_u ],         z_v  = h2_v / |h2_v|

Обучение — функция потерь из статьи на парах, встретившихся в коротких блужданиях; отрицательные ~ степень^0.75,
общие для всего батча:

    J = -log s(z_u . z_v) - sum_{q=1..Q} log s(-z_u . z_{n_q})

Градиенты выписаны вручную и сверены с конечными разностями в тестах. Оптимизатор — Adam.

Признаки вершин x_v (features), нормируются по статистике train:
  behavioral  поведенческий профиль train-окна — те же признаки, что у поведенческого эталона (пересечение задокументировано);
  structural  [log степени, log силы, log средней степени соседей, коэффициент кластеризации];
  degree      [log степени].

GraphSAGE ИНДУКТИВЕН: обученные на train веса применяются к графу любого позднего окна, поэтому у окон
общая система координат (shared_coordinate_system = True), хотя сами входы дрейфуют.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
from scipy.special import expit

from methods.baselines.temporal import _frame
from methods.graph.base import GraphEmbeddingMethod
from methods.graph.graph_input import Graph, from_flows
from methods.graph.walks import walk_corpus
from evaluation.knn_consistency import build_reference_features

EPS = 1e-12


# ------------------------------------------------------------------------------------ признаки
def structural_features(g: Graph) -> np.ndarray:
    B = g.adjacency().copy()
    B.data[:] = 1.0
    deg = np.asarray(B.sum(axis=1)).ravel()
    tri = np.zeros(g.n)
    for s in range(0, g.n, 2000):
        rows = B[s:s + 2000]
        tri[s:s + 2000] = np.asarray((rows @ B).multiply(rows).sum(axis=1)).ravel() / 2.0
    nbr_mean = (B @ deg) / np.maximum(deg, 1.0)
    clust = tri / np.maximum(deg * (deg - 1) / 2.0, 1.0)
    return np.column_stack([np.log1p(deg), np.log1p(g.strength), np.log1p(nbr_mean), clust])


class Features:
    """Признаки вершин: подбираются на train и применяются к поздним окнам."""

    def __init__(self, kind: str):
        if kind not in ("behavioral", "structural", "degree"):
            raise ValueError("features должен быть 'behavioral', 'structural' или 'degree'")
        self.kind, self._fitted, self._mu, self._sd = kind, None, None, None

    def _raw(self, flows, g: Graph, fit: bool) -> np.ndarray:
        if self.kind == "behavioral":
            if fit:
                X, self._fitted = build_reference_features(flows, hosts=list(g.nodes), return_fitted=True)
            else:
                X = build_reference_features(flows, hosts=list(g.nodes), fitted=self._fitted)
            return X.to_numpy(float)
        if self.kind == "degree":
            return np.log1p(g.degree.astype(float))[:, None]
        return structural_features(g)

    def fit_transform(self, flows, g: Graph) -> np.ndarray:
        X = self._raw(flows, g, True)
        self._mu, self._sd = X.mean(0), np.where(X.std(0) > 0, X.std(0), 1.0)
        return (X - self._mu) / self._sd

    def transform(self, flows, g: Graph) -> np.ndarray:
        return (self._raw(flows, g, False) - self._mu) / self._sd


# ------------------------------------------------------------------------------------ модель
def sample_neighbors(g: Graph, nodes: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """size соседей на вершину, равномерно с повторами (степень каждой вершины >= 1)."""
    deg = g.degree[nodes]
    off = np.minimum((rng.random((len(nodes), size)) * deg[:, None]).astype(np.int64), deg[:, None] - 1)
    return g.indices[g.indptr[nodes][:, None] + off]


def _unit(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nrm = np.sqrt((P * P).sum(axis=1, keepdims=True)) + EPS
    return P / nrm, nrm


class Batch:
    """Всё случайное для одного батча (чтобы потери были детерминированной функцией весов)."""

    def __init__(self, targets, nbr2, nbr1, pu, pv, neg):
        self.targets, self.nbr2, self.nbr1, self.pu, self.pv, self.neg = targets, nbr2, nbr1, pu, pv, neg


def forward(W1: np.ndarray, W2: np.ndarray, X: np.ndarray, b: Batch):
    T, S2 = b.nbr2.shape
    M_nodes = np.concatenate([b.targets, b.nbr2.ravel()])
    C1 = np.hstack([X[M_nodes], X[b.nbr1].mean(axis=1)])
    P1 = C1 @ W1
    N1, H1n = _unit(np.maximum(P1, 0.0))
    h = N1.shape[1]
    C2 = np.hstack([N1[:T], N1[T:].reshape(T, S2, h).mean(axis=1)])
    P2 = C2 @ W2
    Z, P2n = _unit(P2)
    return Z, (C1, P1, N1, H1n, C2, P2n, T, S2, h)


def loss_and_grads(W1, W2, X, b: Batch, want_grads: bool = True):
    Z, (C1, P1, N1, H1n, C2, P2n, T, S2, h) = forward(W1, W2, X, b)
    B = len(b.pu)
    Zu, Zv, Zn = Z[b.pu], Z[b.pv], Z[b.neg]
    s_pos, s_neg = (Zu * Zv).sum(1), Zu @ Zn.T
    loss = (-np.log(expit(s_pos) + 1e-300).sum() - np.log(expit(-s_neg) + 1e-300).sum()) / B
    if not want_grads:
        return loss, None
    d_pos, d_neg = (expit(s_pos) - 1.0) / B, expit(s_neg) / B
    dZ = np.zeros_like(Z)
    np.add.at(dZ, b.pu, d_pos[:, None] * Zv + d_neg @ Zn)
    np.add.at(dZ, b.pv, d_pos[:, None] * Zu)
    np.add.at(dZ, b.neg, d_neg.T @ Zu)
    dP2 = (dZ - Z * (Z * dZ).sum(1, keepdims=True)) / P2n                 # через z = h / |h|
    dW2 = C2.T @ dP2
    dC2 = dP2 @ W2.T
    dN1 = np.zeros_like(N1)
    dN1[:T] += dC2[:, :h]
    dN1[T:] += np.repeat(dC2[:, h:] / S2, S2, axis=0)                      # среднее по S2 выбранным соседям
    dH1 = (dN1 - N1 * (N1 * dN1).sum(1, keepdims=True)) / H1n
    dP1 = dH1 * (P1 > 0)
    return loss, (C1.T @ dP1, dW2)


def infer(W1: np.ndarray, W2: np.ndarray, g: Graph, X: np.ndarray) -> np.ndarray:
    """Детерминированный эмбеддинг всех вершин: среднее по всем соседям с обученными весами."""
    A = g.adjacency().copy()
    A.data[:] = 1.0
    P = sp.diags(1.0 / np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1.0)) @ A            # оператор среднего (строки суммируются в 1)
    N1, _ = _unit(np.maximum(np.hstack([X, P @ X]) @ W1, 0.0))
    return _unit(np.hstack([N1, P @ N1]) @ W2)[0]


class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999):
        self.lr, self.b1, self.b2, self.t = lr, b1, b2, 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, params, grads):
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            p -= self.lr * (m / (1 - self.b1 ** self.t)) / (np.sqrt(v / (1 - self.b2 ** self.t)) + 1e-8)


def train(g: Graph, X: np.ndarray, dim: int, *, hidden: int, samples: tuple[int, int], walk_length: int,
          num_walks: int, window: int, negatives: int, batch_size: int, epochs: int, lr: float, seed: int):
    """Вернуть (W1, W2, история потерь)."""
    rng = np.random.default_rng(seed)
    d0 = X.shape[1]
    W1 = rng.normal(0, np.sqrt(2.0 / (2 * d0)), (2 * d0, hidden))
    W2 = rng.normal(0, np.sqrt(1.0 / (2 * hidden)), (2 * hidden, dim))
    opt = Adam([W1, W2], lr)
    walks = walk_corpus(g, num_walks, walk_length, seed=seed)
    us, vs = [], []
    for o in range(1, min(window, walk_length - 1) + 1):
        us += [walks[:, :-o].ravel(), walks[:, o:].ravel()]
        vs += [walks[:, o:].ravel(), walks[:, :-o].ravel()]
    pu_all, pv_all = np.concatenate(us), np.concatenate(vs)
    keep = pu_all != pv_all
    pu_all, pv_all = pu_all[keep], pv_all[keep]
    cdf = np.cumsum((g.degree.astype(float) ** 0.75) / (g.degree.astype(float) ** 0.75).sum())
    history = []
    for _ in range(epochs):
        order = rng.permutation(len(pu_all))
        total, count = 0.0, 0
        for s in range(0, len(order), batch_size):
            sel = order[s:s + batch_size]
            u, v = pu_all[sel], pv_all[sel]
            neg = np.minimum(np.searchsorted(cdf, rng.random(negatives)), g.n - 1)
            targets, inv = np.unique(np.concatenate([u, v, neg]), return_inverse=True)
            nb2 = sample_neighbors(g, targets, samples[1], rng)
            m_nodes = np.concatenate([targets, nb2.ravel()])
            nb1 = sample_neighbors(g, m_nodes, samples[0], rng)
            b = Batch(targets, nb2, nb1, inv[:len(u)], inv[len(u):2 * len(u)], inv[2 * len(u):])
            loss, (g1, g2) = loss_and_grads(W1, W2, X, b)
            opt.step([W1, W2], [g1, g2])
            total += loss * len(u); count += len(u)
        history.append(total / max(count, 1))
    return W1, W2, history


# ------------------------------------------------------------------------------------ метод бенчмарка
class GraphSAGE(GraphEmbeddingMethod):
    name = "graphsage"
    shared_coordinate_system = True
    description = "unsupervised GraphSAGE (mean aggregator, 2 layers, neighbour sampling); inductive"
    reference = "Hamilton, Ying, Leskovec, Inductive Representation Learning on Large Graphs, NeurIPS 2017"
    default_params = {"features": "behavioral", "hidden": 64, "samples": [25, 10], "walk_length": 5, "num_walks": 10,
                      "window": 2, "negatives": 20, "batch_size": 256, "epochs": 2, "lr": 0.01, "weight": "log1p"}
    fast_params = {"hidden": 16, "samples": [5, 3], "num_walks": 2, "epochs": 1, "batch_size": 128}
    overlap = {"neighbors": "strong with features='behavioral': same feature definitions as the behavioural reference",
               "edges": "strong: link persistence through the neighbourhood aggregation",
               "pairs": "strong: shared peers = shared neighbourhood"}

    def __init__(self, dim: int = 16, seed: int = 0, **params):
        super().__init__(dim, seed, **params)
        s = self.params["samples"]
        if len(s) != 2 or min(s) < 1:
            raise ValueError("graphsage: samples — два целых > 0 (соседи 1-го и 2-го слоя)")
        Features(self.params["features"])                                       # проверяет имя набора признаков
        self.W1 = self.W2 = self.features_ = None
        self.loss_history: list[float] = []
        self._embedded: dict[str, tuple[Graph, np.ndarray]] = {}                # свой кэш, независимый от базового класса

    def fit(self, data):
        p = self.params
        flows = data.training_flows()
        g = from_flows(flows, p["weight"])
        self.features_ = Features(p["features"])
        X = self.features_.fit_transform(flows, g)
        t0 = time.perf_counter()
        self.W1, self.W2, self.loss_history = train(
            g, X, self.dim, hidden=p["hidden"], samples=tuple(p["samples"]), walk_length=p["walk_length"],
            num_walks=p["num_walks"], window=p["window"], negatives=p["negatives"], batch_size=p["batch_size"],
            epochs=p["epochs"], lr=p["lr"], seed=self.seed)
        self._embedded["train"] = (g, infer(self.W1, self.W2, g, X))
        self.fit_seconds["train"] = time.perf_counter() - t0
        return self

    def embed(self, data, window, hosts=None):
        if self.W1 is None:
            raise RuntimeError("graphsage: сначала fit(data), потом embed()")
        return super().embed(data, window, hosts)

    def _window(self, data, window, hosts=None):
        if self.W1 is None:
            raise RuntimeError("graphsage: сначала вызовите fit(data)")
        if window not in self._embedded:
            flows = data.flows(window)                                           # только применение: веса не меняются
            g = from_flows(flows, self.params["weight"])
            self._embedded[window] = (g, infer(self.W1, self.W2, g, self.features_.transform(flows, g)))
        return self._embedded[window]

    def info(self):
        out = super().info()
        out["fit_windows"] = ["train"]
        out["final_training_loss"] = self.loss_history[-1] if self.loss_history else None
        return out


METHODS = {"graphsage": GraphSAGE}
