"""RolX (Henderson et al., KDD 2012): роли по рекурсивным структурным признакам.

1. ReFeX: структурные признаки без атрибутов вершин.
   * базовые: степень, сила (взвешенная степень), рёбра внутри эго-сети (deg + треугольники)
     и рёбра наружу (sum_{u in N(v)} deg(u) - 2 треугольника - deg(v));
   * рекурсивные: на каждом раунде СУММА и СРЕДНЕЕ по соседям каждого нового признака (раунд k — k-шаговая структура);
   * прореживание: логарифмическое разбиение на корзины; признак выбрасывается, если отличается от уже
     оставленного не более чем на s корзин на каждой вершине; раунды идут, пока появляется новое.
2. Роли: матрица признаков (log1p, по столбцам в [0, 1]) раскладывается V ~ G F неотрицательной факторизацией
   (обобщённая KL, детерминированная инициализация NNDSVD). Строки G (принадлежность ролям, L1-нормированы) —
   эмбеддинг; n_roles по умолчанию = размерность. n_roles="mdl" выбирает число ролей по минимальной длине описания;
   на чистых графах это часто минимум ролей, поэтому не по умолчанию.

Алгоритм не использует, кто есть кто: вершины с одинаковой ролью получают похожие векторы даже в разных
компонентах — в отличие от DeepWalk и подобных. Роли подбираются на каждом окне отдельно.
"""
from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

from benchmark.methods.base import GraphEmbeddingMethod
from benchmark.methods.graph_input import Graph

BASE_NAMES = ["degree", "strength", "ego_edges", "ego_out_edges"]


def _binary(g: Graph) -> sp.csr_matrix:
    a = g.adjacency().copy()
    a.data = np.ones_like(a.data)
    return a


def triangles(B: sp.csr_matrix, chunk: int = 2000) -> np.ndarray:
    """Число треугольников через каждую вершину (по кускам строк, чтобы хабы не съели память)."""
    out = np.zeros(B.shape[0])
    for s in range(0, B.shape[0], chunk):
        rows = B[s:s + chunk]
        out[s:s + chunk] = np.asarray((rows @ B).multiply(rows).sum(axis=1)).ravel() / 2.0
    return out


def base_features(g: Graph) -> np.ndarray:
    """(n, 4): степень, сила, рёбра внутри эго-сети, рёбра из эго-сети."""
    B = _binary(g)
    deg = np.asarray(B.sum(axis=1)).ravel()
    tri = triangles(B)
    nbr_deg = B @ deg
    return np.column_stack([deg, g.strength, deg + tri, nbr_deg - 2 * tri - deg])


def aggregate(B: sp.csr_matrix, F: np.ndarray) -> np.ndarray:
    """[сумма по соседям, среднее по соседям] для каждого столбца F."""
    deg = np.asarray(B.sum(axis=1)).ravel()
    s = B @ F
    return np.hstack([s, s / np.maximum(deg, 1.0)[:, None]])


def log_bin(x: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Логарифмические корзины: в корзине 0 — доля p самых малых, в корзине 1 — доля p остальных, ...

    Равные значения всегда в одной корзине.
    """
    n = len(x)
    order = np.argsort(x, kind="stable")
    sx = x[order]
    first = np.searchsorted(sx, sx, side="left")
    last = np.searchsorted(sx, sx, side="right")
    c = ((first + last) / 2.0) / n                                  # перцентиль среднего ранга в (0, 1)
    bins = np.empty(n, dtype=np.int64)
    bins[order] = np.floor(np.log(1.0 - c) / np.log(1.0 - p)).astype(np.int64)
    return bins


def refex(g: Graph, rounds: int = 3, s: int = 0, bin_p: float = 0.5, max_features: int = 200) -> tuple[np.ndarray, list[str]]:
    """Рекурсивные структурные признаки с прореживанием: матрица (n, f) и имена признаков."""
    B = _binary(g)
    cand = base_features(g)
    names = list(BASE_NAMES)
    kept_cols: list[np.ndarray] = []
    kept_bins: list[np.ndarray] = []
    kept_names: list[str] = []

    def accept(M: np.ndarray, nms: list[str]) -> list[int]:
        added = []
        for j in range(M.shape[1]):
            col = M[:, j]
            if np.ptp(col) == 0 or len(kept_cols) >= max_features:
                continue                                                # константа (или достигнут предел)
            b = log_bin(col, bin_p)
            if any(np.abs(b - kb).max() <= s for kb in kept_bins):
                continue                                                # похож на уже оставленный признак
            kept_cols.append(col); kept_bins.append(b); kept_names.append(nms[j]); added.append(j)
        return added

    added = accept(cand, names)
    for r in range(1, rounds + 1):
        if not added:
            break
        prev, prev_names = cand[:, added], [names[j] for j in added]
        cand = aggregate(B, prev)
        names = [f"sum({n})" for n in prev_names] + [f"mean({n})" for n in prev_names]
        added = accept(cand, names)
    if not kept_cols:
        raise ValueError("ReFeX не дал ни одного непостоянного признака (граф регулярный?)")
    return np.column_stack(kept_cols), kept_names


def prepare_matrix(F: np.ndarray) -> np.ndarray:
    """log1p и масштаб столбцов в [0, 1]: неотрицательный вход NMF с сопоставимыми масштабами."""
    V = np.log1p(np.maximum(F, 0.0))
    return V / np.maximum(V.max(axis=0), 1e-12)


def nmf_roles(V: np.ndarray, r: int, seed: int, max_iter: int = 400) -> np.ndarray:
    r = min(r, min(V.shape))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = NMF(n_components=r, init="nndsvda", solver="mu", beta_loss="kullback-leibler", max_iter=max_iter,
                    random_state=seed)
        G = model.fit_transform(V)
    return G


def kl_bits(V: np.ndarray, R: np.ndarray) -> float:
    """Обобщённая KL-дивергенция sum(V log(V/R) - V + R) в битах."""
    R = np.maximum(R, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(V > 0, V * np.log(V / R), 0.0)
    return float((t - V + R).sum() / np.log(2.0))


def mdl_roles(V: np.ndarray, rmin: int, rmax: int, seed: int, precision: float = 1.0 / 64) -> tuple[int, dict]:
    """Число ролей в [rmin, rmax], минимизирующее стоимость модели + ошибку восстановления (в битах)."""
    n, f = V.shape
    b = np.log2(1.0 / precision)
    costs = {}
    for r in range(rmin, min(rmax, min(n, f)) + 1):
        model = NMF(n_components=r, init="nndsvda", solver="mu", beta_loss="kullback-leibler", max_iter=400, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            G = model.fit_transform(V)
        # ошибка в единицах точности: KL однородна степени 1, KL(V/d || R/d) = KL(V || R) / d
        costs[r] = b * r * (n + f) + kl_bits(V, G @ model.components_) / precision
    return min(costs, key=costs.get), costs


class RolX(GraphEmbeddingMethod):
    name = "rolx"
    description = "ReFeX recursive structural features + NMF role memberships (structural-role embedding)"
    reference = "Henderson et al., RolX: Structural Role Extraction & Mining in Large Graphs, KDD 2012"
    default_params = {"n_roles": None, "rounds": 3, "similarity_bins": 0, "bin_fraction": 0.5, "max_features": 200,
                      "normalize": True, "nmf_iter": 400, "weight": "log1p"}
    fast_params = {"rounds": 2, "nmf_iter": 100}
    overlap = {"neighbors": "weak: degree/volume-like structure correlates with behaviour",
               "edges": "weak: structural roles, not specific peers, so link persistence is not encoded",
               "pairs": "weak: shared peers are not encoded (roles ignore identity)"}

    def embed_graph(self, g):
        p = self.params
        F, self.feature_names = refex(g, p["rounds"], p["similarity_bins"], p["bin_fraction"], p["max_features"])
        V = prepare_matrix(F)
        if p["n_roles"] == "mdl":
            r, self.mdl_costs = mdl_roles(V, 2, self.dim, self.seed)
        else:
            r = int(p["n_roles"] or self.dim)
        G = nmf_roles(V, r, self.seed, p["nmf_iter"])
        if p["normalize"]:
            G = G / np.maximum(G.sum(axis=1, keepdims=True), 1e-12)
        return G

    def info(self):
        out = super().info()
        out["n_structural_features"] = len(getattr(self, "feature_names", []))
        return out


METHODS = {"rolx": RolX}
