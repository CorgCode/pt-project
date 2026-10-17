"""Взвешенный неориентированный граф по потокам (CSR), общий для всех графовых методов.

graph="hosts"     вершины — все хосты окна, ребро — хосты общались;
graph="services"  двудольный граф хост — сервис: «uses:proto/port» для инициатора, «serves:proto/port» для получателя;
                  хосты с одинаковыми сервисами оказываются рядом (роль хоста определяют в основном сервисы);
graph="both"      оба вида рёбер сразу.
Сервисы, которыми пользуются меньше min_service_hosts хостов, выбрасываются (обезличенные редкие порты — шум).

weight: log1p(число потоков) | count | binary | hub_norm — log1p, делённый на sqrt(степень_a * степень_b),
чтобы хабы (сервер каталога, почта) не забирали на себя все случайные блуждания.
Вершины отсортированы; петли выброшены; соседи в строке отсортированы — проверка ребра бинарным поиском.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
import scipy.sparse as sp

WEIGHTS = ("log1p", "count", "binary", "hub_norm")
GRAPHS = ("hosts", "services", "both")
SERVICE_PREFIXES = ("serves:", "uses:")


@dataclass(frozen=True)
class Graph:
    nodes: np.ndarray            # host_id по возрастанию, dtype object
    indptr: np.ndarray           # (n+1,) int64
    indices: np.ndarray          # (2m,) int64, по возрастанию внутри строки
    weights: np.ndarray          # (2m,) float64, > 0

    @property
    def n(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.indices) // 2

    @cached_property
    def degree(self) -> np.ndarray:
        return np.diff(self.indptr)

    @cached_property
    def strength(self) -> np.ndarray:
        return np.add.reduceat(self.weights, self.indptr[:-1]) * (self.degree > 0)

    @cached_property
    def cum(self) -> np.ndarray:
        """cum[i] = sum(weights[:i]); позволяет выбрать соседа пропорционально весу бинарным поиском."""
        return np.concatenate([[0.0], np.cumsum(self.weights)])

    @cached_property
    def keys(self) -> np.ndarray:
        """Отсортированные ключи src * n + dst всех ориентированных рёбер (для быстрой проверки ребра)."""
        src = np.repeat(np.arange(self.n, dtype=np.int64), self.degree)
        return src * self.n + self.indices

    def has_edge(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        key = np.asarray(a, dtype=np.int64) * self.n + np.asarray(b, dtype=np.int64)
        pos = np.searchsorted(self.keys, key)
        return (pos < len(self.keys)) & (self.keys[np.minimum(pos, len(self.keys) - 1)] == key)

    def index_of(self, hosts) -> np.ndarray:
        """Номер строки каждого хоста; KeyError с примерами, если хоста нет среди вершин."""
        pos = np.searchsorted(self.nodes, np.asarray(list(hosts), dtype=object))
        pos = np.minimum(pos, self.n - 1)
        bad = [h for h, i in zip(hosts, pos) if self.nodes[i] != h]
        if bad:
            raise KeyError(f"у {len(bad)} хостов нет рёбер в графе этого окна (например {bad[:5]})")
        return pos

    def neighbors(self, v: int) -> np.ndarray:
        return self.indices[self.indptr[v]:self.indptr[v + 1]]

    def adjacency(self) -> sp.csr_matrix:
        return sp.csr_matrix((self.weights, self.indices, self.indptr), shape=(self.n, self.n))


def from_edges(src: np.ndarray, dst: np.ndarray, w: np.ndarray, nodes: np.ndarray) -> Graph:
    """Построить по целочисленным концам (строка на неориентированное ребро, без петель и дублей)."""
    n = len(nodes)
    a = sp.coo_matrix((np.r_[w, w], (np.r_[src, dst], np.r_[dst, src])), shape=(n, n)).tocsr()
    a.sum_duplicates()
    a.sort_indices()
    return Graph(nodes=np.asarray(nodes, dtype=object), indptr=a.indptr.astype(np.int64),
                 indices=a.indices.astype(np.int64), weights=a.data.astype(np.float64))


def _weights(lo: np.ndarray, hi: np.ndarray, counts: np.ndarray, n: int, weight: str) -> np.ndarray:
    w = {"log1p": np.log1p(counts), "count": counts.astype(float), "binary": np.ones(len(counts)),
         "hub_norm": np.log1p(counts)}[weight]
    if weight == "hub_norm":
        deg = np.bincount(np.r_[lo, hi], minlength=n).astype(float)
        w = w / np.sqrt(deg[lo] * deg[hi])
    return w


def _service_pairs(flows: pd.DataFrame, min_service_hosts: int) -> pd.DataFrame:
    """Пары (хост, сервис) с числом потоков; редкие сервисы выброшены."""
    proto = flows["proto"].astype(str) if "proto" in flows else pd.Series("", index=flows.index)
    svc = proto + "/" + flows["dst_port"].astype(str)
    pairs = pd.concat([pd.DataFrame({"a": flows["src"].astype(str), "b": "uses:" + svc}),
                       pd.DataFrame({"a": flows["dst"].astype(str), "b": "serves:" + svc})], ignore_index=True)
    pairs = pairs.groupby(["a", "b"]).size().rename("count").reset_index()
    users = pairs.groupby("b")["a"].nunique()
    common = pairs["b"].isin(users[users >= min_service_hosts].index)
    alone = ~pairs["a"].isin(pairs.loc[common, "a"])            # хост только с редкими сервисами — оставляем их
    return pairs[common | alone]


def from_flows(flows: pd.DataFrame, weight: str = "log1p", graph: str = "hosts", min_service_hosts: int = 5) -> Graph:
    if weight not in WEIGHTS:
        raise ValueError(f"weight должен быть одним из {WEIGHTS}")
    if graph not in GRAPHS:
        raise ValueError(f"graph должен быть одним из {GRAPHS}")
    f = flows[flows["src"].astype(str) != flows["dst"].astype(str)]
    if f.empty:
        raise ValueError("нет потоков между разными хостами: граф не построить")
    parts = []
    if graph in ("hosts", "both"):
        parts.append(pd.DataFrame({"a": f["src"].astype(str), "b": f["dst"].astype(str)}).assign(count=1))
    if graph in ("services", "both"):
        parts.append(_service_pairs(f, min_service_hosts))
    e = pd.concat(parts, ignore_index=True)
    nodes = np.unique(np.concatenate([e["a"].to_numpy(), e["b"].to_numpy()]).astype(object))
    ia, ib = np.searchsorted(nodes, e["a"].to_numpy()), np.searchsorted(nodes, e["b"].to_numpy())
    lo, hi = np.minimum(ia, ib), np.maximum(ia, ib)
    key = lo.astype(np.int64) * len(nodes) + hi
    ukey, inv = np.unique(key, return_inverse=True)
    counts = np.bincount(inv, weights=e["count"].to_numpy(float))
    lo, hi = ukey // len(nodes), ukey % len(nodes)
    return from_edges(lo, hi, _weights(lo, hi, counts, len(nodes), weight), nodes)
