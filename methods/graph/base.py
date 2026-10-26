"""Общая обёртка, превращающая алгоритм графового эмбеддинга в метод бенчмарка.

Наследник реализует embed_graph(graph) -> массив (n_nodes, d) (строки в порядке graph.nodes). Обёртка
строит взвешенный граф хостов по потокам ОДНОГО окна, кэширует результат и отдаёт векторы хостов
когорты в формате host_id, emb_*.

fit(data) читает только train; оценивается embed(data, "train"). Для позднего окна алгоритм
запускается заново на его графе с теми же параметрами — координаты между окнами не общие
(shared_coordinate_system=False), кроме индуктивных методов (GraphSAGE).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from methods.baselines.temporal import Baseline, _frame
from methods.graph.graph_input import Graph, from_flows
from pipeline.data import ProcessedData


class GraphEmbeddingMethod(Baseline):
    name = "graph_method"
    default_params: dict = {}
    fast_params: dict = {}                           # крошечные настройки для CI (не для результатов)
    reference = ""                                   # статья / алгоритм
    # что метод делит с ИСТОЧНИКОМ каждого эталона по построению (таблица аудита)
    overlap = {"neighbors": "weak: peers/degree structure correlates with behaviour",
               "edges": "strong: link persistence (peers seen in train recur)",
               "pairs": "strong: shared peers = shared neighbourhood"}

    def __init__(self, dim: int = 16, seed: int = 0, **params):
        unknown = set(params) - set(self.default_params)
        if unknown:
            raise ValueError(f"{self.name}: неизвестные параметры {sorted(unknown)}; допустимые: {sorted(self.default_params)}")
        if dim < 1:
            raise ValueError("dim должен быть >= 1")
        self.dim, self.seed = dim, seed
        self.params = {**self.default_params, **params}
        self._graphs: dict[str, Graph] = {}
        self._cache: dict[str, tuple] = {}
        self.fit_seconds: dict[str, float] = {}

    # ---- реализовать в наследнике
    def embed_graph(self, g: Graph, focus: np.ndarray | None = None) -> np.ndarray:
        """Эмбеддинг графа, строки в порядке g.nodes. focus (индексы строк) передаётся, только если uses_focus."""
        raise NotImplementedError

    # ---- обёртка
    uses_focus = False        # True: модель строится только по нужным хостам (например struc2vec)

    def _window(self, data: ProcessedData, window: str, hosts: list[str]) -> tuple[Graph, np.ndarray]:
        if window not in self._graphs:
            flows = data.training_flows() if window == "train" else data.flows(window)
            self._graphs[window] = from_flows(flows, self.params.get("weight", "log1p"), self.params.get("graph", "hosts"))
        g = self._graphs[window]
        focus_set = None
        if self.uses_focus:
            need = set(g.index_of(hosts).tolist())
            cached = self._cache.get(window)
            if cached is not None and need <= cached[2]:
                return g, cached[1]
            focus_set = need | (cached[2] if cached else set())          # последующие запросы фокус не сужают
        elif window in self._cache:
            return g, self._cache[window][1]
        t0 = time.perf_counter()
        emb = np.asarray(self.embed_graph(g, np.array(sorted(focus_set))) if self.uses_focus else self.embed_graph(g),
                         dtype=float)
        self.fit_seconds[window] = time.perf_counter() - t0
        if emb.ndim != 2 or len(emb) != g.n or not np.isfinite(emb).all():
            raise ValueError(f"{self.name}: embed_graph вернул некорректный массив")
        self._cache[window] = (g, emb, focus_set)
        return g, emb

    def _reset(self) -> None:
        """Забыть графы и эмбеддинги прошлого обучения: ключ кэша — только имя окна, и при повторном fit
        (другие данные или параметры) старые векторы иначе вернулись бы как новые."""
        self._graphs, self._cache, self.fit_seconds = {}, {}, {}

    def fit(self, data: ProcessedData) -> "GraphEmbeddingMethod":
        self._reset()
        self._window(data, "train", data.cohort)
        return self

    def embed(self, data: ProcessedData, window: str, hosts: list[str] | None = None) -> pd.DataFrame:
        hosts = list(hosts) if hosts is not None else data.cohort
        g, emb = self._window(data, window, hosts)
        X = emb[g.index_of(hosts)]
        if (np.linalg.norm(X, axis=1) == 0).any():
            raise ValueError(f"{self.name}: у части хостов нулевой эмбеддинг (косинус не определён)")
        return _frame(X, hosts)

    def info(self) -> dict:
        return {"name": self.name, "description": self.description, "reference": self.reference,
                "fit_windows": ["train"], "shared_coordinate_system": self.shared_coordinate_system, "overlap": self.overlap,
                "dim": self.dim, "seed": self.seed, "params": self.params,
                "fit_seconds": {k: round(v, 3) for k, v in self.fit_seconds.items()}}
