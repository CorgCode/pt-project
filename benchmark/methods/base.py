"""Common wrapper that turns a graph-embedding algorithm into a benchmark method.

A subclass implements `embed_graph(graph) -> (n_nodes, d) array` (rows aligned with graph.nodes). The wrapper
builds the weighted host graph from ONE window's flows, caches the result, and returns the cohort hosts'
vectors in the canonical `host_id, emb_*` frame.

Protocol: `fit(data)` reads only the TRAIN window (`data.training_flows()`); the scored embedding is
`embed(data, "train")`. For a later window (identity retrieval) the algorithm is re-run on that window's
graph with the same hyper-parameters; such coordinates are NOT shared between windows, so
`shared_coordinate_system` is False unless a method is inductive (e.g. GraphSAGE overrides this).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from benchmark.baselines.temporal import Baseline, _frame
from benchmark.methods.graph_input import Graph, from_flows
from pipeline.data import ProcessedData


class GraphEmbeddingMethod(Baseline):
    name = "graph_method"
    default_params: dict = {}
    fast_params: dict = {}                           # tiny settings for smoke tests / CI (not for results)
    reference = ""                                   # paper / algorithm reference
    # information shared with each reference's SOURCE by construction (shown in the audit table)
    overlap = {"neighbors": "weak: peers/degree structure correlates with behaviour",
               "edges": "strong: link persistence (peers seen in train recur)",
               "pairs": "strong: shared peers = shared neighbourhood"}

    def __init__(self, dim: int = 16, seed: int = 0, **params):
        unknown = set(params) - set(self.default_params)
        if unknown:
            raise ValueError(f"{self.name}: unknown parameter(s) {sorted(unknown)}; valid: {sorted(self.default_params)}")
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.dim, self.seed = dim, seed
        self.params = {**self.default_params, **params}
        self._graphs: dict[str, Graph] = {}
        self._cache: dict[str, tuple] = {}
        self.fit_seconds: dict[str, float] = {}

    # ---- to implement
    def embed_graph(self, g: Graph, focus: np.ndarray | None = None) -> np.ndarray:
        """Embed the graph; rows aligned with g.nodes. `focus` (row indices) is given only if `uses_focus`."""
        raise NotImplementedError

    # ---- wrapper
    uses_focus = False        # True: the method builds its model only over the hosts to be embedded (e.g. struc2vec)

    def _window(self, data: ProcessedData, window: str, hosts: list[str]) -> tuple[Graph, np.ndarray]:
        if window not in self._graphs:
            flows = data.training_flows() if window == "train" else data.flows(window)
            self._graphs[window] = from_flows(flows, self.params.get("weight", "log1p"))
        g = self._graphs[window]
        focus_set = None
        if self.uses_focus:
            need = set(g.index_of(hosts).tolist())
            cached = self._cache.get(window)
            if cached is not None and need <= cached[2]:
                return g, cached[1]
            focus_set = need | (cached[2] if cached else set())          # later requests never shrink the focus
        elif window in self._cache:
            return g, self._cache[window][1]
        t0 = time.perf_counter()
        emb = np.asarray(self.embed_graph(g, np.array(sorted(focus_set))) if self.uses_focus else self.embed_graph(g),
                         dtype=float)
        self.fit_seconds[window] = time.perf_counter() - t0
        if emb.ndim != 2 or len(emb) != g.n or not np.isfinite(emb).all():
            raise ValueError(f"{self.name}: embed_graph returned an invalid array")
        self._cache[window] = (g, emb, focus_set)
        return g, emb

    def fit(self, data: ProcessedData) -> "GraphEmbeddingMethod":
        self._window(data, "train", data.cohort)
        return self

    def embed(self, data: ProcessedData, window: str, hosts: list[str] | None = None) -> pd.DataFrame:
        hosts = list(hosts) if hosts is not None else data.cohort
        g, emb = self._window(data, window, hosts)
        X = emb[g.index_of(hosts)]
        if (np.linalg.norm(X, axis=1) == 0).any():
            raise ValueError(f"{self.name}: some requested hosts have a zero embedding (cosine undefined)")
        return _frame(X, hosts)

    def info(self) -> dict:
        return {"name": self.name, "description": self.description, "reference": self.reference,
                "fit_windows": ["train"], "shared_coordinate_system": self.shared_coordinate_system, "overlap": self.overlap,
                "dim": self.dim, "seed": self.seed, "params": self.params,
                "fit_seconds": {k: round(v, 2) for k, v in self.fit_seconds.items()}}
