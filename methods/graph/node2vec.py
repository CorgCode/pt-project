"""node2vec (Grover & Leskovec, KDD 2016).

Блуждания второго порядка с параметрами p (возврат) и q (вглубь/вширь) -> skip-gram с negative sampling.
p = q = 1 — блуждания DeepWalk. Малый q держит блуждание в сообществе (похожие соседи), большой q уводит
наружу (сигнал структурной роли), малый p держит его рядом. p и q подбирайте на validation
(experiments/tune_methods.py); по умолчанию нейтральные p = q = 1. Окна обучаются отдельно.
"""
from __future__ import annotations

import numpy as np

from methods.graph.base import GraphEmbeddingMethod
from methods.graph.sgns import SkipGram
from methods.graph.walks import walk_corpus


class Node2Vec(GraphEmbeddingMethod):
    name = "node2vec"
    description = "second-order biased random walks (p, q) + skip-gram with negative sampling"
    reference = "Grover & Leskovec, node2vec: Scalable Feature Learning for Networks, KDD 2016"
    default_params = {"p": 1.0, "q": 1.0, "walk_length": 80, "num_walks": 10, "window": 10, "epochs": 1,
                      "lr": 0.025, "negative": 5, "batch_size": 512, "weight": "log1p", "graph": "hosts"}
    fast_params = {"num_walks": 2, "walk_length": 10, "window": 3}

    def __init__(self, dim: int = 16, seed: int = 0, **params):
        super().__init__(dim, seed, **params)
        if self.params["p"] <= 0 or self.params["q"] <= 0:
            raise ValueError("node2vec: p и q должны быть > 0")

    def embed_graph(self, g):
        p = self.params
        if (g.degree == 0).any():
            raise ValueError("в графе есть изолированные вершины")
        walks = walk_corpus(g, p["num_walks"], p["walk_length"], p=p["p"], q=p["q"], seed=self.seed)
        freq = np.bincount(walks.ravel(), minlength=g.n).astype(float)
        sg = SkipGram(g.n, self.dim, objective="neg", window=p["window"], negative=p["negative"], lr=p["lr"],
                      epochs=p["epochs"], batch_size=p["batch_size"], seed=self.seed, node_freq=freq)
        return sg.fit(walks).embeddings


METHODS = {"node2vec": Node2Vec}
