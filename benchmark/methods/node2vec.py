"""node2vec (Grover & Leskovec, KDD 2016).

Second-order biased random walks controlled by the return parameter `p` and the in-out parameter `q`
(`walks.second_order_step`: next node x after the edge (t, v) has probability proportional to
w(v, x) * alpha(t, x), alpha = 1/p if x == t, 1 if x is adjacent to t, 1/q otherwise), fed to skip-gram with
negative sampling, as in the reference implementation. p = q = 1 reduces to DeepWalk's walks with the negative-
sampling objective. Low q keeps walks in a community (homophily); high q pushes them outward (structural-role
signal); low p keeps walks local. Choose p and q on the validation window (experiments/tune_methods.py); the
defaults are the neutral p = q = 1. Coordinates of different windows are fitted separately (not shared).
"""
from __future__ import annotations

import numpy as np

from benchmark.methods.base import GraphEmbeddingMethod
from benchmark.methods.sgns import SkipGram
from benchmark.methods.walks import walk_corpus


class Node2Vec(GraphEmbeddingMethod):
    name = "node2vec"
    description = "second-order biased random walks (p, q) + skip-gram with negative sampling"
    reference = "Grover & Leskovec, node2vec: Scalable Feature Learning for Networks, KDD 2016"
    default_params = {"p": 1.0, "q": 1.0, "walk_length": 80, "num_walks": 10, "window": 10, "epochs": 1,
                      "lr": 0.025, "negative": 5, "batch_size": 512, "weight": "log1p"}
    fast_params = {"num_walks": 2, "walk_length": 10, "window": 3}

    def __init__(self, dim: int = 16, seed: int = 0, **params):
        super().__init__(dim, seed, **params)
        if self.params["p"] <= 0 or self.params["q"] <= 0:
            raise ValueError("node2vec: p and q must be positive")

    def embed_graph(self, g):
        p = self.params
        if (g.degree == 0).any():
            raise ValueError("graph has isolated nodes")
        walks = walk_corpus(g, p["num_walks"], p["walk_length"], p=p["p"], q=p["q"], seed=self.seed)
        freq = np.bincount(walks.ravel(), minlength=g.n).astype(float)
        sg = SkipGram(g.n, self.dim, objective="neg", window=p["window"], negative=p["negative"], lr=p["lr"],
                      epochs=p["epochs"], batch_size=p["batch_size"], seed=self.seed, node_freq=freq)
        return sg.fit(walks).embeddings


METHODS = {"node2vec": Node2Vec}
