"""DeepWalk (Perozzi, Al-Rfou, Skiena, KDD 2014).

Truncated uniform random walks (here weighted by log1p flow count, p = q = 1) feed a skip-gram model whose
output layer is a hierarchical softmax over a Huffman tree built from node frequencies, exactly the objective
of the paper. Hyper-parameters (paper defaults): walks per node `num_walks` (gamma = 10), `walk_length` (t = 80),
`window` (w = 10); `epochs` passes over the walk corpus; `objective` may be switched to 'neg' (negative sampling).
Coordinates of different windows are fitted separately (not shared).
"""
from __future__ import annotations

import numpy as np

from benchmark.methods.base import GraphEmbeddingMethod
from benchmark.methods.sgns import SkipGram
from benchmark.methods.walks import walk_corpus


class DeepWalk(GraphEmbeddingMethod):
    name = "deepwalk"
    description = "uniform weighted random walks + skip-gram with hierarchical softmax"
    reference = "Perozzi et al., DeepWalk: Online Learning of Social Representations, KDD 2014"
    # inherits the graph-method overlap profile from GraphEmbeddingMethod
    default_params = {"walk_length": 80, "num_walks": 10, "window": 10, "epochs": 1, "lr": 0.025,
                      "objective": "hs", "negative": 5, "batch_size": 512, "weight": "log1p"}

    fast_params = {"num_walks": 2, "walk_length": 10, "window": 3}

    def embed_graph(self, g):
        p = self.params
        walks = walk_corpus(g, p["num_walks"], p["walk_length"], seed=self.seed)
        nodes = np.flatnonzero(g.degree > 0)
        if len(nodes) != g.n:
            raise ValueError("graph has isolated nodes")
        freq = np.bincount(walks.ravel(), minlength=g.n).astype(float)        # corpus frequencies (Huffman / negatives)
        sg = SkipGram(g.n, self.dim, objective=p["objective"], window=p["window"], negative=p["negative"],
                      lr=p["lr"], epochs=p["epochs"], batch_size=p["batch_size"], seed=self.seed, node_freq=freq)
        return sg.fit(walks).embeddings


METHODS = {"deepwalk": DeepWalk}
