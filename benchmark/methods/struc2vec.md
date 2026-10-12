# struc2vec

Ribeiro, Saverese, Figueiredo, *struc2vec: Learning Node Representations from Structural Identity*, KDD 2017.
Implementation: `struc2vec.py`.

Ring degree sequences, DTW distances accumulated over layers, a multilayer similarity graph and layer-staying biased
walks (see the module docstring for the exact definitions). Parameters: `layers` (K, default 3), `stay_prob` (q in the
paper, 0.3), `candidates_per_side` (OPT2 candidates per node; default `ceil(log2 n)` on each side).

* Only the hosts to be embedded become nodes of the multilayer graph (`uses_focus`); degrees come from the full window graph.
* The shared `GraphEmbeddingMethod` passes the hosts to embed and extends the focus if a later call asks for more.
* **Scale** (measured on heavy-tailed random graphs, 3 layers, one core): DTW dominates at ~0.5 ms per candidate pair
  (memoised). Distances: 1,000 nodes 2 s, 3,000 nodes 13 s, 8,000 nodes 54 s; skip-gram adds ~30 s at 8,000 nodes. Growth is
  roughly n log n, so ~5 min at 20,000 and ~30 min at 100,000 hosts are extrapolations; real hubs lengthen rings but
  not (much) the compressed sequences. Untested on real data.
* On the synthetic fixture it aligns only weakly with the references (pair AUC 0.61, kNN@10 0.09), as its declared overlap
  predicts, while recovering structural roles across disconnected components (>= 95% for every role).
