# Graph-embedding methods

Real implementations of the graph-embedding algorithms of issue #20, in numpy / scipy / scikit-learn only (no
gensim, no torch), so CI stays light and results are exactly reproducible (multi-worker word2vec is not).
Every method follows the benchmark's temporal protocol and is selected with `--methods`.

| method | algorithm | notes |
|---|---|---|
| `deepwalk` | uniform weighted walks + skip-gram with hierarchical softmax (Perozzi et al., KDD 2014) | below |
| `node2vec` | second-order biased walks (p, q) + skip-gram with negative sampling (Grover & Leskovec, KDD 2016) | `node2vec.md` |
| `rolx` | ReFeX recursive structural features + NMF roles (Henderson et al., KDD 2012) | `rolx.md` |
| `struc2vec` | multilayer structural-similarity graph, DTW degree-sequence distances (Ribeiro et al., KDD 2017) | `struc2vec.md` |
| `graphsage` | unsupervised mean-aggregator GraphSAGE, inductive (Hamilton et al., NeurIPS 2017) | `graphsage.md` |

The shared layer and DeepWalk come first; each other method is added by its own pull request (issue #20) together with
its notes file, and `--methods` accepts exactly the methods present in the checked-out tree.

## Use

```bash
python -m experiments.end_to_end --netflow data/raw/netflow_day-0*.bz2 --config cfg.json --workdir runs/lanl \
    --data-kind lanl --source-note "..." --methods deepwalk \
    --method-config '{"deepwalk": {"num_walks": 10, "walk_length": 80, "window": 10, "epochs": 1}}'
```

or `python -m experiments.run_experiment ... --methods deepwalk` on existing artifacts. `--methods all` selects every registered
method; `--fast-methods` swaps in tiny hyper-parameters (`fast_params`) for smoke tests and CI - its results are not meaningful. `--method-config` takes JSON
or a JSON file mapping a method to hyper-parameter overrides (unknown names are an error). The embedding
dimension is `--dim`; `--seed` seeds everything. Expected-but-unrun methods stay listed as *not evaluated*.

## Protocol

* **Input:** the weighted undirected host graph of ONE window's flows (`graph_input.py`): all hosts seen
  (cohort + peers), edge weight `log1p(#flows in either direction)`, self flows dropped.
* **Fit:** `fit(data)` reads only the train window (`data.training_flows()`), enforced by a test that makes any
  other window raise. The scored embedding is `embed(data, "train")` for the cohort hosts.
* **Later windows** (identity retrieval): the algorithm is re-run on that window's graph with the same
  hyper-parameters; coordinates are *not* shared between windows (`shared_coordinate_system: False`), so the
  Procrustes-aligned identity score is the meaningful one. (Inductive methods may declare otherwise.)
* **Overlap with the references:** graph methods use peer structure, as the future-link and shared-peer
  references do (later window). They are informative against the baselines but their agreement with those
  references is partly by construction; each method states its overlap in `info()["overlap"]` and the audit table.

## Shared building blocks

* `walks.py`: first-order and exact second-order biased walks, vectorised over walks. The node2vec transition
  `w(v,x) * alpha(t,x)` is sampled by rejection with an exact fallback; tests compare the empirical next-node
  distribution with the analytic one (chi-square) for several (p, q).
* `sgns.py`: skip-gram with negative sampling or hierarchical softmax over a Huffman tree, dynamic window,
  mini-batch SGD with linear learning-rate decay. Tests check both objectives' gradients against finite
  differences and that the hierarchical softmax is a proper distribution.
* `base.py`: `GraphEmbeddingMethod` (graph building, caching, host alignment, zero-vector guard, `info()`).
* Methods register themselves: any module here with `METHODS = {"name": Class}` is discovered automatically.

## Scale (measured on this implementation, one CPU core)

Skip-gram runs at ~200k (centre, context) pairs per second with negative sampling and ~140-190k with
hierarchical softmax; walk generation is negligible (uniform: 25k walks of length 40 in 0.1 s; biased: 0.5 s).
With the paper defaults (`num_walks=10, walk_length=80, window=10`) a walk yields ~836 pairs, so a graph of
20,000 hosts needs ~167M pairs, roughly **15 minutes per window per epoch**; 100,000 hosts ~ 70 minutes.
Memory is small (a 20k-node graph peaked at ~280 MB). Use fewer walks (`num_walks=5`) for a first real smoke run.
The scored (train) window is embedded once; identity retrieval re-runs the method on each later window, so budget
3 runs per method for validation + test identity. Quality of the results on real data is unknown until run.

## Adding a method

Subclass `GraphEmbeddingMethod`, set `name`, `default_params`, `reference`, implement
`embed_graph(graph) -> (n_nodes, d)`, and export `METHODS = {name: Class}`; add tests next to it.

## Tuning hyper-parameters (validation window only)

```bash
python -m experiments.tune_methods --processed-dir data/processed --reference-dir data/reference \
    --method deepwalk --grid '{"walk_length": [40, 80], "window": [5, 10]}' --metric pair_roc_auc --out runs/tune_deepwalk.json
```

Each grid point is fitted on the train window and scored on `reference/validation`; the test reference is never read
and `--target test` is refused. Fix the selection metric **before** looking at results, keep the grid small (at most 64
configurations), and report the frozen configuration's *test* score: the best validation score is optimistically
biased by the number of configurations tried (the output says so). Then run the final experiment once with
`--method-config` set to the chosen configuration.
