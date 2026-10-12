# node2vec

Grover & Leskovec, *node2vec: Scalable Feature Learning for Networks*, KDD 2016. Implementation: `node2vec.py`.

Second-order biased walks (return parameter `p`, in-out parameter `q`; exact sampler in `walks.py`) feed skip-gram with
negative sampling, as in the reference implementation. `p = q = 1` reproduces DeepWalk with the negative-sampling
objective bit for bit (tested). Low `q` keeps walks inside a community (homophily); high `q` pushes them outward;
low `p` keeps walks local. On planted-community graphs the within-minus-across cosine gap is monotone in `q`
(0.44 at q = 0.25, 0.41 at q = 1, 0.36 at q = 4 over 8 graphs).

Choose `p`, `q` on the validation window only:

```bash
python -m experiments.tune_methods --processed-dir data/processed --reference-dir data/reference \
    --method node2vec --grid '{"p": [0.25, 1, 4], "q": [0.25, 1, 4]}' --metric pair_roc_auc
```

Cost is the same as DeepWalk (skip-gram dominates, ~200k pairs/s); biased walks are ~5x slower to generate than
uniform ones but still negligible (0.5 s for 25,000 walks of length 40).
