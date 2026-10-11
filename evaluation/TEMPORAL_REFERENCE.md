# Temporal holdout: what exactly k-NN consistency measures

**Research question:** If a model embeds a host using an earlier network-traffic
window, are its nearest neighbors similar under **behavior observed later**?

This is a **proxy for cross-window behavioral similarity**, not a test of known
host roles. No external manual labels are necessary for this protocol, but a
chosen reference definition **is necessary**. A high score cannot establish
that two machines perform the same function.

## Inputs

1. Training-period flows (train.csv), with columns
   src,dst,dst_port,proto,bytes,packets,ts.
2. Later evaluation-period flows (test.csv), same columns.
3. embeddings.csv produced **only from training-period information**, with
   the project's standard host_id,emb_0,... format.

Timestamps must be Unix seconds or ISO-8601 timestamps (UTC assumed for naive
ISO strings). The last training event must be **strictly before** the first
test event. The host universe is taken from embeddings.csv. Every host must be
active in both periods: never silently evaluate all-zero host vectors. Choose
the common cohort *before* training each method.

## Run

From the repo root:

~~~bash
python -m evaluation.temporal_reference \
  --train-flows data/train.csv \
  --test-flows data/test.csv \
  --embeddings submissions/METHOD/embeddings.csv \
  --output-dir reference/heldout

python -m evaluation.evaluate \
  --embeddings submissions/METHOD/embeddings.csv \
  --reference-dir reference/heldout \
  --method METHOD \
  --output benchmark/results/METHOD_heldout.json
~~~

For small experiments use --ks 2 5; the default is 5 10 20, requiring
at least 21 hosts. The command writes:

- reference/heldout/features.csv: future-period behavioral features,
  calculated with the existing behavioral reference builder.
- reference/heldout/manifest.json: tested time ranges, sample sizes, cohort
  size, chance overlap and training/reference-neighborhood stability.

The stability diagnostic compares neighbor lists from independently built
train and test behavioral profiles. If this is low, behavior itself may drift
between periods; low embedding consistency is then harder to interpret.

## Interpretation and caveats

- **Not ground truth.** Behavioral feature weights and time window determine
  what "similar" means. Report these assumptions.
- **Only window nonoverlap is verified automatically.** The code cannot
  prove what records the embedding model was trained on. Confirm and report
  training provenance separately.
- **Avoid leakage.** Never tune models, feature weights, temporal windows or
  hyperparameters repeatedly on the held-out test set. Tune using an earlier
  train/validation window, then freeze choices for an untouched final test.
- **Compare against baselines** (random, PCA, SVD), with identical hosts,
  splits and reference. Reference identity is a deliberately leaky ceiling.
- **No artificial classes.** This only produces features.csv, not
  pairs.csv, triplets.csv or labels.csv. Those require independent definitions.
- **Temporal generalization is not functional-role detection.** An embedding
  may preserve traffic-volume patterns without separating host types.
- **Raw LANL schemas vary.** Convert the chosen source to the canonical flow
  columns with a documented mapping before using this tool.

For the complete reference contract see
[reference specification](../benchmark/REFERENCE_DATA.md).
Tracking reference validity and independent evidence: issue #45.
