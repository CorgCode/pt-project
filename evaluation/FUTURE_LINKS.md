# Future Link Prediction — fixed evaluation candidates

Objective: determine whether embeddings derived from **source-window data only**
rank communication edges observed in a later **target window** higher than
sampled pairs with no observed target-window communication.

This is a *future communication prediction* task, not host-role ground truth.

## Input

A documented adapter must map the selected raw LANL source into canonical
flows with columns src,dst,dst_port,proto,bytes,packets,ts. Raw LANL sources
vary; this code does not invent unknown fields.

Select exactly the same host cohort for every method. Use a predeclared
source/target split. Do not select or tune time boundaries by looking at final
test AUC. Source and target flows must be strictly non-overlapping in time.

## Produce one shared edges.csv

From the repository root:

    python -m evaluation.future_links build \
      --source-flows data/train.csv \
      --target-flows data/test.csv \
      --hosts submissions/METHOD/embeddings.csv \
      --seed 0 --negatives-per-positive 1 \
      --output reference/edges.csv

Required columns:

    host_a,host_b,source_window,target_window,label,split

Pairs are undirected, canonical, and distinct (host_a < host_b).
Positive: the pair communicated at least once in the target window.
Negative: no communication was OBSERVED in that window. It is not proof of
a physically impossible edge. Fixed-seed negative sampling is performed ONCE
and the same file must be reused across all methods.

The host cohort should be declared up front and contain the same hosts for
every method; pointing --hosts at an embedding is only a convenient way to
load the fixed IDs. Ensure it is not a model-selected cohort.

## Evaluate each method using the SAME fixed candidates

    python -m evaluation.future_links score \
      --embeddings submissions/METHOD/embeddings.csv \
      --edges reference/edges.csv \
      --output results/METHOD_future_links.json

The output reports:

- ROC-AUC, PR-AUC (average precision)
- number of examples, positives and candidate prevalence
- host coverage
- target/source windows and split
- SHA-256 fingerprint of edges.csv

Use the same fixed candidate file for every method. Scores are conditional
on the sampled candidate pool. A balanced 1:1 dataset does not represent the
very low positive prevalence in real communication-graph link prediction,
so do not call its PR-AUC production deployment AP.

No test-window data may be used to construct source-window embeddings.
Do not call an embedding's ability to predict future links evidence that it
directly identifies true functional host roles.

## Experimental next steps

- Compare all five real embedding methods and random, behavioral and graph
  baselines with exactly identical candidates and model training windows.
- Add a validation split for choosing seeds/hyperparameters and a final
  untouched test split.
- Add hard-negative experiments (fixed in advance), different temporal
  windows, sensible confidence intervals and missing/cold-start host reports.
- Real LANL metrics require a verified real-data pipeline and actual models.
