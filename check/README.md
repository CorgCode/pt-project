# Pairwise evaluator for host embeddings

`pairwise_auc.py` answers a narrow question: do hosts marked as behaviourally
similar receive higher vector similarity than hosts marked as different? It is
one comparable quality signal, not proof that an embedding or anomaly detector
is universally correct.

## Reference data

`role_labels.csv` is a weak/silver annotation: a host receives a role only
when a known service port dominates and enough distinct clients use it.
`build_reference.py` deterministically selects six roles, makes disjoint
train/test host groups, then creates fixed 1:5 positive-to-negative pairs. A positive is two hosts with
the same silver role; a negative has different roles. Unlabelled or ambiguous
hosts are excluded.

This is safer for the benchmark than the previous `build_pairs.py`: it does not
use graph-neighbour overlap as the target, which would favour graph embeddings
trained on that same flow graph. Limitation: an embedding based primarily on
destination service ports can leak this silver signal. Report that result only
as a diagnostic, or evaluate it using labels from another time window/source.

Build the shared, compact reference once:

```bash
python3 build_reference.py --labels role_labels.csv --out-dir reference
```

It yields `reference/hosts.csv` (60 selected hosts, role and split) and
`reference/pairs.csv` (`host_a,host_b,label,split`). Commit/share those exact
files; never rebuild them separately for each embedding method.

To hand teammates a manageable raw-data slice for developing embeddings, make
a deterministic 50,000-flow sample that touches only those canonical hosts:

```bash
python3 extract_benchmark_flows.py --flows netflow
```

It scans the large input in chunks and writes `reference/netflow_sample.csv`.
The file is intentionally not committed here: it can be regenerated locally
and may have data-handling restrictions. Its rows include the normalised header
from `label_hosts_by_ports.py`.

## Evaluate an embedding

```bash
python3 pairwise_auc.py \
  --embeddings embeddings.csv \
  --pairs reference/pairs.csv \
  --split test --metric cosine --strict
```

The embedding file needs `device_id` and one or more numeric dimensions.
Cosine is the default because it removes arbitrary vector-scale differences;
negative Euclidean distance is only a diagnostic when all methods use the same
normalisation policy. `--strict` rejects incomplete coverage instead of quietly
evaluating an easier subset.

Output includes ROC-AUC, PR-AUC (average precision), prevalence and coverage.
Random ranking has ROC-AUC near 0.5; its PR-AUC baseline equals prevalence
(about 0.167 here). Compare results only on the same `test` pairs.

## Why both metrics

ROC-AUC measures whether a random positive ranks above a random negative. With
rare positives it can look high while producing too many false positives, since
the false-positive-rate denominator is dominated by negatives. PR-AUC focuses
on precision among retrieved positives and is usually more useful for the
short alert queue that an SOC analyst sees. Its absolute number should be read
against prevalence, not compared directly to ROC-AUC.

## Smoke test

```bash
python3 -m unittest -v test_pairwise_auc.py
```

The test makes structured and random synthetic vectors only to check that the
metric distinguishes them; they are not benchmark results.

`reference/example_role_structured_embeddings.csv` is the same kind of
deliberately label-structured smoke-test input. It demonstrates the CLI only;
do not use its perfect score to make a claim about a real embedding.
