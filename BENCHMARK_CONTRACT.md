# Host Embedding Benchmark Contract (v1)

This is the canonical shared format for contributors working on issues #40–#45.
Implemented entry points are benchmark/validate_submission.py,
evaluation/evaluate.py, benchmark/REFERENCE_DATA.md, and
benchmark/build_leaderboard.py. Do not maintain incompatible schemas.

## Single-window embedding file

    host_id,emb_0,emb_1,emb_2
    Comp001,0.12,-0.31,0.04
    Comp002,0.06,0.25,0.81

- One row per host, unique nonblank host_id (a **string**).
- Finite numeric emb_* columns; equal dimensions within each file.
- Same host IDs across methods for a given experiment window.
- Embedding dimensions may differ across methods.
- Strings "001" and "1" represent different hosts.
- **Not canonical:** device_id,dim_0,... or id,embedding.

## Multiple windows

Use one file with the above columns per method and per time window.
An optional submission_manifest.csv lists each:

    method,window_id,path
    model_a,2026-10-06T10:00Z,embeddings/model_a/10.csv
    model_a,2026-10-06T11:00Z,embeddings/model_a/11.csv

Each (method,window_id) must be unique. Paths must exist. Commands:

    python validate_benchmark.py --manifest submission_manifest.csv
    python validate_benchmark.py submissions/model_a/embeddings.csv

Validation checks format and host cohorts, NOT actual provenance or window
boundaries of model training. A source window and a prediction target window
are distinct concepts. Do not infer training windows from filenames.

## Frozen evaluation reference files

See benchmark/REFERENCE_DATA.md:
- reference/features.csv: independently built behavioral proxy vectors.
- reference/neighbors.csv: frozen neighbor ranks from logs (issue #42).
- reference/pairs.csv: fixed pair judgments.
- reference/triplets.csv: fixed triplet judgments.
- reference/labels.csv: independently determined probe targets.

Every embedding must be measured against the same relevant cohort, time
window, split, pair candidate pool, negative sampling scheme and reference.

#40 future link prediction: labels are future observed/non-observed edges
within a FIXED candidate set; they are not functional role labels.
#41 temporal self-retrieval: same host ID across windows is an identity
judgment, NOT behavioral similarity. Independent embedding fits may rotate
their coordinates; direct cross-window cosine can require documented
alignment, otherwise resulting AUC has no interpretable meaning.
#42 neighborhood matching: frozen log-derived neighbors are a behavioral
PROXY, NOT true labels. Prefer held-out windows and report temporal drift.

## Outputs

The **implemented canonical output** is per-method benchmark/results/METHOD.json
from python -m evaluation.evaluate. The leaderboard code reads these JSONs.
Do not mix scores computed with different references or host cohorts.

A long-form results.csv can be a derived experimental report using:

    method,metric,score_name,score_value,split,window_id,n_examples,n_positive,positive_rate,coverage,details

This report is an *extension*, not currently consumed by the JSON leaderboard.
Store source and target windows, split, candidate coverage, reference
fingerprint and experiment config in details.

## Interpretation and leakage

- Behavioral proxy similarity does not imply matching host roles.
- Never derive test neighbors/pairs/triplets from the embedding being evaluated.
- Preserve split and train-vs-test separation; do not tune on held-out test.
- Flow-file window nonoverlap does not prove embedding training provenance.
- Reference identity is deliberately leaky and ONLY a sanity-check ceiling.
- Synthetic data results cannot be represented as LANL performance.
- For sampled non-edges, report sampling scheme and positive rate, and do
  not compare PR-AUC across different candidate distributions.
- Metrics needing real labels must be unavailable when these labels are
  missing; do not invent LLM/pseudo-label ground truth.

See evaluation/TEMPORAL_REFERENCE.md and issue #45 for the protocol.
