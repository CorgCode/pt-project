# Reference data contract

All embedding methods must be evaluated against the same reference artifacts. The evaluator must match rows by `host_id`, never by row position.

## Canonical files

### `reference/features.csv`
Used by neighborhood-preservation metrics such as k-NN consistency.

```csv
host_id,ref_0,ref_1,ref_2
C1,0.12,1.40,-0.22
C2,0.09,1.31,-0.18
```

Rules:
- unique `host_id`;
- numeric `ref_*` columns;
- no NaN/Inf;
- built independently from the embedding under evaluation.

### `reference/pairs.csv`
Used by pairwise ROC-AUC / PR-AUC and retrieval-style metrics.

```csv
host_a,host_b,label,split
C1,C2,1,test
C1,C9,0,test
```

`label=1` means similar/relevant, `label=0` means different/non-relevant.

### `reference/triplets.csv`
Used by Triplet Accuracy.

```csv
anchor,positive,negative,split
C1,C2,C9,test
```

### `reference/labels.csv`
Used by linear/logistic probes.

```csv
host_id,target,split
C1,workstation,train
C2,server,test
```

If several probe targets are used, add separate `target_*` columns or separate documented files.

## Splits

Use the same documented split definition across metrics where applicable. Do not create a new random split inside each metric implementation.

Preferred split policy for temporal data:
- train/reference-construction window;
- validation window if needed;
- held-out test window for reported scores.

## Leakage rules

Reference artifacts must not be derived from the embedding being evaluated or from clustering performed on that embedding.

If the embedding is trained from related behavioral features, this is allowed only when documented. Prefer held-out feature groups, another time window, or another independent similarity view when possible.

## Reproducibility

The pipeline that creates these files must be deterministic or use a fixed seed and must preserve stable host IDs. The same committed/generated reference artifacts must be used for every embedding method in a benchmark run.
