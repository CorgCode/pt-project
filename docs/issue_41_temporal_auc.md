# Issue #41: temporal pairs and AUC protocol

## Contract

- Embeddings use `host_id,emb_0,emb_1,...`; flow CSVs passed to the builder use `src,dst`.
- A host is eligible only if it reaches `--min-flows` incident flows in both windows and exists in both windows for every method.
- Positive pair: the same host ID across windows. Negative pair: different host IDs across windows.
- Hosts are split into disjoint train/validation/test cohorts before pairs are generated. Generate `temporal_pairs.csv` once and reuse it for every method.
- Procrustes is fitted only on same-host positives in `train`. Validation and test are held out.
- Raw cosine is reported only for methods whose coordinate system is explicitly declared shared across windows.
- PR-AUC depends on positive prevalence, so use the same negative sampling and report `positive_rate`.

## Build pairs

Replace example paths with actual canonical paths and repeat embedding arguments for every method:

```bash
python -m evaluation.temporal_pairs \
  --flows-a data/windows/10_to_11/flows.csv \
  --flows-b data/windows/11_to_12/flows.csv \
  --embeddings-a method1=submissions/method1/10_to_11.csv \
  --embeddings-b method1=submissions/method1/11_to_12.csv \
  --embeddings-a method2=submissions/method2/10_to_11.csv \
  --embeddings-b method2=submissions/method2/11_to_12.csv \
  --window-a 10_to_11 --window-b 11_to_12 \
  --min-flows 5 --negatives-per-positive 5 --seed 41 \
  --output reference/temporal_pairs.csv
```

Run once, review the reference, and freeze it. Do not generate a different pair file for each method.

## Evaluate all methods

```bash
python -m evaluation.temporal_auc \
  --pairs reference/temporal_pairs.csv \
  --embeddings-a method1=submissions/method1/10_to_11.csv \
  --embeddings-b method1=submissions/method1/11_to_12.csv \
  --embeddings-a method2=submissions/method2/10_to_11.csv \
  --embeddings-b method2=submissions/method2/11_to_12.csv \
  --bootstrap 200 --seed 41 \
  --output runs/temporal/results.csv
```

Add `--shared-coordinates METHOD` only when the two embedding files genuinely share coordinates. The evaluator reports ROC-AUC and average precision (PR-AUC) for validation and test, plus host-cluster bootstrap intervals where estimable.

## Nearest-candidate identity retrieval

For each validation/test host, the evaluator queries its embedding in window A against all hosts in the same split in window B. The score is the fraction whose same-ID host is tied for the highest cosine similarity. Ties within `1e-12` count as hits. Candidate pools do not cross split boundaries. The result is written to the same `results.csv` as `metric=nearest_candidate_accuracy`; `score_value` is in `[0, 1]`. Raw cosine is included only when `--shared-coordinates METHOD` is declared; Procrustes-aligned cosine is always reported.

## Interpretation

These metrics assess ranking against temporal identity labels. They do not establish similarity of functional roles. Report real-data provenance and actual window definitions; synthetic smoke tests are not LANL results.
