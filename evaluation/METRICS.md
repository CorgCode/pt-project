# Evaluation metrics: definitions, inputs, interpretation

One evaluator (`python -m evaluation.evaluate`) runs every metric whose reference artifact
exists. Artifacts are built once by `python -m pipeline.reference` and shared by all methods.
**Each reference view answers a different question; do not read one as evidence for another.**

| # | metric | question it answers | reference file | kind |
|---|---|---|---|---|
| 1 | kNN Consistency / Jaccard / nDCG @k | do neighbours match a *later* behavioural-profile neighbourhood? | `neighbors.csv` | agreement with a behavioural **proxy** |
| 2 | Future-link ROC-AUC / PR-AUC | does similarity rank *future communication* pairs above non-links? | `edges.csv` | **predictive** (observed outcome) |
| 3 | Identity retrieval (Top-1 / MRR / Recall@k) | is a host recognisable in a later window? | two embedding files | **identity** / stability |
| 4 | Pair ROC-AUC / PR-AUC | does similarity separate shared-peer pairs from zero-overlap pairs? | `pairs.csv` | agreement with a **structural proxy** |
| 5 | Triplet Accuracy | is `a` closer to the shared-peer host than to a zero-overlap host? | `triplets.csv` | agreement with a **structural proxy** |
| 6 | Precision@k / Recall@k | within the judged candidates of a host, are relevant ones ranked first? | `pairs.csv` | agreement with a **structural proxy** |
| 7 | Linear probe | can an external host property be read out linearly? | `labels.csv` + `labels_source` | **external labels only** |

Behavioural proxy ≠ structural proxy ≠ future links ≠ identity ≠ functional role. Only (7)
speaks about roles, and only if real external labels exist. None are generated.

Similarity is cosine, matched by `host_id`; ties and zero vectors are handled explicitly.
Hosts: the evaluator scores exactly the hosts in the reference's `eval_hosts.csv` (the hosts evaluable in
the target window under the `train_only` cohort policy; the whole cohort otherwise). An embedding must cover
all of them (an error otherwise); extra hosts are ignored. Coverage is reported in `reference_info`.

Default evaluation split: the one declared by the reference manifest (`test`, or `validation`
for `reference/validation`). Tune on validation, report on test.

## Definitions

Let `s(x,y)` be cosine similarity of the embeddings of hosts x, y.

**1. kNN Consistency / Jaccard / nDCG@k.** `R_k(h)` = frozen top-k behavioural neighbours of
h (from the target window), `E_k(h)` = top-k by embedding, self excluded.
`Consistency@k = |R∩E|/k`, `Jaccard@k = |R∩E|/|R∪E|`, nDCG with relevance `k - rank_R`.
Averaged over hosts. Chance Consistency@k ≈ k/(n-1). Source: `evaluation/knn_consistency.py`.

**2. Future link prediction.** Candidates (`edges.csv`): positive = pair with ≥1 flow in the
target window; negative = sampled pair with none (ratio `negatives_per_positive`, seeded, once).
Score `s(a,b)`; ROC-AUC and average precision. Additional, because sampling distorts prevalence:
`PR-AUC lift = AP / sampled positive rate`; `PR-AUC@natural prevalence` reweights negatives so
the pool has the natural positive rate (target positives / all pairs of the evaluable hosts; every pair is
either linked or not in the window, so this denominator is the full pair population); and a **new-links**
view restricted to candidates unseen in the source window (persisting links are much easier).
"No flow observed" ≠ "impossible link". Negatives are uniform random pairs, i.e. easy.

**3. Identity retrieval.** For evaluation hosts, query = window-A vector, gallery = window-B
vectors of all evaluation hosts. `Top-1 = P(rank=1)`, `MRR = E[1/rank]`, `Recall@k = P(rank≤k)`
with exact expectations over random tie-breaking (a constant embedding scores exactly chance:
1/n, H_n/n, k/n). Independently fitted models may rotate their axes between windows, so raw cosine
across windows is valid **only for a shared coordinate system**; otherwise use `procrustes`,
fitted on a seeded half of the hosts and evaluated on the other half (identity of evaluated hosts
never informs the alignment). `python -m evaluation.identity_retrieval --embeddings-a A --embeddings-b B`.

**4. Pair ROC-AUC / PR-AUC.** Over judged pairs of the split. Labels: positive = target-window
peer-set Jaccard ≥ τ (τ = max(0.1, 99th percentile of pairwise Jaccard) unless fixed),
negative = Jaccard 0. Pairs in between are *unjudged* and are not in the file.
Prevalence reporting: `pr_auc_lift` (AP / sampled positive rate) and
`pair_pr_auc_at_judged_prevalence`, the AP reweighted to the positive rate **among all judged pairs**
(`judged_population_prevalence` = positives / (positives + negatives) from the manifest).
This is a *conditional* quantity: it describes pairs the rule can judge. Unjudged pairs - the vast
majority in a sparse network - are outside its support, so it is **not** a population AP and the judged
prevalence is **not** the positive rate of all host pairs; that is reported separately as
`population_positive_rate` (= positives / all possible pairs) together with `unjudged_share`.

**5. Triplet Accuracy.** `TA = mean[ 1{s(a,p) > s(a,n)} + 0.5·1{s(a,p) = s(a,n)} ]`; chance 0.5.
Distinct hosts required; missing hosts are an error.

**6. Precision@k / Recall@k.** Condensed-list protocol over judged candidates only:
`C(a)` = hosts judged against a; rank `C(a)` by `s(a,·)` (ties by host_id);
`P@k = hits_k/k`, `R@k = hits_k/|relevant(a)|`. Anchors need `|C(a)| ≥ k` and ≥1 relevant
(otherwise excluded and counted; `coverage` is reported). **A pair absent from `pairs.csv` is
unjudged, never a negative.** Reported with the oracle (best possible) and chance precision of the
same anchors, and `precision_lift`. Retrieval k values: 1, 3, 5 (pools hold ≤ 2·pairs_per_anchor
candidates, so large k would be degenerate).

**7. Linear probe.** StandardScaler + LogisticRegression (C=1, seed 0) fit on `split=train`
rows only, scored on `test`: accuracy, macro-F1, balanced accuracy, and the majority-class
baseline. Refused unless the manifest declares `labels_source`.

## Uncertainty

`--bootstrap N` (default 200 in the CLI) adds percentile 95% intervals (`metrics_ci`) by
resampling the evaluation items (hosts for kNN, pairs for AUC, triplets, anchors for P@k/R@k).
They quantify sampling variability of the *fixed evaluation set* only, not model-training
seeds, the choice of reference, or the choice of windows.

## Predictive value vs agreement with a proxy

* **Predictive (outcome observed later):** (2) future links; (3) identity stability.
* **Agreement with a proxy chosen by us:** (1) behavioural profile, (4)(5)(6) shared peers.
  A high score means "similar under that definition", which may or may not be what a user wants.
* **Ground truth:** only (7), and only with real external labels.

Baselines built from the *same* log windows/features as a reference (for example PCA of the
behavioural profile versus the behavioural neighbour proxy) share information with it by
construction; see `docs/METHODOLOGY.md` for the overlap audit.
