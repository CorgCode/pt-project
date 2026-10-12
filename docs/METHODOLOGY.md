# Methodology and validity audit

Status: the pipeline, metrics and audit are implemented and tested; **all numbers below come from a
synthetic LANL-format fixture** (`python -m pipeline.fixtures`, seed 0, 300 hosts) and demonstrate the
protocol, not network behaviour. Real LANL data was not accessible (the download link returned 401).
Every run is labelled `synthetic_fixture`, `unverified` (default for any other input) or `lanl_declared`
(an operator declaration with a source note); provenance is **never verified** by the pipeline, and reports
say which label applies. Findings on synthetic or unverified data are not LANL findings.

## 1. Question and scope

Can an embedding of a host, computed from an *earlier* window of network traffic, be used to find
similar hosts, predict later communication, and recognise the host again? The benchmark deliberately
does **not** require host-role labels: none exist for LANL netflow, and none are fabricated. Every
metric states what it measures; they must not be conflated:

| view | judgment comes from | what a high score means |
|---|---|---|
| behavioural neighbourhood (kNN) | later-window behavioural profile of each host | embedding neighbours match who behaves alike later (a **proxy**) |
| future links | whether a pair really communicates later | embedding supports **predicting** communication |
| identity | same `host_id` in two windows | embedding is stable **and** discriminative over time |
| shared-peer structure (pairs, triplets, P@k/R@k) | overlap of later communication partners | embedding agrees with a **structural proxy** |
| linear probe | external host labels with declared provenance | an external property is linearly readable (unavailable without labels) |

## 2. Data protocol and leakage control

* Source: LANL Unified Host and Network Dataset (2017) netflow; exact schema in `pipeline/README.md`.
* Windows `train < validation < test` are half-open, ordered, non-overlapping (validated); flows still
  running at a window end are dropped.
* **Cohort policy** (decided before any model exists; see 4.9). Default `train_only`: the cohort is the
  hosts active in the train window; a host is *evaluable* in a later window only if active there, and every
  metric is computed on the evaluable hosts with the coverage reported. The legacy `all_windows` policy
  (hosts active in every window) lets future activity decide who is evaluated and is a transductive
  stable-host benchmark.
* **Provenance** (`pipeline/provenance.py`): input is `unverified` unless declared; a LANL declaration needs a
  source note, LANL-style file names, and no fixture marker/signature. A declaration is recorded as such.
* Models may read only `ProcessedData.training_flows()/training_graph()`. This is enforced for the
  built-in baselines by a test that makes `flows(w)`/`graph(w)` raise for `w != train` during `fit()`
  (`benchmark/baselines/test_temporal.py`). External methods cannot be audited by code and must declare
  their training window (the reference manifest says `provenance_verified: false`).
* References are built only from a later window (`validation`, `test`); hyper-parameters and
  analysis choices are made on `validation`, final numbers reported on `test`.
* The learned link scorer is fitted on the validation candidates and scored on test; it is refused
  unless the fit window ends before the evaluation window and both share the source window.
* The leaderboard refuses to mix different reference files (every file is SHA-256 fingerprinted, also
  the fit reference), host cohorts, splits, or synthetic with real data.

## 3. Is each reference independent of the evaluated embedding?

References are computed from raw logs of a later window, never from an embedding. What can still be
shared is **information**: a method's input and a reference's source can overlap by construction. The
audit records this explicitly (`experiments/audit.py::OVERLAP`, printed in every report):

| method | behavioural neighbours | future links | shared peers |
|---|---|---|---|
| random | none | none | none |
| behavioral / PCA / SVD | same feature definitions, earlier window | none | weak (behaviour correlates with peers) |
| graph_spectral | weak (degree leaks into the spectrum) | **strong** (links persist) | **strong** (peer sets persist) |
| reference_ceiling | identical by construction | none | weak |

Consequences: the baselines are calibration points, not independent evidence; a graph embedding
scoring well on link prediction is largely *link persistence* (see 4.5); behavioural features scoring
well on the behavioural proxy mostly measures *temporal persistence of behaviour*. A real learned model
must be compared against these baselines, not against chance alone.

## 4. Validity checks (implemented, run by `experiments.run_experiment`)

**4.1 Sensitivity to feature weights.** The behavioural proxy depends on block weights (volume, degree,
direction, ports, protocols, hour of day). The audit rebuilds the reference under 12 perturbed weight
vectors (each weight ×exp(N(0,0.5)); one block removed in a third of the configs) and reports (i) agreement
of the perturbed neighbour lists with the canonical reference and (ii) the Kendall tau of the method
ranking. Fixture: neighbour agreement 0.74 on average (min 0.58; chance 0.033); ranking tau mean 0.81, min 0.56.
Interpretation: the proxy is moderately robust but the *ordering of close methods* can change with the
weights, so differences between near-equal methods should not be over-read.

**4.2 Stability across time.** Consistency@10 between behavioural neighbour lists built from two different
windows (fixture: train→test 0.28, train→validation 0.28, validation→test 0.33; chance 0.033). This bounds
what any earlier-window embedding can be expected to reach on the kNN metric: the behavioural baseline,
which is exactly "persistence", scores 0.28. A model that does not clearly exceed the persistence
baseline has not shown predictive value beyond behaviour that simply persists.

**4.3 Chance bands.** 30 random embeddings are scored on the test reference (`audit.json:random_null`).
Fixture: kNN 0.034±0.003, link AUC (cosine) 0.501±0.007, pair AUC 0.501±0.021, triplet 0.502±0.023.
Use these, not 0.5, to judge significance.

Bootstrap intervals (`--bootstrap`). Pairs and triplets **share hosts**, so resampling them as independent
items understates the variance. The link, pair and triplet metrics therefore use a **host-cluster bootstrap**:
hosts are resampled with replacement and an item is weighted by the product of its endpoints' multiplicities
(`metrics_ci` plus `ci_methods` in every result). In a simulation with host-level effects shared across pairs
(`evaluation/test_host_bootstrap.py`) the nominal 95% interval of the independent-pairs bootstrap covered the
true AUC in 70% of the draws and was half as wide, versus 97% for the host-cluster bootstrap. kNN and
retrieval intervals resample hosts / anchors. All intervals cover evaluation-set sampling only - not model
training seeds, the choice of reference, or the choice of windows.

**4.4 Candidate prevalence.** Sampled candidate pools are balanced (50% positives) while real prevalence is
low (fixture natural prevalence 0.083). PR-AUC from a sampled pool is therefore not a deployment AP. The
evaluator reports PR-AUC *lift* (AP / sampled rate), PR-AUC *reweighted to the natural prevalence*, and the
pool's positive rate. Reweighting fixes only the class prior, **not** which negatives were drawn: uniform
random non-links are easy, so absolute values remain optimistic. Shared-peer pairs: see 4.8 - their reweighting target is the prevalence **among judged pairs**, a
conditional quantity, not a population prevalence.

**4.5 Persistence, and why raw cosine is the wrong link scorer.** 51% of future links were already present
in the source window. The trivial predictor "pair seen in source" has ROC-AUC 0.708 and is stored with every
result. Cosine similarity of the embeddings scores ≈0.47–0.49 for *every* method, including the one built
from the communication graph: in client–server traffic linked hosts are dissimilar and same-role clients
(similar, never linked) populate the negatives. Reporting only cosine would wrongly conclude the embeddings
carry no link information. The evaluator therefore also reports an edge classifier on `[|e_a−e_b|, e_a∘e_b]`
fitted on the validation candidates and scored on test (fixture: graph 0.730, behavioural 0.668, random 0.546 vs
persistence 0.708; the reference ceiling is excluded - it only embeds the target-evaluable hosts). **Random is above 0.5 for the learned score** (null band 0.543±0.007):
random vectors act as host identifiers, so the classifier memorises hub hosts. The learned score must
always be read against the random row; a `new-links only` view (candidates unseen in the source window) is
reported separately.

**4.6 Cross-window alignment.** Independently fitted embeddings (here: graph_spectral) rotate between
windows; raw cross-window cosine is then meaningless (fixture: 0.000 Top-1 vs 0.180 after Procrustes). Identity
retrieval fits the orthogonal map on a seeded half of the hosts and evaluates on the other half, so evaluated
identities never inform the alignment; the raw score is only interpretable for methods that declare one
coordinate system. Ties are scored by exact expectation, so a constant embedding is exactly at chance.
Procrustes can also absorb systematic drift of a shared-coordinate method (behavioural 0.047 → 0.133), so the
two scores answer different questions (stability of fixed features vs recoverability after a linear map).

**4.7 Ranking stability between splits.** Kendall tau between the validation and test orderings per metric
(fixture: kNN 0.71, pair AUC 1.00, triplet 0.93, cosine link AUC 0.00). Where all methods sit near chance
(cosine link AUC) the ordering is noise and tau ≈ 0 is expected.

**4.8 Judged-pair incompleteness and what "prevalence" means for pairs.** Pairs are judged by a fixed rule
(peer-set Jaccard ≥ τ positive, = 0 negative, in between **unjudged**). Unjudged pairs are never negatives;
P@k/R@k use the condensed list over judged candidates and report coverage and the oracle/chance precision of
the same anchors. The positive rate computed from the judged counts, `judged_population_prevalence`
= positives / (positives + negatives), is **conditional on a pair being judged**; it is not the positive rate
of all host pairs (`population_positive_rate` = positives / all possible pairs, reported next to
`unjudged_share`; fixture: judged prevalence 0.070 versus population rate 0.010, 86% of pairs unjudged).
`pair_pr_auc_at_judged_prevalence` is therefore an AP over the judgeable pairs with the
sampled class prior corrected - not a population AP. (Review of #50; regression tests with a 50x larger
unjudged share in `evaluation/test_prevalence_semantics.py`.) For future links every pair is either linked or
not in the window, so there the natural prevalence (positives / all pairs of the evaluable hosts) is a
population quantity - with the caveat that "no flow observed" is not "impossible link".

**4.9 Cohort selection and coverage.** With the legacy `all_windows` policy a host enters the benchmark only if
it is active in the validation *and test* windows, so future activity influences inclusion. Models still get
no future flows or edges, but the scores condition on host survival and overstate persistence, identity and
link persistence for prospective use. The default `train_only` policy removes this: the cohort is defined from
the train window and survival into a later window is part of what is measured; a host that is inactive in a
window is simply not evaluable there, and the report states the coverage per window and a survivorship
diagnostic (median train activity of hosts active in every window versus the others). Not covered: a
cold-start analysis of hosts that appear only later, and any metric on vanished hosts themselves. Identity
retrieval and window-stability checks necessarily use the hosts active in both windows concerned.

**4.10 Reproducibility.** Everything is seeded (`--seed`); `experiments/test_protocol.py` runs the full
experiment twice and asserts that every JSON/CSV/Markdown artifact is identical (only the git SHA in the
environment block may differ), and that the random baseline changes with the seed while the deterministic
baselines do not. Scored embeddings are shown to be independent of the validation/test flows (test: perturb
only later windows, train embeddings of every baseline are identical). Environment versions are in
`manifest.json`; exact bit-for-bit equality across different library versions is not claimed.

**4.11 External methods.** Training provenance cannot be verified by code. A submission may ship
`provenance.json` (`{"trained_on_windows": ["train"]}`); a declaration that includes a later window is refused,
and a missing declaration is flagged as *undeclared* in the report.

**4.12 Port-free behavioural baseline and feature ablation (issue #43).** The behavioural profile has six
blocks (volume, peers, ratio, ports, proto, hourly); only `ports` is derived from port fields (`proto` is the
protocol number). `exclude_blocks=("ports",)` removes the block entirely (it is not computed; a test shuffles the
port column and the port-free features are bit-for-bit unchanged) and, because every block is standardised on its
own, the kept columns equal the full profile's. `behavioral_portfree` / `behavioral_portfree_pca` are added with
`--port-free-baselines`; `python -m experiments.feature_ablation` compares `full`, `no_ports` and every
leave-one-block-out variant on the **same** hosts, windows and frozen references (file hashes recorded), with paired
host-level intervals. What it means:
* The standard kNN reference *contains* the port block (~18% of the profile's squared weight), so the full baseline
  shares information with it by construction; a kNN gap is overlap, not representation quality. The shared-peer pair /
  triplet references and the link candidates contain no ports, so there the ablation measures what service identity
  adds to structural and predictive signal.
* A diagnostic rebuilds the kNN reference without ports (analysis only: different fingerprint, never mixed into the
  leaderboard). If the gap vanishes there, it was port overlap.
* Synthetic fixture only, where ports are informative **by construction** (role-specific ports): full vs port-free,
  kNN@10 0.280 vs 0.272 (paired difference +0.008, 95% CI [-0.000, +0.016]); pair AUC 0.907 vs 0.885 (+0.022
  [+0.006, +0.042]); cosine link AUC 0.478 vs 0.457 (+0.021 [+0.015, +0.027]); triplet 0.918 vs 0.887 (+0.031
  [0.000, +0.072]). Against the port-free reference the full baseline has no advantage (-0.004 [-0.013, +0.004]), and
  the two references' neighbours agree at Consistency@10 = 0.77 (chance 0.033). Reading: on this fixture the kNN
  proxy depends only weakly on ports (the other blocks carry most of the behaviour), while ports add measurable
  structural signal. Nothing follows for LANL, where non-well-known ports are anonymised `PortNNNN` tokens.

## 5. Results on the synthetic fixture (not LANL)

Test reference, 300 hosts, dim 16, seed 0 (full tables, CIs and plots: `demo/synthetic_run/report.md`):

| method | kNN@10 | link AUC (cosine) | link AUC (learned) | pair AUC | triplet acc. | P@3 | R@3 |
|---|---|---|---|---|---|---|---|
| random | 0.037 | 0.502 | 0.546 | 0.516 | 0.514 | 0.232 | 0.268 |
| behavioral | 0.280 | 0.478 | 0.668 | 0.907 | 0.918 | 0.591 | 0.745 |
| behavioral PCA / SVD | 0.282 | 0.478 | 0.653 | 0.908 | 0.920 | 0.599 | 0.758 |
| graph_spectral | 0.244 | 0.474 | 0.730 | 0.938 | 0.940 | 0.623 | 0.805 |
| reference_ceiling (leaks) | 1.000 | 0.487 | n/a | 0.905 | 0.914 | 0.580 | 0.720 |

Reading: random sits at chance everywhere; the reference ceiling is perfect only on the metric it is built
from; the graph embedding wins the peer-based and link metrics (expected, by construction, section 3); the
behavioural baseline matches the persistence bound on the kNN metric. PCA and SVD coincide because the
features are already centred. No conclusion about LANL follows.

## 6. Threats to validity and limitations

* Synthetic fixture only; effect sizes on real traffic are unknown.
* Provenance of real files is a declaration, not a verification (no published checksums to compare against).
* The behavioural proxy and the shared-peer judgments are *our* definitions of similarity.
* Uniform negatives make link prediction easier than a realistic hard-negative setting.
* Netflow records originator-direction counters in the features; responder counters are stored but unused.
* `Time` has no documented origin, so hour-of-day features are relative to an unknown clock origin.
* External methods' training provenance cannot be verified by code.
* Graph/behavioural baselines are fitted once; seed variance of real learned models is not covered by the CIs.
* P@k/R@k use small sampled candidate pools (k ≤ 5); τ for the pair judgments is data-adaptive (declared in the manifest).

## 7. What is needed for the real experiment

1. LANL `netflow_day-NN.bz2` files (download form; links are tokenised and expire).
2. `describe` them, declare windows in `Time` units, run `pipeline.prepare` (with `--data-kind lanl
   --source-note ...` to label the run as declared LANL; the default is `unverified`) and `pipeline.reference`.
3. Embeddings of the real methods (issue #20; node2vec, DeepWalk, GraphSAGE, struc2vec, RolX), trained on the
   train window only, one folder per method in a submissions directory; optional per-window files enable identity.
4. `python -m experiments.end_to_end --netflow ... --config ... --workdir runs/lanl --submissions-dir ...`.
5. If external host labels are ever available, pass `--labels F --labels-source "..."` to enable the probe.

Claims allowed from such a run: ranking of methods under *these* references with the listed caveats.
Not allowed: "the embedding recovers host roles" (no role ground truth), or any statement about methods that
were not evaluated.
