# Benchmark run report

> **SYNTHETIC FIXTURE - not LANL traffic.**
>
> The numbers come from generated data in the LANL netflow layout. They demonstrate the pipeline and the metrics only and say nothing about real network behaviour.

* data kind: `synthetic_fixture` (basis: marker_file); cohort hosts: **300**; dim=16, seed=0, bootstrap=200; targets: validation, test
* methods evaluated: behavioral, behavioral_pca, behavioral_svd, deepwalk, graph_spectral, graphsage, node2vec, random, reference_ceiling, rolx, struc2vec
* **not available in this run** (no embeddings supplied, nothing simulated): none
* cohort policy `train_only`; evaluable hosts per window: validation: 300/300 (100%), test: 300/300 (100%)
  * The cohort is defined from the TRAIN window only. Hosts that are inactive in a later window are not evaluable there; every reference is built on the evaluable hosts and the coverage is reported. Survivorship of hosts into later windows is therefore part of the measured outcome, not of the cohort definition.


## Scores - validation reference

| method | consistency@10_mean | ndcg@10_mean | link_roc_auc | link_learned_roc_auc | link_new_roc_auc | partner_new_mrr | pair_roc_auc | triplet_accuracy | precision@3 | recall@3 |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 0.032 | 0.026 | 0.501 | n/a | 0.509 | 0.144 | 0.481 | 0.528 | 0.331 | 0.540 |
| behavioral | 0.284 | 0.263 | 0.477 | n/a | 0.480 | 0.159 | 0.893 | 0.896 | 0.529 | 0.875 |
| behavioral_pca | 0.279 | 0.265 | 0.477 | n/a | 0.480 | 0.158 | 0.897 | 0.896 | 0.531 | 0.877 |
| behavioral_svd | 0.279 | 0.265 | 0.477 | n/a | 0.480 | 0.158 | 0.897 | 0.896 | 0.531 | 0.877 |
| graph_spectral | 0.253 | 0.237 | 0.479 | n/a | 0.484 | 0.155 | 0.945 | 0.937 | 0.553 | 0.911 |
| deepwalk | 0.197 | 0.194 | 0.659 | n/a | 0.504 | 0.153 | 0.886 | 0.840 | 0.509 | 0.845 |
| graphsage | 0.245 | 0.237 | 0.574 | n/a | 0.491 | 0.157 | 0.883 | 0.874 | 0.519 | 0.851 |
| node2vec | 0.210 | 0.206 | 0.618 | n/a | 0.490 | 0.158 | 0.905 | 0.877 | 0.537 | 0.890 |
| rolx | 0.078 | 0.075 | 0.501 | n/a | 0.505 | 0.154 | 0.636 | 0.647 | 0.387 | 0.624 |
| struc2vec | 0.080 | 0.075 | 0.492 | n/a | 0.494 | 0.144 | 0.644 | 0.584 | 0.381 | 0.620 |
| reference_ceiling | 1.000 | 1.000 | 0.476 | n/a | 0.485 | 0.160 | 0.887 | 0.874 | 0.533 | 0.892 |

Link-prediction reference point (no embedding): *pair already seen in the source window* scores ROC-AUC **0.710**. `link_learned_*` fits an edge classifier on the validation candidates and scores the test candidates; cosine is shown for completeness.

![scorecard](figures/scorecard_validation.png)


## Scores - test reference

| method | consistency@10_mean | ndcg@10_mean | link_roc_auc | link_learned_roc_auc | link_new_roc_auc | partner_new_mrr | pair_roc_auc | triplet_accuracy | precision@3 | recall@3 |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 0.037 | 0.035 | 0.502 | 0.546 | 0.509 | 0.164 | 0.516 | 0.514 | 0.232 | 0.268 |
| behavioral | 0.280 | 0.264 | 0.478 | 0.668 | 0.487 | 0.169 | 0.907 | 0.918 | 0.591 | 0.745 |
| behavioral_pca | 0.282 | 0.260 | 0.478 | 0.653 | 0.487 | 0.165 | 0.908 | 0.920 | 0.599 | 0.758 |
| behavioral_svd | 0.282 | 0.260 | 0.478 | 0.653 | 0.487 | 0.165 | 0.908 | 0.920 | 0.599 | 0.758 |
| graph_spectral | 0.244 | 0.234 | 0.474 | 0.730 | 0.479 | 0.174 | 0.938 | 0.940 | 0.623 | 0.805 |
| deepwalk | 0.187 | 0.193 | 0.652 | 0.660 | 0.492 | 0.170 | 0.887 | 0.884 | 0.553 | 0.704 |
| graphsage | 0.245 | 0.239 | 0.568 | 0.640 | 0.480 | 0.176 | 0.890 | 0.912 | 0.573 | 0.722 |
| node2vec | 0.206 | 0.203 | 0.619 | 0.659 | 0.490 | 0.179 | 0.900 | 0.897 | 0.572 | 0.729 |
| rolx | 0.081 | 0.078 | 0.497 | 0.605 | 0.501 | 0.146 | 0.616 | 0.657 | 0.292 | 0.336 |
| struc2vec | 0.089 | 0.081 | 0.493 | 0.621 | 0.499 | 0.150 | 0.607 | 0.605 | 0.270 | 0.263 |
| reference_ceiling | 1.000 | 1.000 | 0.487 | n/a | 0.500 | 0.176 | 0.905 | 0.914 | 0.580 | 0.720 |

Link-prediction reference point (no embedding): *pair already seen in the source window* scores ROC-AUC **0.708**. `link_learned_*` fits an edge classifier on the validation candidates and scores the test candidates; cosine is shown for completeness.

![scorecard](figures/scorecard_test.png)


## Metrics not available

* linear_probe: нет labels.csv (внешних меток хостов нет, придумывать их нельзя)

## 95% bootstrap intervals (test)

| method | metric | estimate | ci_low | ci_high |
|---|---|---|---|---|
| random | consistency@10_mean | 0.037 | 0.030 | 0.044 |
| random | link_roc_auc | 0.502 | 0.482 | 0.523 |
| random | pair_roc_auc | 0.516 | 0.463 | 0.576 |
| random | triplet_accuracy | 0.514 | 0.410 | 0.606 |
| behavioral | consistency@10_mean | 0.280 | 0.256 | 0.304 |
| behavioral | link_roc_auc | 0.478 | 0.451 | 0.506 |
| behavioral | pair_roc_auc | 0.907 | 0.866 | 0.944 |
| behavioral | triplet_accuracy | 0.918 | 0.851 | 0.967 |
| behavioral_pca | consistency@10_mean | 0.282 | 0.260 | 0.313 |
| behavioral_pca | link_roc_auc | 0.478 | 0.450 | 0.505 |
| behavioral_pca | pair_roc_auc | 0.908 | 0.866 | 0.945 |
| behavioral_pca | triplet_accuracy | 0.920 | 0.857 | 0.968 |
| behavioral_svd | consistency@10_mean | 0.282 | 0.260 | 0.313 |
| behavioral_svd | link_roc_auc | 0.478 | 0.450 | 0.505 |
| behavioral_svd | pair_roc_auc | 0.908 | 0.866 | 0.945 |
| behavioral_svd | triplet_accuracy | 0.920 | 0.857 | 0.968 |
| graph_spectral | consistency@10_mean | 0.244 | 0.225 | 0.266 |
| graph_spectral | link_roc_auc | 0.474 | 0.448 | 0.498 |
| graph_spectral | pair_roc_auc | 0.938 | 0.894 | 0.966 |
| graph_spectral | triplet_accuracy | 0.940 | 0.879 | 0.980 |
| deepwalk | consistency@10_mean | 0.187 | 0.171 | 0.207 |
| deepwalk | link_roc_auc | 0.652 | 0.630 | 0.677 |
| deepwalk | pair_roc_auc | 0.887 | 0.846 | 0.925 |
| deepwalk | triplet_accuracy | 0.884 | 0.805 | 0.946 |
| graphsage | consistency@10_mean | 0.245 | 0.222 | 0.272 |
| graphsage | link_roc_auc | 0.568 | 0.542 | 0.598 |
| graphsage | pair_roc_auc | 0.890 | 0.845 | 0.930 |
| graphsage | triplet_accuracy | 0.912 | 0.853 | 0.959 |
| node2vec | consistency@10_mean | 0.206 | 0.191 | 0.225 |
| node2vec | link_roc_auc | 0.619 | 0.597 | 0.642 |
| node2vec | pair_roc_auc | 0.900 | 0.857 | 0.933 |
| node2vec | triplet_accuracy | 0.897 | 0.813 | 0.957 |
| rolx | consistency@10_mean | 0.081 | 0.069 | 0.096 |
| rolx | link_roc_auc | 0.497 | 0.472 | 0.520 |
| rolx | pair_roc_auc | 0.616 | 0.555 | 0.682 |
| rolx | triplet_accuracy | 0.657 | 0.540 | 0.766 |
| struc2vec | consistency@10_mean | 0.089 | 0.077 | 0.106 |
| struc2vec | link_roc_auc | 0.493 | 0.470 | 0.518 |
| struc2vec | pair_roc_auc | 0.607 | 0.541 | 0.665 |
| struc2vec | triplet_accuracy | 0.605 | 0.500 | 0.717 |
| reference_ceiling | consistency@10_mean | 1.000 | 1.000 | 1.000 |
| reference_ceiling | link_roc_auc | 0.487 | 0.462 | 0.512 |
| reference_ceiling | pair_roc_auc | 0.905 | 0.859 | 0.936 |
| reference_ceiling | triplet_accuracy | 0.914 | 0.852 | 0.966 |

## Identity retrieval (train -> test)

| method | shared coords declared | Top-1 raw | Top-1 Procrustes | MRR Procrustes | pair ROC-AUC Procrustes | chance Top-1 |
|---|---|---|---|---|---|---|
| random | False | 0.000 | 0.007 | 0.035 | 0.523 | 0.007 |
| behavioral | True | 0.047 | 0.133 | 0.262 | 0.911 | 0.007 |
| behavioral_pca | True | 0.053 | 0.107 | 0.252 | 0.892 | 0.007 |
| behavioral_svd | True | 0.053 | 0.107 | 0.252 | 0.892 | 0.007 |
| graph_spectral | False | 0.000 | 0.180 | 0.324 | 0.936 | 0.007 |
| deepwalk | False | 0.027 | 0.100 | 0.251 | 0.909 | 0.007 |
| graphsage | True | 0.027 | 0.080 | 0.206 | 0.911 | 0.007 |
| node2vec | False | 0.093 | 0.140 | 0.285 | 0.899 | 0.007 |
| rolx | False | 0.013 | 0.027 | 0.079 | 0.648 | 0.007 |
| struc2vec | False | 0.020 | 0.020 | 0.064 | 0.570 | 0.007 |

![identity](figures/identity_test.png)

Raw-coordinate scores are only interpretable for methods that declare a shared coordinate system.

## Validity audit

* **Weight sensitivity** (12 perturbed behavioural references, k=10): neighbour agreement with the canonical reference 0.74 on average (min 0.58; chance 0.033); method-ranking Kendall tau mean 0.92, min 0.86.
* **Reference stability** train->validation: Consistency@10 = 0.28 (chance 0.033) - behavioural neighbourhoods drift between windows.
* **Reference stability** train->test: Consistency@10 = 0.28 (chance 0.033) - behavioural neighbourhoods drift between windows.
* **Reference stability** validation->test: Consistency@10 = 0.33 (chance 0.033) - behavioural neighbourhoods drift between windows.
* **Validation vs test ranking agreement** (Kendall tau): consistency@10_mean: 0.89, link_roc_auc: 0.70, pair_roc_auc: 0.78, triplet_accuracy: 0.90
* **Chance band (random embeddings, test reference)**: consistency@10_mean 0.034 ± 0.003, link_roc_auc 0.501 ± 0.007, link_learned_roc_auc 0.543 ± 0.007, pair_roc_auc 0.501 ± 0.021, triplet_accuracy 0.502 ± 0.023
* **Link-prediction prevalence**: sampled positive rate 0.50 vs natural 0.083; 51% of positive links were already present in the source window (compare 'New-link ROC-AUC').

![weights](figures/weight_sensitivity.png)


### Information shared by construction (method input vs reference source)

| method | neighbors | edges | pairs |
|---|---|---|---|
| random | none | none | none |
| behavioral | same feature definitions, earlier window (feature overlap) | none | weak: behaviour correlates with peers |
| behavioral_pca | same feature definitions, earlier window (feature overlap) | none | weak: behaviour correlates with peers |
| behavioral_svd | same feature definitions, earlier window (feature overlap) | none | weak: behaviour correlates with peers |
| behavioral_portfree | same feature definitions MINUS the port block, earlier window (reduced feature overlap) | none | weak: behaviour correlates with peers |
| behavioral_portfree_pca | same feature definitions MINUS the port block, earlier window (reduced feature overlap) | none | weak: behaviour correlates with peers |
| graph_spectral | weak: degree/volume leaks into the spectrum | strong: link persistence (peers seen in train recur) | strong: peer sets persist across windows |
| reference_ceiling | identical by construction (target-window features) | none | weak |
| deepwalk | weak: peers/degree structure correlates with behaviour | strong: link persistence (peers seen in train recur) | strong: shared peers = shared neighbourhood |
| graphsage | strong with features='behavioral': same feature definitions as the behavioural reference | strong: link persistence through the neighbourhood aggregation | strong: shared peers = shared neighbourhood |
| node2vec | weak: peers/degree structure correlates with behaviour | strong: link persistence (peers seen in train recur) | strong: shared peers = shared neighbourhood |
| rolx | weak: degree/volume-like structure correlates with behaviour | weak: structural roles, not specific peers, so link persistence is not encoded | weak: shared peers are not encoded (roles ignore identity) |
| struc2vec | weak: degree structure correlates with behaviour | weak: structural identity, not specific peers, so link persistence is not encoded | weak: shared peers are not encoded (structure ignores identity) |

## Reading these numbers

* **Predictive:** future links (outcome observed later) and identity stability. **Agreement with a proxy we chose:** kNN consistency (behavioural profile), pair/triplet/P@k (shared peers).
* `reference_ceiling` uses the target-window features themselves and leaks by design; baselines share feature definitions or peer structure with the references (table above), so they are calibration points, not independent evidence.
* Intervals: kNN and retrieval metrics resample hosts/anchors; link, pair and triplet metrics use a host-cluster bootstrap (pairs share hosts). They cover evaluation-set sampling only, not model seeds, reference choice or window choice.
* Because the data are synthetic, no statement about LANL follows from any number above.
