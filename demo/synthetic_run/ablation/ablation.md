# Port-free behavioural baseline: feature ablation

> **SYNTHETIC FIXTURE - not LANL traffic.** Synthetic data: the fixture's roles use role-specific ports BY CONSTRUCTION, so ports are informative here; nothing follows for LANL.

Identical for every variant (checked by file hash): split `test`, 300 scored hosts, the same frozen reference files, fit on the `train` window only, cohort policy `train_only`.

## Standard reference: full profile vs port-free

| metric | full | port-free | full - port-free (paired 95% CI) |
|---|---|---|---|
| kNN Consistency@10 | 0.280 | 0.272 | +0.008 [-0.000, +0.016] |
| nDCG@10 | 0.264 | 0.261 | not paired |
| Link ROC-AUC (cosine) | 0.478 | 0.457 | +0.021 [+0.015, +0.027] * |
| Link ROC-AUC (learned) | 0.668 | 0.656 | not paired |
| Pair ROC-AUC | 0.907 | 0.885 | +0.022 [+0.006, +0.042] * |
| Triplet accuracy | 0.918 | 0.887 | +0.031 [+0.000, +0.072] |

`*` the paired interval excludes 0. Intervals resample hosts jointly for both variants (host-cluster bootstrap for pair/link/triplet metrics); they cover evaluation-set sampling only.

## Leave-one-block-out (same protocol)

| variant (blocks removed) | kNN Consistency@10 | nDCG@10 | Link ROC-AUC (cosine) | Link ROC-AUC (learned) | Pair ROC-AUC | Triplet accuracy |
|---|---|---|---|---|---|---|
| full (none) | 0.280 | 0.264 | 0.478 | 0.668 | 0.907 | 0.918 |
| no_ports (ports) | 0.272 | 0.261 | 0.457 | 0.656 | 0.885 | 0.887 |
| no_volume (volume) | 0.276 | 0.263 | 0.481 | 0.666 | 0.911 | 0.923 |
| no_peers (peers) | 0.275 | 0.266 | 0.501 | 0.667 | 0.913 | 0.927 |
| no_ratio (ratio) | 0.282 | 0.266 | 0.500 | 0.665 | 0.908 | 0.923 |
| no_proto (proto) | 0.277 | 0.261 | 0.466 | 0.667 | 0.894 | 0.900 |
| no_hourly (hourly) | 0.284 | 0.270 | 0.475 | 0.666 | 0.906 | 0.917 |

## Diagnostic: reference rebuilt WITHOUT ports (analysis only)

_reference rebuilt from the same target window WITHOUT the port block; different fingerprint, NOT comparable with leaderboard scores_

| kNN Consistency@k | full | port-free | chance |
|---|---|---|---|
| @5 | 0.192 | 0.191 | 0.017 |
| @10 | 0.265 | 0.269 | 0.033 |
| @20 | 0.398 | 0.395 | 0.067 |

Paired full - port-free at k=10: -0.004 [-0.013, +0.004]. Agreement of the port-free reference's neighbours with the standard reference's: Consistency@10 = 0.770 (chance 0.033).

## Methodological implications

1. **The standard behavioural reference contains the port block** (about 18% of the profile's squared weight). The full baseline therefore shares that information with the reference **by construction**; the port-free baseline removes this overlap. A gap on the kNN metrics measures information overlap, not representation quality (here the paired kNN@10 difference is +0.008).
2. **Only the kNN reference involves ports.** The shared-peer pair/triplet references and the future-link candidates are built from communication partners, so the port ablation on those metrics measures how much *service identity* adds to structural/predictive signal, not overlap with the reference.
3. **The port-free reference diagnostic isolates what is left.** If the full-minus-port-free gap shrinks or vanishes when ports are removed from the reference too, the standard-reference gap was port overlap; if it persists, other blocks drive it. The diagnostic is not a leaderboard reference and must not be mixed with leaderboard scores.
4. **What ports are in LANL netflow.** Well-known ports are kept and others are anonymised to stable `PortNNNN` tokens, so the port block partly acts as a service identifier. A port-free profile tests reliance on service identity versus traffic shape (volume, direction, peers, protocols, time of day).
5. **The leaderboard is unchanged.** The standard references, hosts and windows are the same; the port-free variants are added rows (`--port-free-baselines`), not a new protocol. Removing a block does not rescale the others (each block is standardised on its own), so the kept columns are identical to the full profile's.
6. **Limits.** One window pair, one cohort, no host-role labels; the ablation says which information the *proxy* rewards, not which information identifies roles.
