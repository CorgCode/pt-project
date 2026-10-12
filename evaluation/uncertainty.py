"""Bootstrap uncertainty and prevalence handling for sampled evaluations."""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def bootstrap_ci(stat: Callable[[np.ndarray], float], n_items: int, *, n_boot: int = 200,
                 seed: int = 0, level: float = 0.95) -> tuple[float, float] | None:
    """Percentile CI of stat(index_array) over resampled items (rows/hosts/pairs).

    Resampling units are the evaluation items; it captures sampling variability of the
    fixed evaluation set, NOT variability from model training seeds or reference choice.
    Returns None if n_boot == 0 or no valid replicate exists.
    """
    if n_boot <= 0 or n_items < 2:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        try:
            v = stat(rng.integers(0, n_items, n_items))
        except ValueError:                       # e.g. a replicate with a single class
            continue
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < max(10, n_boot // 2):
        return None
    lo, hi = np.quantile(vals, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def auc_with_ci(labels: np.ndarray, scores: np.ndarray, *, n_boot: int = 200, seed: int = 0) -> dict:
    labels = np.asarray(labels).astype(int)
    out = {"roc_auc": float(roc_auc_score(labels, scores)),
           "pr_auc": float(average_precision_score(labels, scores))}
    out["roc_auc_ci"] = bootstrap_ci(lambda i: roc_auc_score(labels[i], scores[i]), len(labels),
                                     n_boot=n_boot, seed=seed)
    out["pr_auc_ci"] = bootstrap_ci(lambda i: average_precision_score(labels[i], scores[i]),
                                    len(labels), n_boot=n_boot, seed=seed)
    return out


def reweighted_average_precision(labels: np.ndarray, scores: np.ndarray,
                                 target_prevalence: float) -> float:
    """AP as if the candidate pool had positive rate `target_prevalence`.

    Negatives are up-weighted so that  sum(w_neg)/sum(w_pos) = (1-pi)/pi. This corrects the
    class prior of a sampled candidate set (e.g. 1:1 sampling) so PR-AUC is comparable with
    a natural, much lower positive rate. It does not remove sampling bias in WHICH negatives
    were drawn (uniform random pairs are easier than hard negatives).
    """
    labels = np.asarray(labels).astype(int)
    if not 0 < target_prevalence < 1:
        raise ValueError("target_prevalence must be in (0, 1)")
    n_pos, n_neg = int(labels.sum()), int(len(labels) - labels.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need both classes")
    w_neg = ((1 - target_prevalence) / target_prevalence) * (n_pos / n_neg)
    weights = np.where(labels == 1, 1.0, w_neg)
    return float(average_precision_score(labels, scores, sample_weight=weights))
