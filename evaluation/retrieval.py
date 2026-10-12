"""Precision@k / Recall@k over JUDGED candidates (reference/pairs.csv).

Relevance judgments are the rows of pairs.csv of the evaluated split (default: test):
label 1 = relevant, label 0 = not relevant. A host pair that is NOT in pairs.csv is
UNJUDGED - it is never counted as non-relevant.

Condensed-list protocol (standard for incomplete judgments):
  for every anchor host a, the candidate set C(a) is the set of hosts b for which {a, b}
  is judged. Rank C(a) by cosine similarity to a (ties: host_id ascending), then

      P@k(a) = hits_k(a) / k            R@k(a) = hits_k(a) / R(a)
      R(a)   = number of relevant candidates of a

  An anchor is ELIGIBLE for a given k iff |C(a)| >= k and R(a) >= 1; other anchors are
  excluded and counted (eligible_anchors / judged_anchors = coverage).
  Reported means are over eligible anchors. Because candidate pools are small and sampled,
  also reported for the SAME anchors:
      oracle_precision@k  = mean(min(R, k) / k)       (best achievable)
      chance_precision@k  = mean(R / |C|)             (expected under random ranking)
  and the lift P@k / chance_precision@k. Compare methods by lift/at equal k and pool.

Interpretation: ranks a fixed, sampled pool - NOT retrieval among all hosts. The meaning of
"relevant" is defined by the reference manifest (shared-peer structure for generated pairs).

Specification: PR #38 (kalivixZ) architecture, extended with the incomplete-judgment rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.io import ReferenceError, cosine_pairs, read_reference_table, select_split
from evaluation.uncertainty import bootstrap_ci, reweighted_average_precision, auc_with_ci

SPLITS = {"train", "validation", "test"}


def load_pairs(path) -> pd.DataFrame:
    p = read_reference_table(path, {"host_a", "host_b", "label", "split"}, split_values=SPLITS)
    if not p["label"].isin(["0", "1"]).all():
        raise ReferenceError(f"{path}: label must be 0 or 1")
    if (p["host_a"] == p["host_b"]).any():
        raise ReferenceError(f"{path}: a host cannot be paired with itself")
    key = p.apply(lambda r: tuple(sorted((r.host_a, r.host_b))) + (r.split,), axis=1)
    if key.duplicated().any():
        raise ReferenceError(f"{path}: duplicate (unordered) pair within a split")
    return p


def _scored(emb: pd.DataFrame, pairs: pd.DataFrame, split: str) -> pd.DataFrame:
    p = select_split(pairs, split, "pairs.csv").copy()
    p["score"] = cosine_pairs(emb, p.host_a.tolist(), p.host_b.tolist())
    p["label"] = p["label"].astype(int)
    return p


def pair_auc(emb: pd.DataFrame, pairs: pd.DataFrame, split: str = "test", *,
             natural_prevalence: float | None = None, n_boot: int = 0, seed: int = 0) -> dict:
    """ROC-AUC / PR-AUC of cosine similarity on judged pairs of one split."""
    p = _scored(emb, pairs, split)
    y, s = p["label"].to_numpy(), p["score"].to_numpy()
    if len(np.unique(y)) < 2:
        raise ReferenceError(f"pairs.csv split={split!r} needs both relevant and non-relevant pairs")
    out = auc_with_ci(y, s, n_boot=n_boot, seed=seed)
    prev = float(y.mean())
    out.update(n_pairs=int(len(y)), n_positive=int(y.sum()), positive_rate=prev,
               pr_auc_lift=out["pr_auc"] / prev, split=split)
    if natural_prevalence is not None:
        out["pr_auc_at_natural_prevalence"] = reweighted_average_precision(y, s, natural_prevalence)
        out["natural_prevalence"] = float(natural_prevalence)
    return out


def precision_recall_at_k(emb: pd.DataFrame, pairs: pd.DataFrame, ks=(1, 3, 5), split: str = "test",
                          *, n_boot: int = 0, seed: int = 0) -> dict:
    p = _scored(emb, pairs, split)
    long = pd.concat([
        p.rename(columns={"host_a": "anchor", "host_b": "cand"})[["anchor", "cand", "label", "score"]],
        p.rename(columns={"host_b": "anchor", "host_a": "cand"})[["anchor", "cand", "label", "score"]],
    ])
    groups = []
    for anchor, g in long.groupby("anchor", sort=True):
        g = g.sort_values(["score", "cand"], ascending=[False, True], kind="stable")
        groups.append((anchor, g["label"].to_numpy()))
    n_judged = len(groups)
    result: dict = {"split": split, "judged_anchors": n_judged, "ks": list(ks), "by_k": {}}
    for k in ks:
        if k < 1:
            raise ValueError("k must be >= 1")
        elig = [(lab[:k].sum(), lab.sum(), len(lab)) for _, lab in groups
                if len(lab) >= k and lab.sum() >= 1]
        if not elig:
            result["by_k"][str(k)] = {"eligible_anchors": 0, "coverage": 0.0}
            continue
        hits, rel, size = (np.array(x, dtype=float) for x in zip(*elig))
        prec, rec = hits / k, hits / rel
        chance = rel / size
        entry = {
            "precision": float(prec.mean()), "recall": float(rec.mean()),
            "oracle_precision": float((np.minimum(rel, k) / k).mean()),
            "chance_precision": float(chance.mean()),
            "precision_lift": float(prec.mean() / chance.mean()),
            "eligible_anchors": len(elig), "coverage": len(elig) / n_judged,
        }
        entry["precision_ci"] = bootstrap_ci(lambda i: float(prec[i].mean()), len(prec), n_boot=n_boot, seed=seed)
        entry["recall_ci"] = bootstrap_ci(lambda i: float(rec[i].mean()), len(rec), n_boot=n_boot, seed=seed)
        result["by_k"][str(k)] = entry
    return result
