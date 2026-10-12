"""Triplet Accuracy against frozen triplets (reference/triplets.csv).

Definition (cosine similarity s, tie = 0.5):

    TA = (1/N) * sum_i [ 1{ s(a_i, p_i) > s(a_i, n_i) } + 0.5 * 1{ s(a_i, p_i) = s(a_i, n_i) } ]

N = number of triplets of the evaluated split (default: test). Chance level is 0.5.

Inputs: embeddings (host_id,emb_*) and triplets.csv (anchor,positive,negative,split).
Rules: anchor/positive/negative must be three distinct hosts; every host must exist in the
embedding (a missing host is an error - it means a different cohort - unless
allow_missing=True, in which case those triplets are excluded and COUNTED); hosts match by
ID, never by row position; blank/unknown-split rows are rejected.

Interpretation: agreement with how the reference judged the triplet. What a triplet means
(behavioural, structural, role) is defined by the reference manifest, not by this module.

Original implementation: PR #36 (Tikhon / khevikos); loader, validation and tests reworked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.io import ReferenceError, cosine_pairs, read_reference_table, select_split
from evaluation.uncertainty import dyadic_bootstrap_ci, host_index

SPLITS = {"train", "validation", "test"}


def load_triplets(path) -> pd.DataFrame:
    t = read_reference_table(path, {"anchor", "positive", "negative", "split"}, split_values=SPLITS)
    for a, b in (("anchor", "positive"), ("anchor", "negative"), ("positive", "negative")):
        if (t[a] == t[b]).any():
            raise ReferenceError(f"{path}: {a} and {b} must be different hosts")
    return t


def triplet_accuracy(emb: pd.DataFrame, triplets: pd.DataFrame, split: str = "test", *,
                     allow_missing: bool = False, n_boot: int = 0, seed: int = 0) -> dict:
    t = select_split(triplets, split, "triplets.csv")
    n_file = len(t)
    known = set(emb.index)
    ok = t["anchor"].isin(known) & t["positive"].isin(known) & t["negative"].isin(known)
    n_missing = int((~ok).sum())
    if n_missing and not allow_missing:
        hosts = sorted((set(t.anchor) | set(t.positive) | set(t.negative)) - known)
        raise ReferenceError(f"{n_missing} triplets use hosts absent from the embedding: {hosts[:5]}")
    t = t[ok]
    if t.empty:
        raise ReferenceError("no usable triplets after excluding missing hosts")
    sp = cosine_pairs(emb, t.anchor.tolist(), t.positive.tolist())
    sn = cosine_pairs(emb, t.anchor.tolist(), t.negative.tolist())
    score = (sp > sn).astype(float) + 0.5 * np.isclose(sp, sn, rtol=0, atol=1e-12)
    out = {
        "accuracy": float(score.mean()),
        "n_triplets": int(len(t)),
        "n_ties": int(np.isclose(sp, sn, rtol=0, atol=1e-12).sum()),
        "n_excluded_missing_hosts": n_missing,
        "n_in_file_for_split": n_file,
        "split": split,
        "chance": 0.5,
    }
    cols, n = host_index(t.anchor, t.positive, t.negative)
    out["accuracy_ci"] = dyadic_bootstrap_ci(lambda w: float(np.average(score, weights=w)), cols, n,
                                             n_boot=n_boot, seed=seed)
    out["ci_method"] = "host-cluster bootstrap (hosts resampled; triplets weighted by endpoint multiplicity)"
    return out
