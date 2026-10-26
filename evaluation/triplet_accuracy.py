"""Точность по тройкам (reference/triplets.csv).

    TA = (1/N) * sum_i [ 1{ s(a_i, p_i) > s(a_i, n_i) } + 0.5 * 1{ s(a_i, p_i) = s(a_i, n_i) } ]

s — косинус, N — число троек нужного split (по умолчанию test). Случайный уровень — 0.5.
anchor, positive, negative — три разных хоста; каждый должен быть в эмбеддинге (иначе это другая
когорта — ошибка; с allow_missing=True такие тройки исключаются и считаются). Сопоставление по ID.

Первая версия — PR #36 (Тихон); загрузка, проверки и тесты переписаны.
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
            raise ReferenceError(f"{path}: {a} и {b} должны быть разными хостами")
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
        raise ReferenceError(f"в {n_missing} тройках есть хосты, которых нет в эмбеддинге: {hosts[:5]}")
    t = t[ok]
    if t.empty:
        raise ReferenceError("после исключения отсутствующих хостов троек не осталось")
    sp = cosine_pairs(emb, t.anchor.tolist(), t.positive.tolist())
    sn = cosine_pairs(emb, t.anchor.tolist(), t.negative.tolist())
    tie = np.isclose(sp, sn, rtol=0, atol=1e-12)                 # сначала ничья, иначе победа или поражение:
    score = np.where(tie, 0.5, (sp > sn).astype(float))         # почти равные не получают 1 + 0.5
    out = {
        "accuracy": float(score.mean()),
        "n_triplets": int(len(t)),
        "n_ties": int(tie.sum()),
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
