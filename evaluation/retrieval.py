"""Precision@k / Recall@k среди РАЗМЕЧЕННЫХ кандидатов (reference/pairs.csv).

Разметка — строки pairs.csv нужного split: 1 — похожи, 0 — не похожи. Пары, которой нет в файле,
НЕ размечена и никогда не считается «непохожей».

Для каждого хоста a кандидаты C(a) — хосты, пара с которыми размечена. Ранжируем C(a) по косинусу
(при равенстве — по host_id):
    P@k(a) = hits_k(a) / k,   R@k(a) = hits_k(a) / R(a),   R(a) — число похожих среди C(a)
Хост участвует для данного k, если |C(a)| >= k и R(a) >= 1; остальные исключаются и считаются (coverage).
Рядом на тех же хостах: лучшее возможное min(R, k)/k, случайный уровень R/|C| и lift = P@k / случайный.

Это ранжирование маленького выборочного набора, а не поиск среди всех хостов.
Основа — архитектура из PR #38 (Даня) + правило про неразмеченные пары.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.io import ReferenceError, cosine_pairs, read_reference_table, select_split
from evaluation.uncertainty import (auc_with_ci, bootstrap_ci, host_index, reweighted_average_precision)

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
             judged_prevalence: float | None = None, n_boot: int = 0, seed: int = 0) -> dict:
    """ROC-AUC / PR-AUC косинуса на размеченных парах одного split.

    judged_prevalence — доля похожих среди ВСЕХ размеченных пар (неразмеченные 0 < J < tau исключены).
    PR-AUC, пересчитанный к ней (pr_auc_at_judged_prevalence), — условная величина: только по парам,
    которые правило умеет разметить. Это не PR-AUC по всем парам хостов.
    """
    p = _scored(emb, pairs, split)
    y, s = p["label"].to_numpy(), p["score"].to_numpy()
    if len(np.unique(y)) < 2:
        raise ReferenceError(f"pairs.csv split={split!r} needs both relevant and non-relevant pairs")
    out = auc_with_ci(y, s, n_boot=n_boot, seed=seed, hosts=host_index(p.host_a, p.host_b))
    prev = float(y.mean())
    out.update(n_pairs=int(len(y)), n_positive=int(y.sum()), positive_rate=prev,
               pr_auc_lift=out["pr_auc"] / prev, split=split)
    if judged_prevalence is not None:
        out["pr_auc_at_judged_prevalence"] = reweighted_average_precision(y, s, judged_prevalence)
        out["judged_population_prevalence"] = float(judged_prevalence)
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
