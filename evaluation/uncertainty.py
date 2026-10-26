"""Бутстреп-интервалы и поправка на долю положительных в выборке."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def bootstrap_ci(stat: Callable[[np.ndarray], float], n_items: int, *, n_boot: int = 200,
                 seed: int = 0, level: float = 0.95) -> tuple[float, float] | None:
    """Процентильный интервал stat(индексы) по пересэмплированным элементам (строкам, хостам, парам).

    Описывает только случайность выбора оцениваемых элементов, а не разброс по seed модели или выбору эталона.
    None, если n_boot == 0 или не получилось ни одного валидного повтора.
    """
    if n_boot <= 0 or n_items < 2:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        try:
            v = stat(rng.integers(0, n_items, n_items))
        except ValueError:                       # например, в повторе остался один класс
            continue
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < max(10, n_boot // 2):
        return None
    lo, hi = np.quantile(vals, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def host_index(*columns) -> tuple[list[np.ndarray], int]:
    """Целые индексы хостов для параллельных колонок ID (один ID — одно число)."""
    flat = pd.Index(pd.unique(np.concatenate([np.asarray(c, dtype=object) for c in columns])))
    return [flat.get_indexer(np.asarray(c, dtype=object)) for c in columns], len(flat)


def dyadic_bootstrap_ci(stat: Callable[[np.ndarray], float], host_cols: list[np.ndarray], n_hosts: int, *,
                        n_boot: int = 200, seed: int = 0, level: float = 0.95) -> tuple[float, float] | None:
    """Бутстреп по хостам для метрик на парах и тройках, которые делят хостов.

    Пары не независимы (хост встречается во многих), и пересэмплирование пар занижает разброс.
    Здесь пересэмплируются ХОСТЫ; элемент с хостами (a, b[, c]) получает вес count[a] * count[b] [* count[c]].
    stat(weights) должен считать взвешенную метрику.
    """
    if n_boot <= 0 or n_hosts < 3:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, n_hosts, n_hosts), minlength=n_hosts).astype(float)
        w = np.ones(len(host_cols[0]))
        for col in host_cols:
            w = w * counts[col]
        try:
            v = stat(w)
        except ValueError:                       # в повторе пропал один из классов
            continue
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < max(10, n_boot // 2):
        return None
    lo, hi = np.quantile(vals, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


class RankedScores:
    """ROC-AUC и PR-AUC (AP) с весами без повторной сортировки: порядок считается один раз.

    Совпадает с sklearn (roc_auc_score / average_precision_score с sample_weight, включая равные скоры),
    но для бутстрепа в сотни раз быстрее. ValueError, если у одного из классов нулевой вес — как у sklearn.
    """

    def __init__(self, labels: np.ndarray, scores: np.ndarray):
        scores = np.asarray(scores, dtype=float)
        self.order = np.argsort(-scores, kind="mergesort")
        self.y = np.asarray(labels).astype(float)[self.order]
        s = scores[self.order]
        self.ends = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]     # последний элемент каждой группы равных

    def _curve(self, w):
        w = np.ones(len(self.y)) if w is None else np.asarray(w, dtype=float)[self.order]
        tps = np.cumsum(w * self.y)[self.ends]
        fps = np.cumsum(w * (1.0 - self.y))[self.ends]
        if tps[-1] <= 0 or fps[-1] <= 0:
            raise ValueError("нужны оба класса с положительным весом")
        return tps, fps

    def roc_auc(self, w: np.ndarray | None = None) -> float:
        tps, fps = self._curve(w)
        tpr, fpr = np.r_[0.0, tps] / tps[-1], np.r_[0.0, fps] / fps[-1]
        return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))

    def average_precision(self, w: np.ndarray | None = None) -> float:
        tps, fps = self._curve(w)
        precision = tps / np.maximum(tps + fps, 1e-300)
        return float(np.sum(np.diff(np.r_[0.0, tps]) / tps[-1] * precision))


def auc_with_ci(labels: np.ndarray, scores: np.ndarray, *, n_boot: int = 200, seed: int = 0,
                hosts: tuple[list[np.ndarray], int] | None = None) -> dict:
    """ROC-AUC / PR-AUC с бутстреп-интервалами.

    hosts=(колонки индексов, n_hosts) — бутстреп по хостам (пары делят хостов); без него элементы
    пересэмплируются независимо (корректно, только если они правда независимы).
    """
    labels = np.asarray(labels).astype(int)
    out = {"roc_auc": float(roc_auc_score(labels, scores)),
           "pr_auc": float(average_precision_score(labels, scores))}
    ranked = RankedScores(labels, scores)
    if hosts is None:                            # повтор с индексами i = веса «сколько раз взят элемент»
        n = len(labels)
        out["roc_auc_ci"] = bootstrap_ci(lambda i: ranked.roc_auc(np.bincount(i, minlength=n)), n,
                                         n_boot=n_boot, seed=seed)
        out["pr_auc_ci"] = bootstrap_ci(lambda i: ranked.average_precision(np.bincount(i, minlength=n)), n,
                                        n_boot=n_boot, seed=seed)
        out["ci_method"] = "item bootstrap (items treated as independent)"
    else:
        cols, n = hosts
        out["roc_auc_ci"] = dyadic_bootstrap_ci(ranked.roc_auc, cols, n, n_boot=n_boot, seed=seed)
        out["pr_auc_ci"] = dyadic_bootstrap_ci(ranked.average_precision, cols, n, n_boot=n_boot, seed=seed)
        out["ci_method"] = "host-cluster bootstrap (hosts resampled; pairs weighted by endpoint multiplicity)"
    return out


def reweighted_average_precision(labels: np.ndarray, scores: np.ndarray,
                                 target_prevalence: float) -> float:
    """PR-AUC (AP) так, будто доля положительных среди кандидатов равна target_prevalence.

    Отрицательные получают вес, чтобы sum(w_neg)/sum(w_pos) = (1-pi)/pi. Так PR-AUC на выборке 1:1
    сравним с настоящей, гораздо меньшей долей. Сложность самих отрицательных (случайные пары проще
    «трудных») это не исправляет.
    """
    labels = np.asarray(labels).astype(int)
    if not 0 < target_prevalence < 1:
        raise ValueError("target_prevalence должен быть в (0, 1)")
    n_pos, n_neg = int(labels.sum()), int(len(labels) - labels.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("нужны оба класса")
    w_neg = ((1 - target_prevalence) / target_prevalence) * (n_pos / n_neg)
    weights = np.where(labels == 1, 1.0, w_neg)
    return float(average_precision_score(labels, scores, sample_weight=weights))
