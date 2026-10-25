"""Метрики Тихона (#40, PR #65): прогноз собеседников по эмбеддингу, как в его коде.

Для каждого хоста A ранжируются кандидаты — все, с кем оцениваемые хосты общались в train-окне
и у кого есть эмбеддинг (сам A исключается). Проверка — по реальным собеседникам A в test-окне.

  Идея 1, классический прогноз (metric1.py): скор кандидата — косинус между A и кандидатом.
    Обратный вариант (−косинус) — контроль: если он лучше прямого, близкие векторы реже общаются.
  Идея 3, перенос окрестности (metric3.py): k = 10 ближайших к A по эмбеддингу (вес — косинус > 0).
    direct — скор кандидата = сумма «вес соседа × число контактов соседа с кандидатом» в train;
    soft   — центроид векторов этих собеседников (с теми же весами), скор — косинус с центроидом.
  Идея 2 (устойчивость к удалению рёбер) требует переобучать метод — она в experiments.robustness.

Метрики — MRR (1 / место первого настоящего собеседника) и Recall@10, среднее по хостам. На тех же
кандидатах — бейзлайны без эмбеддинга: history (с кем A общался раньше), popularity, random.
Отличие от его кода: при равных скорах берётся средний ранг, а не алфавитный порядок.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from evaluation.partners import _per_host
from evaluation.uncertainty import bootstrap_ci

VARIANTS = ("lp_direct", "lp_inverted", "transfer_direct", "transfer_soft", "history", "popularity", "random")


def tikhon_metrics(emb: pd.DataFrame, partners: pd.DataFrame, *, k: int = 10, top_n: int = 10,
                   n_boot: int = 0, seed: int = 0, batch: int = 512) -> dict:
    hosts = pd.Index(emb.index)
    src = partners[(partners.window == "source") & partners.host_id.isin(hosts)]
    tgt = partners[(partners.window == "target") & partners.host_id.isin(hosts)]
    universe = pd.Index(sorted(set(src.partner) & set(hosts)))      # кандидаты с эмбеддингом
    if len(universe) < 2:
        raise ValueError("в train-окне меньше двух собеседников с эмбеддингом")
    s = src[src.partner.isin(universe)]
    M = sparse.csr_matrix((s["count"].to_numpy(float), (hosts.get_indexer(s.host_id), universe.get_indexer(s.partner))),
                          shape=(len(hosts), len(universe)))
    own = universe.get_indexer(hosts)                                # позиция самого хоста среди кандидатов

    def cand_mask(h):
        m = np.ones(len(universe), bool)
        if own[h] >= 0:
            m[own[h]] = False
        return m

    pos = universe.get_indexer(tgt.partner)
    covered = pos >= 0
    targets = [np.array([], int)] * len(hosts)
    t = pd.DataFrame({"h": hosts.get_indexer(tgt.host_id[covered]), "pos": pos[covered]})
    for h, g in t.groupby("h"):
        targets[h] = g.pos.to_numpy()

    X = emb.to_numpy(float)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    U = X[hosts.get_indexer(universe)]                               # векторы кандидатов
    pop = np.asarray(M.sum(0)).ravel()
    rng = np.random.default_rng(seed)
    kk = min(k, len(hosts) - 1)
    acc = {v: ([], []) for v in VARIANTS}
    for start in range(0, len(hosts), batch):
        idx = np.arange(start, min(start + batch, len(hosts)))
        sim = X[idx] @ X.T
        sim[np.arange(len(idx)), idx] = -np.inf
        top = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
        w = np.take_along_axis(sim, top, 1).clip(min=0)
        W = sparse.csr_matrix((w.ravel(), (np.repeat(np.arange(len(idx)), kk), top.ravel())), shape=(len(idx), len(hosts)))
        friends = (W @ M).toarray()                                  # друзья: собеседники соседей с весами
        lp = X[idx] @ U.T
        soft_w = friends.copy()
        mine = own[idx] >= 0
        soft_w[np.flatnonzero(mine), own[idx][mine]] = 0             # сам хост — не друг себе
        centroid = soft_w @ U
        centroid /= np.maximum(np.linalg.norm(centroid, axis=1, keepdims=True), 1e-12)
        scores = {"lp_direct": lp, "lp_inverted": -lp, "transfer_direct": friends, "transfer_soft": centroid @ U.T,
                  "history": M[idx].toarray(), "popularity": np.broadcast_to(pop, (len(idx), len(universe))),
                  "random": rng.random((len(idx), len(universe)))}
        for name, S in scores.items():
            rr, rec = _per_host(S, idx, cand_mask, targets, top_n)
            acc[name][0].extend(rr)
            acc[name][1].extend(rec)
    out = {"k_neighbors": k, "top_n": top_n, "n_candidates": int(len(universe)),
           "target_coverage": float(covered.mean()) if len(covered) else 0.0}
    for name in VARIANTS:
        rr, rec = map(np.array, acc[name])
        out[name] = {"mrr": float(rr.mean()) if len(rr) else None,
                     "recall_at_n": float(rec.mean()) if len(rec) else None, "n_hosts": int(len(rr))}
        if name in ("lp_direct", "transfer_direct", "transfer_soft") and len(rr):
            out[name]["mrr_ci"] = bootstrap_ci(lambda i, v=rr: float(v[i].mean()), len(rr), n_boot=n_boot, seed=seed)
    return out
