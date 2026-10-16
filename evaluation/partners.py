"""Прогноз собеседников через похожих хостов (идея 3 из #40, автор — Тихон).

Гипотеза: похожие хосты общаются с похожими собеседниками. Для хоста A берём k ближайших
по эмбеддингу хостов (вес = косинус, только положительный), собираем их собеседников
в train-окне и ранжируем кандидатов по сумме «вес соседа × число контактов».
Проверяем по реальным собеседникам A в целевом окне.

Вход: reference/partners.csv — host_id, partner, window (source|target), count.
Метрики (среднее по хостам): MRR, Recall@N. При равных скорах берётся средний ранг.
Рядом считаются бейзлайны без эмбеддинга на тех же кандидатах:
  history    — с кем A общался раньше (по частоте);
  popularity — самые популярные собеседники.
Отдельно — только НОВЫЕ собеседники (которых у A не было в train): старые собеседники A
убираются из кандидатов, история здесь бессильна, и видно, что даёт именно эмбеддинг.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from evaluation.io import ReferenceError
from evaluation.uncertainty import bootstrap_ci

COLUMNS = {"host_id", "partner", "window", "count"}


def build_partners(source: pd.DataFrame, target: pd.DataFrame, hosts: list[str]) -> pd.DataFrame:
    """Собеседники хостов `hosts` в двух окнах. Связь ненаправленная, петли выкидываются."""
    hosts_set = set(hosts)
    parts = []
    for window, flows in (("source", source), ("target", target)):
        f = flows[["src", "dst"]].astype(str)
        f = f[f.src != f.dst]
        both = pd.concat([f.rename(columns={"src": "host_id", "dst": "partner"}),
                          f.rename(columns={"dst": "host_id", "src": "partner"})], ignore_index=True)
        both = both[both.host_id.isin(hosts_set)]
        c = both.groupby(["host_id", "partner"]).size().rename("count").reset_index()
        c.insert(2, "window", window)
        parts.append(c)
    return pd.concat(parts, ignore_index=True).sort_values(["window", "host_id", "partner"]).reset_index(drop=True)


def load_partners(path: Path) -> pd.DataFrame:
    p = pd.read_csv(path, dtype={"host_id": str, "partner": str, "window": str})
    if not COLUMNS <= set(p.columns):
        raise ReferenceError(f"{path}: нужны колонки {sorted(COLUMNS)}")
    if not p.window.isin(["source", "target"]).all():
        raise ReferenceError(f"{path}: window должен быть source или target")
    if (p["count"] <= 0).any():
        raise ReferenceError(f"{path}: count должен быть > 0")
    return p


def _mid_ranks(scores: np.ndarray, items: np.ndarray) -> np.ndarray:
    """Ранг (1 = лучший) элементов `items` в строке `scores`; при равенстве — средний ранг."""
    s = scores[items]
    greater = (scores[None, :] > s[:, None]).sum(1)
    ties = (scores[None, :] == s[:, None]).sum(1) - 1
    return 1 + greater + ties / 2


def _per_host(score_rows, hosts_idx, cand_mask_fn, targets, top_n):
    rr, rec = [], []
    for row, h in zip(score_rows, hosts_idx):
        rel = targets[h]
        if len(rel) == 0:
            continue
        row = np.array(row, float)
        row[~cand_mask_fn(h)] = -np.inf
        ranks = _mid_ranks(row, rel)
        rr.append(1.0 / ranks.min())
        rec.append(float((ranks <= top_n).mean()))
    return np.array(rr), np.array(rec)


def partner_ranking(emb: pd.DataFrame, partners: pd.DataFrame, *, k: int = 10, top_n: int = 10,
                    n_boot: int = 0, seed: int = 0, batch: int = 512) -> dict:
    hosts = list(emb.index)
    src = partners[partners.window == "source"]
    tgt = partners[partners.window == "target"]
    universe = pd.Index(sorted(src.partner.unique()))          # кандидаты — все, кто встречался в train
    if len(universe) < 2:
        raise ValueError("в source-окне меньше двух собеседников")
    h_idx = pd.Index(hosts)
    s = src[src.host_id.isin(h_idx)]
    M = sparse.csr_matrix((s["count"].to_numpy(float), (h_idx.get_indexer(s.host_id), universe.get_indexer(s.partner))),
                          shape=(len(hosts), len(universe)))
    own = universe.get_indexer(h_idx)                           # позиция самого хоста среди кандидатов (-1 если нет)

    def cand_mask(h, new_only=False):
        m = np.ones(len(universe), bool)
        if own[h] >= 0:
            m[own[h]] = False
        if new_only:                                            # старые собеседники не могут быть «новыми»
            m[M.indices[M.indptr[h]:M.indptr[h + 1]]] = False
        return m

    # цели: собеседники в target-окне, которые вообще есть среди кандидатов
    t = tgt[tgt.host_id.isin(h_idx)]
    t_pos = universe.get_indexer(t.partner)
    n_target = len(t)
    covered = t_pos >= 0
    t = t[covered].assign(pos=t_pos[covered], h=h_idx.get_indexer(t.host_id[covered]))
    seen = set(zip(h_idx.get_indexer(s.host_id), universe.get_indexer(s.partner)))
    targets_all = [np.array([], int)] * len(hosts)
    targets_new = [np.array([], int)] * len(hosts)
    for h, g in t.groupby("h"):
        targets_all[h] = g.pos.to_numpy()
        targets_new[h] = np.array([p for p in g.pos if (h, p) not in seen], int)

    X = emb.to_numpy(float)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    pop = np.asarray(M.sum(0)).ravel()
    out = {"k_neighbors": k, "top_n": top_n, "n_candidates": int(len(universe)),
           "target_coverage": float(covered.mean()) if n_target else 0.0}
    views = {"all": targets_all, "new": targets_new}
    acc = {(v, n): ([], []) for v in views for n in ("embedding", "history", "popularity")}
    kk = min(k, len(hosts) - 1)
    for start in range(0, len(hosts), batch):
        idx = np.arange(start, min(start + batch, len(hosts)))
        sim = X[idx] @ X.T
        sim[np.arange(len(idx)), idx] = -np.inf
        top = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
        w = np.take_along_axis(sim, top, 1).clip(min=0)
        W = sparse.csr_matrix((w.ravel(), (np.repeat(np.arange(len(idx)), kk), top.ravel())), shape=(len(idx), len(hosts)))
        scores = {"embedding": (W @ M).toarray(), "history": M[idx].toarray(),
                  "popularity": np.broadcast_to(pop, (len(idx), len(universe)))}
        for v, targets in views.items():
            for name, S in scores.items():
                rr, rec = _per_host(S, idx, lambda h, v=v: cand_mask(h, v == "new"), targets, top_n)
                acc[v, name][0].extend(rr)
                acc[v, name][1].extend(rec)
    for view in views:
        res = {}
        for name in ("embedding", "history", "popularity"):
            rr, rec = map(np.array, acc[view, name])
            res[name] = {"mrr": float(rr.mean()) if len(rr) else None,
                         "recall_at_n": float(rec.mean()) if len(rec) else None, "n_hosts": int(len(rr))}
            if name == "embedding" and len(rr):
                res[name]["mrr_ci"] = bootstrap_ci(lambda i, v=rr: float(v[i].mean()), len(rr), n_boot=n_boot, seed=seed)
        best = max((res[b]["mrr"] or 0) for b in ("history", "popularity"))
        res["lift_vs_best_baseline"] = ((res["embedding"]["mrr"] - best) / best
                                        if res["embedding"]["mrr"] is not None and best > 0 else None)
        out[view] = res
    return out
