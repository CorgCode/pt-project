"""Узнаётся ли хост через время? (#41; поиск — Даниил, AUC по парам — Даня)

Берём эмбеддинги ОДНОГО метода в окне A (раньше) и окне B (позже) для одних и тех же хостов.

Поиск: для хоста h запрос — его вектор из A, галерея — векторы всех оцениваемых хостов из B.
  Top-1 = P(rank = 1), MRR = E[1/rank], Recall@k = P(rank <= k).
  Ничьи разыгрываются честно (считается матожидание), поэтому константный эмбеддинг
  даёт ровно случайный уровень: 1/n, H_n/n, k/n.
AUC по парам: положительная пара — (h в A, h в B), отрицательные — (h в A, другой хост в B),
  по `negatives_per_positive` случайных на хост. ROC-AUC и PR-AUC по косинусу.

Системы координат. Если модель обучали отдельно на каждом окне, оси могут повернуться,
и прямой косинус между окнами ничего не значит. Поэтому:
  alignment="none"        сырые координаты — только если у метода одна система координат;
  alignment="procrustes"  поворот B->A подбирается на половине хостов (по seed),
                          метрики считаются только на другой половине.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes

from evaluation.uncertainty import auc_with_ci, bootstrap_ci


def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    if (n == 0).any():
        raise ValueError("нулевой вектор: косинус не определён")
    return x / n


def identity_retrieval(emb_a: pd.DataFrame, emb_b: pd.DataFrame, *, alignment: str = "none",
                       seed: int = 0, ks=(1, 5, 10), n_boot: int = 0, eval_fraction: float = 0.5,
                       negatives_per_positive: int = 5) -> dict:
    if alignment not in {"none", "procrustes"}:
        raise ValueError("alignment должен быть 'none' или 'procrustes'")
    if set(emb_a.index) != set(emb_b.index):
        only_a, only_b = sorted(set(emb_a.index) - set(emb_b.index)), sorted(set(emb_b.index) - set(emb_a.index))
        raise ValueError(f"в окнах разные хосты (только в A: {only_a[:5]}, только в B: {only_b[:5]})")
    hosts = sorted(emb_a.index)
    n = len(hosts)
    if n < 20:
        raise ValueError("нужно хотя бы 20 хостов")
    A, B = emb_a.loc[hosts].to_numpy(float), emb_b.loc[hosts].to_numpy(float)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_eval = int(round(n * eval_fraction))
    ev, al = np.sort(perm[:n_eval]), np.sort(perm[n_eval:])
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"размерности окон различаются ({A.shape[1]} и {B.shape[1]})")
    if alignment == "procrustes":
        mu_a, mu_b = A[al].mean(0), B[al].mean(0)
        R, _ = orthogonal_procrustes(B[al] - mu_b, A[al] - mu_a)
        A, B = A - mu_a, (B - mu_b) @ R
    sim = _unit(A[ev]) @ _unit(B[ev]).T
    true = np.diag(sim)[:, None]
    greater = (sim > true).sum(1)
    ties = (np.isclose(sim, true, rtol=0, atol=1e-12)).sum(1) - 1
    m = len(ev)

    def p_rank_le(k: int) -> np.ndarray:
        return np.clip(k - greater, 0, ties + 1) / (ties + 1)

    rr = np.array([np.sum(1.0 / (g + 1 + np.arange(t + 1))) / (t + 1) for g, t in zip(greater, ties)])
    out = {
        "alignment": alignment, "n_eval_hosts": int(m),
        "n_alignment_hosts": int(len(al)) if alignment == "procrustes" else 0,
        "top1": float(p_rank_le(1).mean()), "mrr": float(rr.mean()),
        "chance_top1": 1.0 / m, "chance_mrr": float(np.sum(1.0 / np.arange(1, m + 1)) / m),
        "recall": {str(k): float(p_rank_le(k).mean()) for k in ks},
        "chance_recall": {str(k): min(1.0, k / m) for k in ks},
        "seed": seed,
    }
    hit = p_rank_le(1)
    out["top1_ci"] = bootstrap_ci(lambda i: float(hit[i].mean()), m, n_boot=n_boot, seed=seed)

    # AUC по парам «тот же хост / другой хост» на тех же оцениваемых хостах
    take = min(negatives_per_positive, m - 1)
    rows, cols = list(range(m)), list(range(m))
    for i in range(m):
        others = np.delete(np.arange(m), i)
        neg = rng.choice(others, size=take, replace=False)
        rows += [i] * take
        cols += neg.tolist()
    rows, cols = np.array(rows), np.array(cols)
    labels = (rows == cols).astype(int)
    auc = auc_with_ci(labels, sim[rows, cols], n_boot=n_boot, seed=seed, hosts=([rows, cols], m))
    out.update({"pair_roc_auc": auc["roc_auc"], "pair_pr_auc": auc["pr_auc"],
                "pair_roc_auc_ci": auc["roc_auc_ci"], "pair_positive_rate": float(labels.mean()),
                "negatives_per_positive": take})
    return out


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    from evaluation.io import load_embeddings

    ap = argparse.ArgumentParser(description="Узнаётся ли хост в более позднем окне")
    ap.add_argument("--embeddings-a", type=Path, required=True, help="раннее окно")
    ap.add_argument("--embeddings-b", type=Path, required=True, help="позднее окно")
    ap.add_argument("--method", required=True)
    ap.add_argument("--shared-coordinate-system", action="store_true",
                    help="оба файла в одной системе координат (тогда сырой косинус осмыслен)")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    A, B = load_embeddings(a.embeddings_a), load_embeddings(a.embeddings_b)
    res = {"method": a.method, "windows": {"a": str(a.embeddings_a), "b": str(a.embeddings_b)},
           "shared_coordinate_system_declared": a.shared_coordinate_system}
    for mode in ("none", "procrustes"):
        try:
            res[mode] = identity_retrieval(A, B, alignment=mode, seed=a.seed, n_boot=a.bootstrap)
        except ValueError as exc:
            res[mode] = {"unavailable": str(exc)}
    text = json.dumps(res, indent=2, sort_keys=True, ensure_ascii=False)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
