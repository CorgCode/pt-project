"""Temporal host-identity retrieval: is a host recognisable in a later window?

Inputs: embeddings of the SAME method for window A (earlier) and window B (later) over the
SAME cohort (host IDs must match exactly). For each evaluation host h, the query is its
window-A vector and the gallery is all evaluation hosts' window-B vectors:

    g = #{g' : s(h,g') > s(h,h)},  t = #{g' != h : s(h,g') = s(h,h)}   (ties with the truth)
    the true host occupies one of the positions g+1 .. g+t+1 with equal probability, so
    (expected values under random tie-breaking - a constant embedding scores exactly chance)
      P(rank<=k) = clip(k-g, 0, t+1) / (t+1)         Top-1 = P(rank<=1),  Recall@k = P(rank<=k)
      E[1/rank]  = (1/(t+1)) * sum_{j=0..t} 1/(g+1+j)  MRR = mean over hosts
    chance (random ranking): Top-1 = 1/n, MRR = H_n / n (harmonic number), Recall@k = k/n

Identity is NOT behavioural similarity and not a role label: same ID in two windows is the
judgment. A high score shows an embedding is stable AND discriminative across time; a
constant or low-rank embedding scores near chance.

Coordinate systems: independently fitted models may rotate/permute axes between windows, so
cosine across windows is meaningless unless the method uses ONE coordinate system.
  alignment="none"        use raw coordinates (valid only if the method declares a shared
                          coordinate system, e.g. fixed features / one model applied to both)
  alignment="procrustes"  orthogonal Procrustes map B->A fitted on a seeded half of the
                          hosts ("alignment hosts"); evaluation uses ONLY the other half, so
                          identities of evaluated hosts never inform the alignment.
Both are computed on the same evaluation hosts. Procrustes needs equal dimensions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes

from evaluation.uncertainty import bootstrap_ci


def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    if (n == 0).any():
        raise ValueError("zero-norm host vectors make cosine undefined")
    return x / n


def identity_retrieval(emb_a: pd.DataFrame, emb_b: pd.DataFrame, *, alignment: str = "none",
                       seed: int = 0, ks=(1, 5, 10), n_boot: int = 0, eval_fraction: float = 0.5) -> dict:
    if alignment not in {"none", "procrustes"}:
        raise ValueError("alignment must be 'none' or 'procrustes'")
    if set(emb_a.index) != set(emb_b.index):
        only_a, only_b = sorted(set(emb_a.index) - set(emb_b.index)), sorted(set(emb_b.index) - set(emb_a.index))
        raise ValueError(f"windows contain different host IDs (only in A: {only_a[:5]}, only in B: {only_b[:5]})")
    hosts = sorted(emb_a.index)
    n = len(hosts)
    if n < 20:
        raise ValueError("need at least 20 hosts for a meaningful identity retrieval")
    A, B = emb_a.loc[hosts].to_numpy(float), emb_b.loc[hosts].to_numpy(float)
    perm = np.random.default_rng(seed).permutation(n)
    n_eval = int(round(n * eval_fraction))
    ev, al = np.sort(perm[:n_eval]), np.sort(perm[n_eval:])
    if alignment == "procrustes":
        if A.shape[1] != B.shape[1]:
            raise ValueError(f"procrustes needs equal dimensions ({A.shape[1]} vs {B.shape[1]})")
        mu_a, mu_b = A[al].mean(0), B[al].mean(0)
        R, _ = orthogonal_procrustes(B[al] - mu_b, A[al] - mu_a)
        A, B = A - mu_a, (B - mu_b) @ R
    elif A.shape[1] != B.shape[1]:
        raise ValueError(f"alignment='none' needs equal dimensions ({A.shape[1]} vs {B.shape[1]})")
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
    return out


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    from evaluation.io import load_embeddings

    ap = argparse.ArgumentParser(description="Temporal host-identity retrieval between two windows")
    ap.add_argument("--embeddings-a", type=Path, required=True, help="earlier window")
    ap.add_argument("--embeddings-b", type=Path, required=True, help="later window")
    ap.add_argument("--method", required=True)
    ap.add_argument("--shared-coordinate-system", action="store_true",
                    help="declare that both files come from ONE coordinate system (enables the "
                         "raw-coordinate score to be interpreted)")
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
    res["interpretation"] = (
        "alignment='none' is meaningful only when a shared coordinate system is declared; "
        "otherwise use the procrustes score. Identity is not behavioural similarity.")
    text = json.dumps(res, indent=2, sort_keys=True)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
