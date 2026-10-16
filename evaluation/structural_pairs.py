"""Пары и тройки «похожи / не похожи» по ОБЩИМ СОБЕСЕДНИКАМ в более позднем окне.

Это приближение структурной похожести — специально отличное от поведенческого профиля (neighbors.csv)
и от будущих связей (edges.csv):

    J(a, b) = |peers(a) ∩ peers(b)| / |peers(a) ∪ peers(b)|      (целевое окно)

peers(h) — все хосты, с которыми h обменялся хотя бы одним потоком (в любую сторону, любые хосты).

Правило разметки (фиксировано заранее, пишется в манифест):
    похожи        J >= tau   (по умолчанию max(0.1, 99-й перцентиль J по всем парам окна))
    не похожи     J == 0     (ни одного общего собеседника)
    не размечено  0 < J < tau (неоднозначно; НИКОГДА не считается «непохожим»)

pairs.csv — до pairs_per_anchor похожих и непохожих на хост (seed фиксирован); triplets.csv — пары
«похожий + непохожий» одного хоста. split пары — детерминированный хеш пары, от эмбеддингов не зависит.
Это не метки ролей.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import scipy.sparse as sp

RULE_VERSION = 1


def _split(key: str, seed: int, validation_fraction: float) -> str:
    h = int(hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "validation" if h < validation_fraction else "test"


def jaccard_matrix(flows: pd.DataFrame, hosts: list[str]) -> np.ndarray:
    """Матрица Jaccard по множествам собеседников, |hosts| x |hosts|, на диагонали 0."""
    mask = flows["src"] != flows["dst"]
    a, b = flows.loc[mask, "src"].astype(str), flows.loc[mask, "dst"].astype(str)
    universe = pd.Index(sorted(set(a) | set(b) | set(hosts)))
    ia, ib = universe.get_indexer(a), universe.get_indexer(b)
    adj = sp.coo_matrix((np.ones(len(ia)), (ia, ib)), shape=(len(universe),) * 2).tocsr()
    adj = ((adj + adj.T) > 0).astype(np.float64)          # неориентированно, 0/1
    rows = adj[universe.get_indexer(pd.Index(hosts))]
    inter = (rows @ rows.T).toarray()
    size = np.asarray(rows.sum(axis=1)).ravel()
    union = size[:, None] + size[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = np.where(union > 0, inter / union, 0.0)
    np.fill_diagonal(jac, 0.0)
    return jac


def build_judgments(flows: pd.DataFrame, hosts: list[str], *, tau_pos: float | None = None,
                    pairs_per_anchor: int = 5, seed: int = 0,
                    validation_fraction: float = 0.3) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(pairs, triplets, summary). Всё берётся только из целевого окна."""
    hosts = sorted(hosts)
    if len(hosts) != len(set(hosts)) or len(hosts) < 3:
        raise ValueError("need >= 3 unique hosts")
    if tau_pos is not None and not 0 < tau_pos <= 1:
        raise ValueError("tau_pos must be in (0, 1]")
    jac = jaccard_matrix(flows, hosts)
    n = len(hosts)
    iu = np.triu_indices(n, 1)
    rule = "fixed" if tau_pos is not None else "max(0.1, 99th percentile of pairwise Jaccard)"
    if tau_pos is None:
        tau_pos = float(max(0.1, np.quantile(jac[iu], 0.99)))
    pos_mask, neg_mask = jac >= tau_pos, jac == 0
    np.fill_diagonal(pos_mask, False)
    np.fill_diagonal(neg_mask, False)
    total = len(iu[0])
    summary = {
        "rule_version": RULE_VERSION, "tau_pos": tau_pos, "tau_rule": rule, "pairs_per_anchor": pairs_per_anchor,
        "seed": seed, "n_hosts": n, "n_possible_pairs": int(total),
        "n_positive_pairs": int(pos_mask[iu].sum()), "n_negative_pairs": int(neg_mask[iu].sum()),
        "n_unjudged_pairs": int(total - pos_mask[iu].sum() - neg_mask[iu].sum()),
        "relevance_definition": "positive: peer-set Jaccard >= tau_pos in the target window; "
                                "negative: Jaccard == 0; otherwise unjudged (never negative)",
    }
    rng = np.random.default_rng(seed)
    pair_rows, trip_rows = {}, []
    for i, anchor in enumerate(hosts):
        pos = np.flatnonzero(pos_mask[i])
        neg = np.flatnonzero(neg_mask[i])
        if len(pos) == 0 or len(neg) == 0:
            continue
        pos = rng.choice(pos, size=min(pairs_per_anchor, len(pos)), replace=False)
        neg = rng.choice(neg, size=min(pairs_per_anchor, len(neg)), replace=False)
        for j, label in [(int(j), 1) for j in pos] + [(int(j), 0) for j in neg]:
            a, b = (anchor, hosts[j]) if anchor < hosts[j] else (hosts[j], anchor)
            pair_rows[(a, b)] = (a, b, label, float(jac[i, j]))
        for p in pos:
            nn = int(rng.choice(neg))
            trip_rows.append((anchor, hosts[int(p)], hosts[nn]))
    if not pair_rows:
        raise ValueError(
            f"No judged pairs: {summary['n_positive_pairs']} positive and "
            f"{summary['n_negative_pairs']} negative pairs at tau_pos={tau_pos}. The target "
            "window graph is too dense/sparse for this rule; adjust tau_pos or the window.")
    pairs = pd.DataFrame(sorted(pair_rows.values()), columns=["host_a", "host_b", "label", "ref_score"])
    pairs["split"] = [_split(f"{a}|{b}", seed, validation_fraction)
                      for a, b in zip(pairs.host_a, pairs.host_b)]
    triplets = pd.DataFrame(sorted(set(trip_rows)), columns=["anchor", "positive", "negative"])
    triplets["split"] = [_split(f"{a}|{p}|{n}", seed, validation_fraction)
                         for a, p, n in zip(triplets.anchor, triplets.positive, triplets.negative)]
    summary.update(n_pairs=int(len(pairs)), n_triplets=int(len(triplets)),
                   n_anchors_without_judgments=int(n - pairs[["host_a", "host_b"]].stack().nunique()))
    return pairs, triplets, summary
