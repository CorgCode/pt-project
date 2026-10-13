"""
idea3_neighborhood_transfer.py — идея 3: перенос окрестностей.

Гипотеза: роли общаются с ролями. Если похожие на A хосты общались
с кем-то в прошлом, то и A будет общаться с кем-то похожим на них.

Шаги для хоста A:
1. Соседи N: k ближайших хостов по эмбеддингу, веса = схожесть (только положительные).
2. Друзья F: все собеседники соседей в source-окне, взвешенные по весу соседа и частоте.
3. Прогноз — два варианта:
   - direct (вариант 1): кандидаты = сами друзья, скор = сумма весов;
   - soft (вариант 2, ваш пересказ): кандидаты, близкие к друзьям,
     скор = схожесть с взвешенным центроидом эмбеддингов друзей.

Метрика: MRR и recall@N по реальным собеседникам A в target-окне,
сравнение с бейзлайнами и lift (через common.py).

Запуск:
    python idea3_neighborhood_transfer.py embeddings.csv edges.csv \
        --source-days 1 --target-days 2 3 --k 10 --top-n 10
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from common import (
    EvalConfig,
    EvalContext,
    EvalResult,
    Item,
    build_context,
    evaluate,
    lift,
    partner_scores_to_items,
    rank_by_scores,
    run_baselines,
)


def _neighbors_with_weights(host: str, ctx: EvalContext) -> Dict[str, float]:
    """k ближайших хостов и их веса (схожесть, только положительная)."""
    sims = ctx.similarity_to_all(host)
    k = min(ctx.config.k_neighbors, len(ctx.ids) - 1)
    top = np.argpartition(-sims, k)[:k]

    weights: Dict[str, float] = {}
    for j in top:
        w = float(sims[j])
        if np.isfinite(w) and w > 0:
            weights[ctx.ids[j]] = w
    return weights


def direct_ranker(host: str, ctx: EvalContext) -> List[Item]:
    """
    Вариант 1: друзья как есть.
    Скор элемента = сумма по соседям: вес соседа × частота контакта с этим элементом.
    """
    scores: Dict[Item, float] = defaultdict(float)
    for neighbor, w in _neighbors_with_weights(host, ctx).items():
        for item, count in ctx.source_neighbors.get(neighbor, {}).items():
            scores[item] += w * count
    return rank_by_scores(host, ctx, dict(scores))


def soft_ranker(host: str, ctx: EvalContext) -> List[Item]:
    """
    Вариант 2: кандидаты, близкие к друзьям.
    Друзья (по партнёрам) с весами; центроид их эмбеддингов;
    скор кандидата = схожесть с центроидом.
    Для каждого кандидата это то же, что близость к взвешенной группе друзей.
    """
    friend_weights: Dict[str, float] = defaultdict(float)
    for neighbor, w in _neighbors_with_weights(host, ctx).items():
        for item, count in ctx.source_neighbors.get(neighbor, {}).items():
            partner = item[0]
            if partner == host or partner not in ctx.index:
                continue
            friend_weights[partner] += w * count

    if not friend_weights:
        return rank_by_scores(host, ctx, {})

    total = sum(friend_weights.values())
    dim = ctx.vectors.shape[1]
    centroid = np.zeros(dim, dtype=np.float64)
    for partner, w in friend_weights.items():
        centroid += (w / total) * ctx.vectors[ctx.index[partner]]

    if ctx.config.metric == "cosine":
        norm = np.linalg.norm(centroid)
        if norm == 0:
            return rank_by_scores(host, ctx, {})
        sims = ctx.vectors @ (centroid / norm)
    else:
        sims = -np.linalg.norm(ctx.vectors - centroid, axis=1)

    partner_scores = {ctx.ids[j]: float(sims[j]) for j in range(len(ctx.ids))}
    scores = partner_scores_to_items(host, ctx, partner_scores)
    return rank_by_scores(host, ctx, scores)


def run_evaluation(
    embeddings_path: str,
    edges_path: str,
    source_days: List[int],
    target_days: List[int],
    use_dst_port: bool = False,
    k: int = 10,
    top_n: int = 10,
    metric: str = "cosine",
    random_seed: int = 0,
    ctx: Optional[EvalContext] = None,
) -> List[EvalResult]:
    """Запускает оба варианта идеи 3 вместе с бейзлайнами."""
    if ctx is None:
        config = EvalConfig(
            embeddings_path=embeddings_path,
            edges_path=edges_path,
            source_days=source_days,
            target_days=target_days,
            use_dst_port=use_dst_port,
            top_n=top_n,
            k_neighbors=k,
            metric=metric,
            random_seed=random_seed,
        )
        ctx = build_context(config)

    baselines = run_baselines(ctx)
    direct = evaluate("idea3_direct", direct_ranker, ctx)
    soft = evaluate("idea3_soft", soft_ranker, ctx)

    best_baseline = max(r.mrr for r in baselines)
    print(f"Прирост над лучшим бейзлайном (MRR={best_baseline:.4f}): "
          f"direct={lift(direct, baselines):+.4f}, soft={lift(soft, baselines):+.4f}")

    return [direct, soft, *baselines]


def main() -> None:
    parser = argparse.ArgumentParser(description="Идея 3: перенос окрестностей")
    parser.add_argument("embeddings", help="embeddings.csv (id, embedding)")
    parser.add_argument("edges", help="edges.csv из build_edges.py")
    parser.add_argument("--source-days", type=int, nargs="+", required=True)
    parser.add_argument("--target-days", type=int, nargs="+", required=True)
    parser.add_argument("--use-dst-port", action="store_true")
    parser.add_argument("--k", type=int, default=10, help="число соседей по эмбеддингу")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = run_evaluation(
        embeddings_path=args.embeddings,
        edges_path=args.edges,
        source_days=args.source_days,
        target_days=args.target_days,
        use_dst_port=args.use_dst_port,
        k=args.k,
        top_n=args.top_n,
        metric=args.metric,
        random_seed=args.seed,
    )
    for r in results:
        print()
        print(r)


if __name__ == "__main__":
    main()