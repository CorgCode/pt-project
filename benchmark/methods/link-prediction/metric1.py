"""
idea1_classic_lp.py — идея 1: классический link prediction по эмбеддингу.

Гипотеза (заведомо ложная для ролей, используется как контроль):
    похожие хосты связаны между собой в графе общения.

Метрика:
    для каждого хоста ранжируем всех кандидатов по близости эмбеддингов
    и смотрим, где оказались реальные собеседники в target-окне.

Варианты:
    - прямой: скор = близость (косинус или −расстояние);
    - инвертированный: скор = −близость. Нужен как контроль: если он даёт
      результат лучше прямого, гипотеза «похожие ⇒ связаны» явно неверна,
      а эмбеддинг, скорее всего, отражает роли, а не соседство.

Запуск:
    python idea1_classic_lp.py embeddings.csv edges.csv \
        --source-days 1 --target-days 2 3 --top-n 10 --metric cosine

Импорт из бенчмарка:
    from idea1_classic_lp import run_evaluation
    results = run_evaluation(embeddings_path, edges_path, source_days=[1], target_days=[2])
"""

from __future__ import annotations

import argparse
from typing import List, Optional

import numpy as np

from common import (
    EvalConfig,
    EvalContext,
    EvalResult,
    build_context,
    evaluate,
    lift,
    rank_by_scores,
    run_baselines,
)


def _similarity_scores(host: str, ctx: EvalContext) -> dict:
    """Скор кандидатов: чем выше, тем ближе к хосту по эмбеддингу."""
    i = ctx.index[host]
    if ctx.config.metric == "cosine":
        # Векторы нормализованы, поэтому произведение = косинус
        sims = ctx.vectors @ ctx.vectors[i]
    else:
        # Для евклида «близость» — это минус расстояние
        sims = -np.linalg.norm(ctx.vectors - ctx.vectors[i], axis=1)

    return {
        it: float(sims[ctx.index[it[0]]]) for it in ctx.candidates_for(host)
    }


def direct_ranker(host, ctx: EvalContext):
    """Гипотеза как есть: ближайшие по эмбеддингу — самые вероятные собеседники."""
    return rank_by_scores(host, ctx, _similarity_scores(host, ctx))


def inverted_ranker(host, ctx: EvalContext):
    """Контроль: инвертируем гипотезу, самые далёкие — самые вероятные."""
    scores = {it: -s for it, s in _similarity_scores(host, ctx).items()}
    return rank_by_scores(host, ctx, scores)


def run_evaluation(
    embeddings_path: str,
    edges_path: str,
    source_days: List[int],
    target_days: List[int],
    use_dst_port: bool = False,
    top_n: int = 10,
    metric: str = "cosine",
    random_seed: int = 0,
    ctx: Optional[EvalContext] = None,
) -> List[EvalResult]:
    """
    Запускает идею 1 (прямой и инвертированный варианты) вместе с бейзлайнами.

    Возвращает список результатов. У каждого метода в результате есть
    поле mrr и per_split; сравнение с бейзлайнами — через lift().
    """
    if ctx is None:
        config = EvalConfig(
            embeddings_path=embeddings_path,
            edges_path=edges_path,
            source_days=source_days,
            target_days=target_days,
            use_dst_port=use_dst_port,
            top_n=top_n,
            metric=metric,
            random_seed=random_seed,
        )
        ctx = build_context(config)

    baselines = run_baselines(ctx)
    direct = evaluate("idea1_direct", direct_ranker, ctx)
    inverted = evaluate("idea1_inverted_control", inverted_ranker, ctx)

    print(f"Прирост над бейзлайнами: direct={lift(direct, baselines):+.4f}, "
          f"inverted={lift(inverted, baselines):+.4f}")

    return [direct, inverted, *baselines]


def main() -> None:
    parser = argparse.ArgumentParser(description="Идея 1: классический link prediction")
    parser.add_argument("embeddings", help="embeddings.csv (id, embedding)")
    parser.add_argument("edges", help="edges.csv из build_edges.py")
    parser.add_argument("--source-days", type=int, nargs="+", required=True)
    parser.add_argument("--target-days", type=int, nargs="+", required=True)
    parser.add_argument("--use-dst-port", action="store_true")
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
        top_n=args.top_n,
        metric=args.metric,
        random_seed=args.seed,
    )
    for r in results:
        print()
        print(r)


if __name__ == "__main__":
    main()
