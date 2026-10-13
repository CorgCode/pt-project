"""
idea2_edge_removal_stability.py — идея 2: стабильность эмбеддинга при удалении рёбер.

Гипотеза: если эмбеддинг отражает поведение, то после удаления части связей
хосты должны получать похожие векторы (то же соседство, та же геометрия).

Работа в два шага:
1. reduce: из edges.csv удаляются целые пары хостов (все события A–B, любой порт).
   Пишется урезанный edges.csv и список удалённых пар. По ним внешний пайплайн
   пересчитывает эмбеддинг (эмбеддер не входит в этот файл).
2. compare: для каждого уровня удаления сравниваем эмбеддинги с эталонными
   (полный граф). Фоновый шум — сравнение эталонных запусков между собой
   (разные seed'ы на полном графе).

Методы сравнения:
- overlap (по умолчанию): доля общих k ближайших соседей хоста в обоих эмбеддингах.
  Не требует выравнивания пространств.
- procrustes: выравнивание поворотом и сдвигом, затем средний косинус строк.
  Требует одинаковой размерности и одинакового набора хостов.

Уровни удаления и seed'ы задаются конфигом. Уровень 0.0 — полный граф (эталон).

Примеры:
    # 1. создать урезанные рёбра
    python idea2_edge_removal_stability.py reduce edges.csv out/ --fractions 0.1 0.2 0.3 --seed 0

    # 2. сравнить эмбеддинги (0.0 — эталон, несколько путей — несколько seed'ов)
    python idea2_edge_removal_stability.py compare \
        --embeddings 0.0=emb_full_s0.csv,emb_full_s1.csv 0.1=emb_r10_s0.csv 0.2=emb_r20_s0.csv \
        --k 10 --method overlap
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
from scipy.linalg import orthogonal_procrustes

from common import load_embeddings

Pair = Tuple[str, str]


# ---------------------------------------------------------------------------
# Шаг 1: удаление пар рёбер
# ---------------------------------------------------------------------------

def reduce_edges(
    edges_path: str,
    out_dir: str,
    fraction: float,
    seed: int = 0,
) -> Tuple[Path, Path, int]:
    """
    Удаляет случайную долю уникальных пар хостов (все их строки целиком).
    Возвращает пути к урезанному edges.csv и списку удалённых пар, и число пар.
    """
    if not 0 < fraction < 1:
        raise ValueError("fraction должен быть между 0 и 1")

    edges_path = Path(edges_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(edges_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    header = list(rows[0].keys()) if rows else ["src", "dst", "dst_port", "day", "count"]

    # Уникальные пары без учёта порта, отсортированные для воспроизводимости
    pairs = sorted({(r["src"], r["dst"]) for r in rows})
    rng = np.random.default_rng(seed)
    n_remove = int(round(len(pairs) * fraction))
    removed_idx = rng.choice(len(pairs), size=n_remove, replace=False)
    removed: Set[Pair] = {pairs[i] for i in removed_idx}

    tag = f"r{int(round(fraction * 100)):02d}_s{seed}"
    reduced_path = out_dir / f"edges_{tag}.csv"
    removed_path = out_dir / f"removed_pairs_{tag}.csv"

    with open(reduced_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            if (r["src"], r["dst"]) not in removed:
                writer.writerow([r[h] for h in header])

    with open(removed_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["src", "dst"])
        for pair in sorted(removed):
            writer.writerow(pair)

    return reduced_path, removed_path, len(removed)


# ---------------------------------------------------------------------------
# Шаг 2: сравнение эмбеддингов
# ---------------------------------------------------------------------------

def _common_matrices(
    emb_a: Dict[str, np.ndarray], emb_b: Dict[str, np.ndarray]
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Хосты, которые есть в обоих эмбеддингах, и их матрицы в одном порядке."""
    common = sorted(set(emb_a) & set(emb_b))
    if not common:
        raise ValueError("Нет общих хостов для сравнения")
    A = np.vstack([emb_a[h] for h in common]).astype(np.float64)
    B = np.vstack([emb_b[h] for h in common]).astype(np.float64)
    return common, A, B


def _normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def _knn_sets(X: np.ndarray, k: int) -> List[Set[int]]:
    """Для каждой строки — множество индексов её k ближайших соседей (косинус)."""
    Xn = _normalize(X)
    k = min(k, len(Xn) - 1)
    result: List[Set[int]] = []
    for i in range(len(Xn)):
        sims = Xn @ Xn[i]
        sims[i] = -np.inf
        top = np.argpartition(-sims, k)[:k]
        result.append(set(int(j) for j in top))
    return result


def compare_overlap(emb_ref: Dict[str, np.ndarray], emb_other: Dict[str, np.ndarray], k: int) -> float:
    """Средняя доля общих k ближайших соседей по общим хостам."""
    _, A, B = _common_matrices(emb_ref, emb_other)
    if len(A) < 3:
        raise ValueError("Слишком мало общих хостов")
    sets_a = _knn_sets(A, k)
    sets_b = _knn_sets(B, k)
    k_eff = min(k, len(A) - 1)
    return float(np.mean([len(a & b) / k_eff for a, b in zip(sets_a, sets_b)]))


def compare_procrustes(emb_ref: Dict[str, np.ndarray], emb_other: Dict[str, np.ndarray]) -> float:
    """
    Выравниваем второе пространство к первому поворотом (центрированные данные),
    затем средний косинус между совпавшими строками.
    """
    _, A, B = _common_matrices(emb_ref, emb_other)
    if A.shape[1] != B.shape[1]:
        raise ValueError("Procrustes требует одинаковой размерности")
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    R, _ = orthogonal_procrustes(B, A)  # минимизирует ||B R − A||
    B_aligned = _normalize(B @ R)
    A_norm = _normalize(A)
    return float(np.mean(np.sum(A_norm * B_aligned, axis=1)))


def compare(
    emb_ref: Dict[str, np.ndarray],
    emb_other: Dict[str, np.ndarray],
    k: int = 10,
    method: str = "overlap",
) -> float:
    if method == "overlap":
        return compare_overlap(emb_ref, emb_other, k)
    if method == "procrustes":
        return compare_procrustes(emb_ref, emb_other)
    raise ValueError(f"method должен быть 'overlap' или 'procrustes', получено {method}")


# ---------------------------------------------------------------------------
# Результат и запуск
# ---------------------------------------------------------------------------

@dataclass
class StabilityResult:
    method: str
    k: int
    reference_paths: List[str]
    noise_floor: float | None            # сходство эталонных seed'ов между собой
    by_level: Dict[float, Dict[str, float]]  # уровень -> {mean, std, n}
    config: dict = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [f"[idea2_stability, method={self.method}, k={self.k}]"]
        if self.noise_floor is not None:
            lines.append(f"Фоновый шум (эталон vs эталон, разные seed'ы): {self.noise_floor:.4f}")
        else:
            lines.append("Фоновый шум: нет (один эталонный запуск, добавьте несколько seed'ов)")
        for level in sorted(self.by_level):
            s = self.by_level[level]
            lines.append(
                f"  удаление {level:.0%}: mean={s['mean']:.4f}, std={s['std']:.4f}, n={int(s['n'])}"
            )
        return "\n".join(lines)


def run_stability(
    embeddings_by_level: Dict[float, Sequence[str]],
    k: int = 10,
    method: str = "overlap",
) -> StabilityResult:
    """
    embeddings_by_level: уровень удаления -> пути к эмбеддингам (разные seed'ы).
    Уровень 0.0 — эталон (полный граф).
    """
    if 0.0 not in embeddings_by_level or not embeddings_by_level[0.0]:
        raise ValueError("Нужен уровень 0.0 с путями к эталонным эмбеддингам")

    ref_paths = list(embeddings_by_level[0.0])
    ref_embs = [load_embeddings(p) for p in ref_paths]

    noise_floor = None
    if len(ref_embs) > 1:
        pair_scores = [
            compare(ref_embs[i], ref_embs[j], k, method)
            for i in range(len(ref_embs))
            for j in range(i + 1, len(ref_embs))
        ]
        noise_floor = float(np.mean(pair_scores))

    by_level: Dict[float, Dict[str, float]] = {}
    for level, paths in embeddings_by_level.items():
        if level == 0.0:
            continue
        scores = []
        for path in paths:
            other = load_embeddings(path)
            # Все запуски с удалением сравниваем с первым эталонным запуском.
            # Сравнение с эталоном того же seed'а потребует указывать seed в путях.
            scores.append(compare(ref_embs[0], other, k, method))
        by_level[level] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "n": float(len(scores)),
        }

    return StabilityResult(
        method=method,
        k=k,
        reference_paths=ref_paths,
        noise_floor=noise_floor,
        by_level=by_level,
        config={"k": k, "method": method, "levels": sorted(embeddings_by_level)},
    )


def _parse_levels(items: List[str]) -> Dict[float, List[str]]:
    """Разбирает аргументы вида 0.1=a.csv,b.csv."""
    result: Dict[float, List[str]] = {}
    for item in items:
        level_str, _, paths_str = item.partition("=")
        if not paths_str:
            raise argparse.ArgumentTypeError(f"ожидается уровень=путь, получено {item}")
        result[float(level_str)] = [p for p in paths_str.split(",") if p]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Идея 2: стабильность эмбеддинга при удалении рёбер")
    sub = parser.add_subparsers(dest="command", required=True)

    p_reduce = sub.add_parser("reduce", help="создать урезанные edges.csv")
    p_reduce.add_argument("edges", help="исходный edges.csv")
    p_reduce.add_argument("out_dir", help="папка для результатов")
    p_reduce.add_argument("--fractions", type=float, nargs="+", required=True)
    p_reduce.add_argument("--seed", type=int, default=0)

    p_cmp = sub.add_parser("compare", help="сравнить эмбеддинги по уровням удаления")
    p_cmp.add_argument("--embeddings", nargs="+", required=True,
                       help="уровень=путь1,путь2 (уровень 0.0 — эталон)")
    p_cmp.add_argument("--k", type=int, default=10)
    p_cmp.add_argument("--method", choices=["overlap", "procrustes"], default="overlap")

    args = parser.parse_args()

    if args.command == "reduce":
        for fraction in args.fractions:
            reduced, removed, n = reduce_edges(args.edges, args.out_dir, fraction, args.seed)
            print(f"доля {fraction:.0%}: удалено пар {n}; рёбра: {reduced}; список: {removed}")
    else:
        result = run_stability(_parse_levels(args.embeddings), k=args.k, method=args.method)
        print(result)


if __name__ == "__main__":
    main()