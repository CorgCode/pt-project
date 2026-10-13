"""
common.py — общая часть для трёх метрик оценки эмбеддинга.

Что здесь:
- EvalConfig: конфигурация (окна, порт, k, метрика расстояния);
- загрузка эмбеддингов (формат как в Triplet Accuracy: id, embedding);
- загрузка рёбер из edges.csv по дням;
- EvalContext: всё, что нужно метрикам, собранное один раз;
- общий цикл оценки evaluate() и бейзлайны;
- EvalResult с единым форматом результата.

Термины:
- item — то, что предсказываем: (partner,) без порта или (partner, dst_port) с портом;
- source window — дни, по которым строятся «друзья» (прошлое);
- target window — дни, по которым проверяется прогноз (будущее); каждый день — отдельный split.
"""

from __future__ import annotations

import csv
import zlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

Item = Tuple[str, ...]


# ---------------------------------------------------------------------------
# Конфигурация и результат
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    embeddings_path: str
    edges_path: str
    source_days: List[int]
    target_days: List[int]
    use_dst_port: bool = False
    top_n: int = 10
    k_neighbors: int = 10
    metric: str = "cosine"  # 'cosine' или 'euclidean'
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.metric not in ("cosine", "euclidean"):
            raise ValueError(f"metric должна быть 'cosine' или 'euclidean', получено {self.metric}")
        if set(self.source_days) & set(self.target_days):
            raise ValueError("source_days и target_days не должны пересекаться")


@dataclass
class EvalResult:
    name: str
    mrr: float
    recall_at_n: float
    n_hosts: int
    coverage: float
    per_split: Dict[str, Dict[str, float]]
    config: dict = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"[{self.name}]",
            f"MRR: {self.mrr:.4f}",
            f"Recall@{self.config.get('top_n', '?')}: {self.recall_at_n:.4f}",
            f"Хостов оценено: {self.n_hosts}",
            f"Coverage (целевые рёбра, попавшие в кандидаты): {self.coverage:.4f}",
        ]
        for split in sorted(self.per_split):
            s = self.per_split[split]
            lines.append(
                f"  {split}: MRR={s['mrr']:.4f}, Recall@N={s['recall_at_n']:.4f}, "
                f"hosts={int(s['n_hosts'])}, coverage={s['coverage']:.4f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------

def load_embeddings(path: str) -> Dict[str, np.ndarray]:
    """Загружает CSV с колонками id, embedding (значения через пробел)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    embeddings: Dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not {"id", "embedding"}.issubset(reader.fieldnames):
            raise ValueError("CSV эмбеддингов должен содержать колонки 'id' и 'embedding'")
        for row in reader:
            try:
                vec = np.array([float(x) for x in row["embedding"].split()], dtype=np.float32)
            except ValueError:
                continue
            embeddings[row["id"].strip()] = vec

    dims = {len(v) for v in embeddings.values()}
    if len(dims) > 1:
        raise ValueError(f"Разные размерности эмбеддингов: {sorted(dims)}")
    return embeddings


def load_edges_by_day(
    edges_path: str,
    days: Iterable[int],
    use_dst_port: bool,
) -> Dict[int, Dict[str, Counter]]:
    """
    Читает edges.csv (src, dst, dst_port, day, count) и возвращает
    для каждого дня: host -> Counter(item -> count).

    Рёбра в edges.csv ненаправленные, поэтому каждое ребро попадает
    в список соседей обоих хостов.
    """
    edges_path = Path(edges_path)
    if not edges_path.exists():
        raise FileNotFoundError(f"Файл не найден: {edges_path}")

    wanted = set(days)
    result: Dict[int, Dict[str, Counter]] = {d: defaultdict(Counter) for d in wanted}

    with open(edges_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            day = int(row["day"])
            if day not in wanted:
                continue
            count = int(row["count"])
            src, dst, port = row["src"], row["dst"], row["dst_port"]
            if use_dst_port:
                item_for_dst: Item = (dst, port)
                item_for_src: Item = (src, port)
            else:
                item_for_dst = (dst,)
                item_for_src = (src,)
            result[day][src][item_for_dst] += count
            result[day][dst][item_for_src] += count

    return result


def merge_counters(per_day: Dict[str, Counter], ids: Iterable[str]) -> Dict[str, Counter]:
    """Складывает счётчики по дням в один Counter на хост."""
    merged: Dict[str, Counter] = defaultdict(Counter)
    for day_map in per_day.values():
        for host, counter in day_map.items():
            merged[host].update(counter)
    return merged


# ---------------------------------------------------------------------------
# Контекст оценки
# ---------------------------------------------------------------------------

@dataclass
class EvalContext:
    config: EvalConfig
    ids: List[str]                       # хосты, у которых есть эмбеддинг
    index: Dict[str, int]                # host -> позиция в vectors
    vectors: np.ndarray                  # нормализованы, если metric=cosine
    source_neighbors: Dict[str, Counter] # host -> Counter(item -> count) за source-окно
    universe: List[Item]                 # все кандидаты (из source-окна, партнёр с эмбеддингом)
    targets: Dict[str, Dict[str, Counter]]  # split (день) -> host -> Counter(item -> count)

    def similarity_to_all(self, host: str) -> np.ndarray:
        """
        Схожесть хоста со всеми хостами (массив по позициям в ids).
        Cosine: косинус. Euclidean: 1 / (1 + расстояние). Чем больше, тем ближе.
        Сам хост получает -inf, чтобы не попадать в соседи.
        """
        i = self.index[host]
        if self.config.metric == "cosine":
            sims = self.vectors @ self.vectors[i]
        else:
            sims = 1.0 / (1.0 + np.linalg.norm(self.vectors - self.vectors[i], axis=1))
        sims = sims.astype(np.float64)
        sims[i] = -np.inf
        return sims

    def similar_hosts(self, host: str, k: int) -> List[str]:
        """k ближайших хостов по эмбеддингу (сам хост исключается)."""
        i = self.index[host]
        if self.config.metric == "cosine":
            dist = 1.0 - self.vectors @ self.vectors[i]
        else:
            dist = np.linalg.norm(self.vectors - self.vectors[i], axis=1)
        dist[i] = np.inf
        k = min(k, len(self.ids) - 1)
        nearest = np.argpartition(dist, k)[:k]
        nearest = nearest[np.argsort(dist[nearest])]
        return [self.ids[j] for j in nearest]

    def candidates_for(self, host: str) -> List[Item]:
        """Кандидаты для хоста: вся вселенная без него самого."""
        return [it for it in self.universe if it[0] != host]


def build_context(config: EvalConfig) -> EvalContext:
    embeddings = load_embeddings(config.embeddings_path)
    if not embeddings:
        raise ValueError("Эмбеддинги пусты")

    ids = list(embeddings.keys())
    index = {h: i for i, h in enumerate(ids)}
    vectors = np.vstack([embeddings[h] for h in ids]).astype(np.float32)
    if config.metric == "cosine":
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms

    source_by_day = load_edges_by_day(config.edges_path, config.source_days, config.use_dst_port)
    target_by_day = load_edges_by_day(config.edges_path, config.target_days, config.use_dst_port)

    source_neighbors = merge_counters(source_by_day, ids)

    # Вселенная кандидатов: items из source-окна, у которых партнёр имеет эмбеддинг
    universe_set: Set[Item] = set()
    for counter in source_neighbors.values():
        for item in counter:
            if item[0] in index:
                universe_set.add(item)
    universe = sorted(universe_set)

    targets = {f"day_{d}": dict(target_by_day[d]) for d in config.target_days}

    return EvalContext(
        config=config,
        ids=ids,
        index=index,
        vectors=vectors,
        source_neighbors=dict(source_neighbors),
        universe=universe,
        targets=targets,
    )


# ---------------------------------------------------------------------------
# Ранжирование и метрики
# ---------------------------------------------------------------------------

Ranker = Callable[[str, EvalContext], List[Item]]


def rank_by_scores(host: str, ctx: EvalContext, scores: Dict[Item, float]) -> List[Item]:
    """Полное ранжирование кандидатов по убыванию скора (при равенстве — по алфавиту)."""
    candidates = ctx.candidates_for(host)
    return sorted(candidates, key=lambda it: (-scores.get(it, 0.0), it))


def partner_scores_to_items(
    host: str, ctx: EvalContext, partner_scores: Dict[str, float]
) -> Dict[Item, float]:
    """
    Переводит скор по хостам в скор по items.
    Если учитывается порт, все порты одного партнёра получают одинаковый скор.
    """
    return {it: partner_scores.get(it[0], 0.0) for it in ctx.candidates_for(host)}


def evaluate(name: str, ranker: Ranker, ctx: EvalContext) -> EvalResult:
    """
    Общий цикл оценки. Для каждого split (дня) и каждого хоста с эмбеддингом:
    - целевые items = то, что хост реально делал в этот день;
    - пропускаем целевые items, которых нет в кандидатах (они идут в coverage);
    - считаем reciprocal rank (по полному ранжированию) и recall@top_n.
    """
    top_n = ctx.config.top_n
    universe_set = set(ctx.universe)

    all_rr: List[float] = []
    all_recall: List[float] = []
    per_split: Dict[str, Dict[str, float]] = {}
    total_items = 0
    covered_items = 0

    for split, host_map in ctx.targets.items():
        rrs: List[float] = []
        recalls: List[float] = []
        split_total = 0
        split_covered = 0

        for host, counter in host_map.items():
            if host not in ctx.index:
                continue  # хосты без эмбеддинга не оцениваем

            true_items = set(counter.keys())
            split_total += len(true_items)
            relevant = true_items & universe_set
            split_covered += len(relevant)
            if not relevant:
                continue  # недостижимо для любого метода; учтено в coverage

            ranking = ranker(host, ctx)
            positions = {it: pos for pos, it in enumerate(ranking, start=1)}
            ranks = sorted(positions[it] for it in relevant if it in positions)
            if not ranks:
                continue

            rrs.append(1.0 / ranks[0])
            found_top = sum(1 for r in ranks if r <= top_n)
            recalls.append(found_top / len(relevant))

        total_items += split_total
        covered_items += split_covered
        all_rr.extend(rrs)
        all_recall.extend(recalls)

        per_split[split] = {
            "mrr": float(np.mean(rrs)) if rrs else 0.0,
            "recall_at_n": float(np.mean(recalls)) if recalls else 0.0,
            "n_hosts": float(len(rrs)),
            "coverage": split_covered / split_total if split_total else 0.0,
        }

    return EvalResult(
        name=name,
        mrr=float(np.mean(all_rr)) if all_rr else 0.0,
        recall_at_n=float(np.mean(all_recall)) if all_recall else 0.0,
        n_hosts=len(all_rr),
        coverage=covered_items / total_items if total_items else 0.0,
        per_split=per_split,
        config=asdict(ctx.config),
    )


# ---------------------------------------------------------------------------
# Бейзлайны (считаются на тех же окнах и тех же кандидатах)
# ---------------------------------------------------------------------------

def history_ranker(host: str, ctx: EvalContext) -> List[Item]:
    """«Как было раньше»: сначала те, с кем хост общался в source-окне, по частоте."""
    scores = {it: float(c) for it, c in ctx.source_neighbors.get(host, Counter()).items()}
    return rank_by_scores(host, ctx, scores)


def popularity_ranker(host: str, ctx: EvalContext) -> List[Item]:
    """«Популярные»: сначала хосты с наибольшим числом контактов в source-окне."""
    popularity: Counter = Counter()
    for counter in ctx.source_neighbors.values():
        for it, c in counter.items():
            popularity[it] += c
    return rank_by_scores(host, ctx, {it: float(c) for it, c in popularity.items()})


def random_ranker(host: str, ctx: EvalContext) -> List[Item]:
    """Жребий: случайный порядок, зафиксированный seed'ом и именем хоста."""
    rng = np.random.default_rng(ctx.config.random_seed ^ zlib.crc32(host.encode()))
    candidates = ctx.candidates_for(host)
    order = rng.permutation(len(candidates))
    return [candidates[i] for i in order]


BASELINES: Dict[str, Ranker] = {
    "baseline_history": history_ranker,
    "baseline_popularity": popularity_ranker,
    "baseline_random": random_ranker,
}


def run_baselines(ctx: EvalContext) -> List[EvalResult]:
    return [evaluate(name, fn, ctx) for name, fn in BASELINES.items()]


def lift(method: EvalResult, results_baselines: Sequence[EvalResult]) -> float:
    """
    Прирост над лучшим бейзлайном по MRR:
    (метод − лучший бейзлайн) / лучший бейзлайн.
    0 — не лучше простых способов; > 0 — добавляет информацию.
    """
    best = max(r.mrr for r in results_baselines)
    if best == 0:
        return float("nan")
    return (method.mrr - best) / best