"""Замороженные соседи по логам — общий эталон для всех методов.

Это ПРИБЛИЖЕНИЕ поведенческой похожести, а не правда о ролях. Строится один раз из эталонных
признаков и одинаков для всех методов. Лучше брать признаки более позднего окна (#45).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.knn_consistency import (
    neighborhood_metrics,
    summarize,
    top_k_neighbors,
)


COLUMNS = ("reference_window", "host_id", "neighbor_id", "rank", "split")


def save_frozen_neighbors(
    reference: pd.DataFrame,
    output: Path,
    *,
    reference_window: str,
    k_max: int = 20,
    split: str = "test",
) -> Path:
    """Воспроизводимые top-k эталонные соседи без самого хоста.

    В reference host_id — индекс. Строки сортируются по host_id, чтобы порядок файла и ничьи
    не зависели от порядка входных строк.
    """
    if not reference_window.strip():
        raise ValueError("нужно указать reference_window")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split должен быть train, validation или test")
    if reference.index.has_duplicates or reference.index.isna().any():
        raise ValueError("host_id эталона должны быть непустыми и уникальными")
    if len(reference) <= k_max or k_max < 1:
        raise ValueError("нужно 1 <= k_max < числа хостов")
    if not all(isinstance(h, str) and h.strip() for h in reference.index):
        raise ValueError("host_id эталона должны быть непустыми строками")
    if not np.isfinite(reference.to_numpy(dtype=float)).all():
        raise ValueError("признаки эталона должны быть конечными числами")

    reference = reference.sort_index()
    hosts = reference.index.tolist()
    nearest = top_k_neighbors(reference.to_numpy(dtype=float), k_max)
    rows = [
        (reference_window, host, hosts[int(neighbor)], rank, split)
        for i, host in enumerate(hosts)
        for rank, neighbor in enumerate(nearest[i], start=1)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(output, index=False)
    return output


def read_frozen_neighbors(
    path: Path, hosts: list[str], *, k_max: int, split: str = "test"
) -> np.ndarray:
    """Проверить полное покрытие и вернуть индексы соседей в ТОЧНОМ порядке эмбеддинга.

    Неполные списки соседей не превращаются молча в «непохожих».
    """
    if len(hosts) != len(set(hosts)) or not hosts:
        raise ValueError("хосты эмбеддинга должны быть непустыми и уникальными")
    if k_max < 1 or k_max >= len(hosts):
        raise ValueError("нужно 1 <= k_max < числа хостов")
    df = pd.read_csv(
        path,
        dtype={"reference_window": "string", "host_id": "string",
               "neighbor_id": "string", "split": "string", "rank": "string"},
    )
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"в neighbors.csv нет колонок: {sorted(missing)}")
    if df.empty or df[list(COLUMNS)].isna().any().any():
        raise ValueError("neighbors.csv пуст или в нём пропуски")
    df = df.loc[df["split"] == split].copy()
    if df.empty:
        raise ValueError(f"в neighbors.csv нет строк split={split}")
    if df["reference_window"].nunique() != 1:
        raise ValueError("в одной оценке нельзя смешивать несколько окон эталона")
    if df["host_id"].str.strip().eq("").any() or df["neighbor_id"].str.strip().eq("").any():
        raise ValueError("пустой host_id в neighbors.csv")
    valid = set(hosts)
    if not set(df["host_id"]).issubset(valid) or not set(df["neighbor_id"]).issubset(valid):
        raise ValueError("в neighbors.csv есть хосты вне когорты эмбеддинга")
    if set(df["host_id"]) != valid:
        raise ValueError("у части хостов эмбеддинга нет эталонных соседей")
    if (df["host_id"] == df["neighbor_id"]).any():
        raise ValueError("хост не может быть эталонным соседом сам себе")
    rank = pd.to_numeric(df["rank"], errors="coerce")
    if rank.isna().any() or (rank < 1).any() or (rank != np.floor(rank)).any():
        raise ValueError("ранги соседей должны быть целыми > 0")
    df["rank"] = rank.astype(int)
    if df.duplicated(["host_id", "rank"]).any():
        raise ValueError("повторяющиеся ранги соседей у хоста")
    if df.duplicated(["host_id", "neighbor_id"]).any():
        raise ValueError("повторяющиеся эталонные соседи у хоста")

    lookup = {host: i for i, host in enumerate(hosts)}
    ordered = np.empty((len(hosts), k_max), dtype=int)
    for i, host in enumerate(hosts):
        rows = df[df["host_id"] == host].sort_values("rank")
        ranks = rows["rank"].tolist()
        if ranks != list(range(1, len(rows) + 1)) or len(rows) < k_max:
            raise ValueError(f"у {host} неполные или непоследовательные ранги соседей")
        ordered[i] = [lookup[h] for h in rows["neighbor_id"].iloc[:k_max]]
    return ordered


def evaluate_frozen_neighbors(
    embeddings: pd.DataFrame, path: Path,
    ks: tuple[int, ...] = (5, 10, 20),
    *, split: str = "test", return_per_host: bool = False,
):
    """Consistency, Jaccard и nDCG по замороженным соседям.

    return_per_host=True — ещё {k: таблица по хостам} (индекс — порядок эмбеддинга).
    """
    if embeddings.index.has_duplicates or not ks:
        raise ValueError("хосты эмбеддинга должны быть уникальны, а список k — непуст")
    highest = max(ks)
    if any(k < 1 or k >= len(embeddings) for k in ks):
        raise ValueError("каждое k должно быть от 1 до числа хостов - 1")
    ref = read_frozen_neighbors(path, list(embeddings.index), k_max=highest, split=split)
    emb = embeddings.to_numpy(dtype=float)
    rows, per_host = [], {}
    for k in ks:
        eb_nn = top_k_neighbors(emb, k)
        ph = neighborhood_metrics(ref[:, :k], eb_nn)
        ph.index = embeddings.index
        per_host[k] = ph
        stats = summarize(ph)
        stats.index = pd.MultiIndex.from_product([[k], stats.index], names=["k", "metric"])
        rows.append(stats)
    summary = pd.concat(rows)
    return (summary, per_host) if return_per_host else summary
