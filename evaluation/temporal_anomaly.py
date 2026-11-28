"""Обнаружение аномальных изменений хоста между временными окнами (issue #21).

Ключевой сигнал — переходы между кластерами в разные окна: хосты, которые были в
одной группе, а в следующем окне оказались в другой, в шуме или образовали
нестабильное сопоставление. Метки HDBSCAN произвольны, поэтому кластеры разных
окон сначала сопоставляются по максимальному пересечению общих хостов.

Это эвристика «подозрительного изменения», а не доказательство атаки: переход
может означать и легитимную смену поведения. Интерпретация требует контекста.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from evaluation.clustering import (DEFAULT_METRIC, DEFAULT_MIN_CLUSTER_SIZE,
                                   DEFAULT_MIN_SAMPLES, cluster_embeddings)

NOISE = -1
TRANSITION_TYPES = (
    "stable", "changed", "to_noise", "from_noise", "noise_noise", "missing",
)
COLUMNS = [
    "status", "n_common_hosts", "n_compared_hosts", "n_stable", "n_changed",
    "n_to_noise", "n_from_noise", "n_noise_noise", "n_anomalous_transitions",
    "transition_rate", "noise_emergence_rate", "delta_transition_rate_vs_random",
]


def _match_clusters(before: np.ndarray, after: np.ndarray) -> dict[int, int]:
    """Сопоставить метки кластеров окон по максимуму пересечения общих хостов.

    Шум (-1) не сопоставляется с обычными кластерами. Жадное one-to-one
    сопоставление по убыванию пересечения даёт детерминированный и понятный
    результат при слияниях/перемещениях кластеров.
    """
    b_labels = sorted(set(int(x) for x in before[before != NOISE]))
    a_labels = sorted(set(int(x) for x in after[after != NOISE]))
    if not b_labels or not a_labels:
        return {}

    overlap = np.zeros((len(b_labels), len(a_labels)), dtype=int)
    valid = (before != NOISE) & (after != NOISE)
    for b, a in zip(before[valid], after[valid]):
        overlap[b_labels.index(int(b)), a_labels.index(int(a))] += 1

    candidates = [(int(overlap[r, c]), b_labels[r], a_labels[c], r, c)
                  for r in range(len(b_labels)) for c in range(len(a_labels))
                  if overlap[r, c] > 0]
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    used_b, used_a, mapping = set(), set(), {}
    for count, b, a, r, c in candidates:
        if r in used_b or c in used_a:
            continue
        mapping[b] = a
        used_b.add(r)
        used_a.add(c)
    return mapping


def _transition_type(before: int, after: int, mapping: dict[int, int]) -> str:
    if before == NOISE and after == NOISE:
        return "noise_noise"
    if before == NOISE:
        return "from_noise"
    if after == NOISE:
        return "to_noise"
    if mapping.get(before) == after:
        return "stable"
    return "changed"


def detect_cluster_transitions(
    emb_before: pd.DataFrame,
    emb_after: pd.DataFrame,
    *,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int | None = DEFAULT_MIN_SAMPLES,
    metric: str = DEFAULT_METRIC,
) -> tuple[dict, pd.DataFrame]:
    """Сравнить кластерные назначения двух окон и вернуть сводку + per-host transitions.

    Возвращает (summary, transitions), где transitions индексирован host_id и содержит
    колонки cluster_before, cluster_after, mapped_after, transition_type, is_anomalous.
    """
    _, assign_before = cluster_embeddings(emb_before, min_cluster_size=min_cluster_size,
                                          min_samples=min_samples, metric=metric)
    _, assign_after = cluster_embeddings(emb_after, min_cluster_size=min_cluster_size,
                                         min_samples=min_samples, metric=metric)

    common = assign_before.index.intersection(assign_after.index).sort_values()
    before = assign_before.loc[common, "cluster"].to_numpy(dtype=int)
    after = assign_after.loc[common, "cluster"].to_numpy(dtype=int)
    mapping = _match_clusters(before, after)

    types = np.array([_transition_type(b, a, mapping) for b, a in zip(before, after)], dtype=object)
    mapped = np.array([mapping.get(int(b), NOISE) if b != NOISE else NOISE for b in before], dtype=int)
    is_anomalous = np.isin(types, ["changed", "to_noise", "from_noise"])

    transitions = pd.DataFrame({
        "cluster_before": before,
        "cluster_after": after,
        "mapped_after": mapped,
        "transition_type": types,
        "is_anomalous": is_anomalous,
    }, index=common)
    transitions.index.name = "host_id"

    compared = len(common)
    counts = {t: int((types == t).sum()) for t in TRANSITION_TYPES if t != "missing"}
    anomalous = int(is_anomalous.sum())
    summary = {
        "status": "ok",
        "n_common_hosts": compared,
        "n_compared_hosts": compared,
        "n_stable": counts["stable"],
        "n_changed": counts["changed"],
        "n_to_noise": counts["to_noise"],
        "n_from_noise": counts["from_noise"],
        "n_noise_noise": counts["noise_noise"],
        "n_anomalous_transitions": anomalous,
        "transition_rate": anomalous / compared if compared else None,
        "noise_emergence_rate": counts["to_noise"] / compared if compared else None,
        "cluster_mapping_size": len(mapping),
    }
    return summary, transitions


def temporal_anomaly_report(
    embeddings: dict[str, dict[str, pd.DataFrame]],
    *,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int | None = DEFAULT_MIN_SAMPLES,
    metric: str = DEFAULT_METRIC,
) -> pd.DataFrame:
    """Отчёт по переходам кластеров train→test для каждого метода.

    Методы без нужных окон или с недостаточной общей когортой получают статус вместо
    исключения: один метод не должен ронять весь прогон.
    """
    rows = []
    for method, windows in sorted(embeddings.items()):
        base = {"method": method}
        if "train" not in windows or "test" not in windows:
            rows.append({**base, "status": "нет эмбеддингов для обоих окон"})
            continue
        train, test = windows["train"], windows["test"]
        if not train.index.is_unique or not test.index.is_unique:
            rows.append({**base, "status": "недоступно: повторяющиеся host_id"})
            continue
        common = train.index.intersection(test.index)
        if len(common) < min_cluster_size:
            rows.append({**base, "status": "недостаточно общих хостов", "n_common_hosts": int(len(common))})
            continue
        try:
            summary, _ = detect_cluster_transitions(
                train.loc[common].sort_index(), test.loc[common].sort_index(),
                min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric)
        except ValueError as exc:
            rows.append({**base, "status": f"недоступно: {exc}", "n_common_hosts": int(len(common))})
            continue
        rows.append({**base, **summary})

    if not rows:
        return pd.DataFrame(columns=["method", *COLUMNS]).set_index("method")
    result = pd.DataFrame(rows).set_index("method")
    for c in COLUMNS:
        if c not in result.columns:
            result[c] = np.nan
    result = result[COLUMNS]

    if "random" in result.index and result.at["random", "status"] == "ok":
        valid = result["status"] == "ok"
        chance = result.at["random", "transition_rate"]
        if pd.notna(chance):
            result.loc[valid, "delta_transition_rate_vs_random"] = result.loc[valid, "transition_rate"] - chance
    return result
