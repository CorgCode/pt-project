"""Кластеризация эмбеддингов хостов и оценка качества разбиений (issue #21).

Хосты группируются HDBSCAN: число групп заранее неизвестно, а хосты, не похожие
ни на одну группу, получают метку шума ``-1``. Для эмбеддингов по умолчанию
используется косинусная близость (L2-нормировка + евклидово расстояние).

Silhouette и Davies–Bouldin описывают только геометрию разбиения: высокие значения
НЕ доказывают, что кластеры соответствуют реальным ролям хостов. Поэтому они
сравниваются со случайным разбиением той же формы (те же размеры кластеров и та же
доля шума), а при наличии внешних меток — ещё и с ARI / NMI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.preprocessing import normalize

NOISE = -1
DEFAULT_METRIC = "cosine"
DEFAULT_MIN_CLUSTER_SIZE = 5
DEFAULT_MIN_SAMPLES = 5
RANDOM_DRAWS = 5

CLUSTERING_CONFIG = {
    "algorithm": "HDBSCAN",
    "metric": "cosine через L2-нормировку и euclidean",
    "min_cluster_size": DEFAULT_MIN_CLUSTER_SIZE,
    "min_samples": DEFAULT_MIN_SAMPLES,
    "random_baseline": "перестановка меток: те же размеры кластеров и доля шума",
    "random_draws": RANDOM_DRAWS,
}


def _prepare(emb: pd.DataFrame, *, metric: str, min_cluster_size: int) -> np.ndarray:
    """Проверить эмбеддинги и подготовить матрицу под выбранную метрику."""
    if not isinstance(emb, pd.DataFrame) or emb.empty:
        raise ValueError("emb должен быть непустым DataFrame")
    if emb.index.has_duplicates or emb.index.isna().any():
        raise ValueError("повторяющиеся или пустые host_id")
    if min_cluster_size < 2:
        raise ValueError("min_cluster_size должен быть >= 2")
    if metric not in ("cosine", "euclidean"):
        raise ValueError("metric должен быть 'cosine' или 'euclidean'")
    try:
        X = emb.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("эмбеддинги должны содержать только числа") from exc
    if X.ndim != 2 or X.shape[1] == 0 or not np.isfinite(X).all():
        raise ValueError("эмбеддинги должны быть конечной непустой матрицей")
    if len(X) < min_cluster_size:
        raise ValueError("число хостов меньше min_cluster_size")
    # Нулевой вектор остаётся нулевым: при косинусе он равноудалён от всех хостов.
    return normalize(X) if metric == "cosine" else X


def _finite(value) -> float | None:
    """sklearn может вернуть inf/nan на вырожденных разбиениях — в отчёт их не пускаем."""
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _external(labels: np.ndarray, expected: pd.Series | None) -> dict:
    """ARI и NMI по внешним меткам; шум и хосты без метки исключаются из обеих сторон."""
    out = {"ari": None, "nmi": None}
    if expected is None:
        return out
    keep = expected.notna().to_numpy() & (labels != NOISE)
    if keep.sum() < 2:
        return out
    truth = expected[keep].to_numpy()
    if len(np.unique(truth)) < 2 or len(np.unique(labels[keep])) < 2:
        return out
    out["ari"] = float(adjusted_rand_score(truth, labels[keep]))
    out["nmi"] = float(normalized_mutual_info_score(truth, labels[keep]))
    return out


def _quality(X: np.ndarray, labels: np.ndarray, expected: pd.Series | None = None) -> dict:
    """Метрики одного разбиения. X уже подготовлен: для косинуса — нормированный."""
    clustered = labels != NOISE
    n_clusters = int(len(np.unique(labels[clustered])))
    out = {
        "n_clusters": n_clusters,
        "n_noise": int((~clustered).sum()),
        "noise_fraction": float((~clustered).mean()),
        "clustered_fraction": float(clustered.mean()),
        "silhouette": None,
        "davies_bouldin": None,
        **_external(labels, expected),
    }
    if n_clusters >= 2 and int(clustered.sum()) > n_clusters:
        y = labels[clustered]
        out["silhouette"] = _finite(silhouette_score(X[clustered], y, metric="euclidean"))
        out["davies_bouldin"] = _finite(davies_bouldin_score(X[clustered], y))
    return out


def cluster_embeddings(
    emb: pd.DataFrame,
    *,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int | None = DEFAULT_MIN_SAMPLES,
    metric: str = DEFAULT_METRIC,
) -> tuple[dict, pd.DataFrame]:
    """Сгруппировать хосты; вернуть метрики разбиения и назначения ``host_id -> cluster``.

    Шум HDBSCAN обозначается меткой ``-1``. Silhouette и Davies–Bouldin считаются
    только по не-шумовым хостам и только если кластеров хотя бы два.
    """
    X = _prepare(emb, metric=metric, min_cluster_size=min_cluster_size)
    labels = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                     metric="euclidean").fit_predict(X).astype(int)
    details = {
        "algorithm": "HDBSCAN",
        "metric": metric,
        "min_cluster_size": int(min_cluster_size),
        "min_samples": min_samples,
        "n_hosts": int(len(labels)),
        **_quality(X, labels),
    }
    assignments = pd.DataFrame({"cluster": labels}, index=emb.index.astype(str))
    assignments.index.name = "host_id"
    return details, assignments


def _labeled(assignments: pd.DataFrame, expected: pd.Series) -> dict:
    """Совместить назначения и внешние метки по host_id и посчитать ARI/NMI."""
    if not assignments.index.is_unique or not expected.index.is_unique:
        raise ValueError("повторяющиеся host_id")
    if "cluster" not in assignments.columns:
        raise ValueError("в assignments нет колонки 'cluster'")
    if set(assignments.index.astype(str)) != set(expected.index.astype(str)):
        raise ValueError("разные host_id в assignments и expected")
    aligned = expected.astype(object).reindex(assignments.index.astype(str))
    out = _external(assignments["cluster"].to_numpy(dtype=int), aligned)
    if out["ari"] is None:
        raise ValueError("недостаточно хостов без шума и двух классов для внешних метрик")
    return out


def clustering_ari(assignments: pd.DataFrame, expected: pd.Series) -> float:
    """ARI по известным меткам; шум не считается отдельным кластером."""
    return _labeled(assignments, expected)["ari"]


def clustering_nmi(assignments: pd.DataFrame, expected: pd.Series) -> float:
    """NMI по известным меткам; шум не считается отдельным кластером."""
    return _labeled(assignments, expected)["nmi"]


def random_baseline(
    emb: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    metric: str = DEFAULT_METRIC,
    expected: pd.Series | None = None,
    seed: int = 0,
    n_draws: int = RANDOM_DRAWS,
) -> dict:
    """Точка отсчёта: случайно перемешать назначения, сохранив размеры кластеров и шум.

    Геометрия эмбеддингов и число кластеров остаются теми же — меняется только то,
    к какому кластеру отнесён хост. Положительная разница с этим бейзлайном означает,
    что найденная структура действительно есть в данных, а не артефакт числа групп.
    """
    if n_draws < 1:
        raise ValueError("n_draws должен быть >= 1")
    X = _prepare(emb, metric=metric, min_cluster_size=2)
    if list(assignments.index.astype(str)) != list(emb.index.astype(str)):
        raise ValueError("assignments и emb должны содержать одних хостов в том же порядке")
    labels = assignments["cluster"].to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    draws = [_quality(X, rng.permutation(labels), expected) for _ in range(n_draws)]
    out = {"seed": int(seed), "n_draws": int(n_draws)}
    for key in ("silhouette", "davies_bouldin", "ari", "nmi"):
        values = [d[key] for d in draws if d[key] is not None]
        out[key] = float(np.mean(values)) if values else None
    return out


def _gain(actual, baseline) -> float | None:
    if actual is None or baseline is None:
        return None
    return float(actual - baseline)


def compare_with_random(details: dict, baseline: dict) -> dict:
    """Добавить к метрикам разбиения дельты против случайного разбиения.

    Для silhouette, ARI и NMI больше — лучше, для Davies–Bouldin — меньше,
    поэтому в дельте для DB вычитание сделано наоборот. Положительная дельта
    всегда означает «лучше случайного разбиения той же формы».
    """
    out = {**details, "random_baseline": baseline}
    for key in ("silhouette", "ari", "nmi"):
        out[f"delta_{key}_vs_random"] = _gain(details.get(key), baseline.get(key))
    out["delta_davies_bouldin_vs_random"] = _gain(baseline.get("davies_bouldin"),
                                                 details.get("davies_bouldin"))
    return out
