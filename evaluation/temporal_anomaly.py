"""Согласованность изменений эмбеддингов и поведения между окнами."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr
from sklearn.preprocessing import normalize

from evaluation.knn_consistency import build_reference_features


def _active_hosts(flows: pd.DataFrame) -> set[str]:
    return set(flows["src"].astype(str)) | set(flows["dst"].astype(str))


def _cosine_change(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Расстояние 1 - cosine similarity для пар строк."""
    a = normalize(a, norm="l2")
    b = normalize(b, norm="l2")
    return np.clip(1.0 - np.sum(a * b, axis=1), 0.0, 2.0)


def temporal_anomaly(
    emb_train: pd.DataFrame,
    emb_test: pd.DataFrame,
    train_flows: pd.DataFrame,
    test_flows: pd.DataFrame,
    target_hosts: list[str],
    *,
    shared_coordinate_system: bool = False,
    seed: int = 0,
    top_fraction: float = 0.10,
) -> dict:
    """Сопоставить изменения эмбеддингов и поведения на общих активных хостах.

    Для несвязанных систем координат Procrustes обучается на alignment subset,
    а расстояния оцениваются только на held-out subset.
    """
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction должен быть в интервале (0, 1]")

    for name, frame in (("train", train_flows), ("test", test_flows)):
        missing = {"src", "dst", "dst_port", "proto", "bytes", "packets", "ts"} - set(frame.columns)
        if missing:
            raise ValueError(f"{name}_flows: отсутствуют колонки {sorted(missing)}")

    train_flows = train_flows.copy()
    test_flows = test_flows.copy()
    for frame in (train_flows, test_flows):
        frame["src"] = frame["src"].astype(str)
        frame["dst"] = frame["dst"].astype(str)

    target = set(map(str, target_hosts))
    active_train = _active_hosts(train_flows)
    active_test = _active_hosts(test_flows)
    common = sorted(target & active_train & active_test)

    excluded = {
        "not_active_train": len(target - active_train),
        "not_active_test": len(target - active_test),
        "not_active_both": len(target - (active_train & active_test)),
    }

    result = {
        "n_target_hosts": len(target),
        "n_common_active_hosts": len(common),
        "excluded_hosts": excluded,
        "shared_coordinate_system": bool(shared_coordinate_system),
        "alignment": "none" if shared_coordinate_system else "procrustes",
        "rank_correlation": None,
        "rank_correlation_pvalue": None,
        "top_k": None,
        "top_k_overlap": None,
        "top_k_overlap_fraction": None,
        "n_scored_hosts": 0,
        "per_host": {},
    }
    if len(common) < 10:
        result["unavailable"] = "меньше 10 хостов активны в обоих окнах"
        return result

    # Строим признаки только для общей активной когорты: нулевые профили
    # отсутствующих в окне хостов не попадут в сравнение.
    train_features, fitted = build_reference_features(
        train_flows, hosts=common, return_fitted=True
    )
    test_features = build_reference_features(
        test_flows, hosts=common, fitted=fitted
    )
    behavior_change = np.linalg.norm(
        test_features.loc[common].to_numpy() -
        train_features.loc[common].to_numpy(),
        axis=1,
    )

    # Оставляем только ID, которые присутствуют в обоих embedding-файлах.
    emb_train = emb_train.copy()
    emb_test = emb_test.copy()
    emb_train.index = emb_train.index.astype(str)
    emb_test.index = emb_test.index.astype(str)
    if not emb_train.index.is_unique or not emb_test.index.is_unique:
        raise ValueError("повторяющиеся host_id после приведения к строкам")

    common = [
        h for h in common
        if h in emb_train.index and h in emb_test.index
    ]
    result["n_common_active_hosts"] = len(common)
    result["excluded_hosts"]["missing_embeddings"] = (
        len(target & active_train & active_test) - len(common)
    )
    if len(common) < 10:
        result["unavailable"] = "меньше 10 общих активных хостов с эмбеддингами"
        return result

    A = emb_train.loc[common].to_numpy(dtype=float)
    B = emb_test.loc[common].to_numpy(dtype=float)
    if (not np.isfinite(A).all()) or (not np.isfinite(B).all()):
        raise ValueError("эмбеддинги содержат NaN или бесконечные значения")
    if A.shape != B.shape or A.ndim != 2 or A.shape[1] == 0:
        raise ValueError("размерности train и test эмбеддингов не совпадают")

    behavior = pd.Series(behavior_change, index=common).loc[common].to_numpy()

    if shared_coordinate_system:
        embedding_change = _cosine_change(A, B)
        scored_hosts = common
    else:
        # Отложенная часть не участвует в подборе преобразования.
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(common))
        n_align = len(common) // 2
        align_idx, eval_idx = order[:n_align], order[n_align:]
        if n_align < 3 or len(eval_idx) < 5:
            result["unavailable"] = "недостаточно хостов для Procrustes и отложенной оценки"
            return result

        mu_a = A[align_idx].mean(axis=0)
        mu_b = B[align_idx].mean(axis=0)
        A_centered = A - mu_a
        B_centered = B - mu_b
        rotation, _ = orthogonal_procrustes(
            B_centered[align_idx], A_centered[align_idx]
        )
        aligned_test = B_centered[eval_idx] @ rotation
        embedding_change = _cosine_change(
            A_centered[eval_idx], aligned_test
        )
        scored_hosts = [common[i] for i in eval_idx]
        behavior = pd.Series(behavior, index=common).loc[scored_hosts].to_numpy()
        result["n_alignment_hosts"] = int(len(align_idx))

    if len(embedding_change) < 5:
        result["unavailable"] = "недостаточно хостов для рангового сравнения"
        return result

    corr = spearmanr(embedding_change, behavior)
    rho = float(corr.statistic)
    pvalue = float(corr.pvalue)
    result["rank_correlation"] = rho if np.isfinite(rho) else None
    result["rank_correlation_pvalue"] = pvalue if np.isfinite(pvalue) else None

    k = max(1, int(np.ceil(len(scored_hosts) * top_fraction)))
    emb_top = set(np.argsort(embedding_change)[-k:])
    behavior_top = set(np.argsort(behavior)[-k:])
    overlap = len(emb_top & behavior_top)

    result.update({
        "n_scored_hosts": len(scored_hosts),
        "top_k": k,
        "top_k_overlap": overlap,
        "top_k_overlap_fraction": overlap / k,
        "per_host": {
            host: {
                "embedding_change": float(e),
                "behavior_change": float(b),
            }
            for host, e, b in zip(scored_hosts, embedding_change, behavior)
        },
    })
    return result