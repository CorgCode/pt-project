"""Детектор аномалий: дрейф хоста и редкие переходы между кластерами.

Идея: фиксируем «нормальную» структуру сети по базовому окну (эмбеддер + KMeans
обучены на окне 0). Для каждого следующего окна представляем хосты в ТОМ ЖЕ
пространстве и смотрим:
  * насколько далеко хост сместился (embedding drift);
  * сменил ли он кластер и насколько редок такой переход по сети в целом.
Комбинация даёт аномальность. Подсвечиваются хосты с резкой сменой поведения
(например, рабочая станция, начавшая сканировать сеть или обслуживать сервис).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .clustering import cluster as _cluster
from .clustering import select_k
from .embeddings import FeatureEmbedder


def _transition_rarity(traj: dict[str, list[tuple[int, int]]]) -> dict[tuple[int, int], float]:
    """Оценить -log P(c_prev -> c_next) по всем хостам/окнам."""
    counts: dict[tuple[int, int], int] = {}
    from_totals: dict[int, int] = {}
    for seq in traj.values():
        for (_, c0), (_, c1) in zip(seq, seq[1:]):
            counts[(c0, c1)] = counts.get((c0, c1), 0) + 1
            from_totals[c0] = from_totals.get(c0, 0) + 1
    rarity: dict[tuple[int, int], float] = {}
    for (c0, c1), n in counts.items():
        p = n / max(1, from_totals[c0])
        rarity[(c0, c1)] = float(-np.log(p + 1e-9))
    return rarity


def detect_anomalies(
    window_feats: list[pd.DataFrame],
    k_min: int = 4,
    k_max: int = 14,
    z_threshold: float = 3.0,
    top_n: int = 15,
    alpha: float = 1.0,
    seed: int = 42,
) -> dict:
    """Вернуть ранжированные аномалии и траектории кластеров по окнам."""
    if len(window_feats) < 2:
        return {"scores": pd.DataFrame(), "trajectories": {}, "n_windows": len(window_feats)}

    # Пространство представления и поведенческие кластеры фиксируем по ПУЛУ всех
    # окон: одно глобальное разбиение на поведенческие типы. Тогда смена поведения
    # хоста проявляется как ПЕРЕХОД между кластерами от окна к окну, а редкость
    # такого перехода по сети в целом задаёт его аномальность.
    pooled = pd.concat(window_feats, axis=0)
    embedder = FeatureEmbedder(seed=seed).fit(pooled)
    pooled_emb = embedder.transform(pooled)
    k, _ = select_k(pooled_emb.values, k_min, k_max, seed)
    _, km = _cluster(pooled_emb.values, k, seed)

    # Представления и назначение кластеров для всех окон.
    emb_by_window: list[pd.DataFrame] = []
    cluster_by_window: list[pd.Series] = []
    for feats in window_feats:
        emb = embedder.transform(feats)
        lab = pd.Series(km.predict(emb.values), index=emb.index)
        emb_by_window.append(emb)
        cluster_by_window.append(lab)

    # Траектории (окно, кластер) и векторов по хостам.
    devices = sorted(set().union(*[set(e.index) for e in emb_by_window]))
    traj: dict[str, list[tuple[int, int]]] = {}
    vec_traj: dict[str, list[tuple[int, np.ndarray]]] = {}
    for dev in devices:
        seq, vseq = [], []
        for w, (emb, lab) in enumerate(zip(emb_by_window, cluster_by_window)):
            if dev in lab.index:
                seq.append((w, int(lab.loc[dev])))
                vseq.append((w, emb.loc[dev].values))
        traj[dev] = seq
        vec_traj[dev] = vseq

    rarity = _transition_rarity(traj)

    # Все величины дрейфа — для нормировки в z-оценку. Используем робастную
    # оценку (медиана/MAD): она устойчива к самим выбросам-аномалиям и не даёт
    # паре «взрывных» дрейфов сжать шкалу для остальных.
    all_drifts = []
    for vseq in vec_traj.values():
        for (_, v0), (_, v1) in zip(vseq, vseq[1:]):
            all_drifts.append(float(np.linalg.norm(v1 - v0)))
    arr = np.asarray(all_drifts) if all_drifts else np.array([0.0])
    mu = float(np.median(arr))
    sd = 1.4826 * float(np.median(np.abs(arr - mu)))
    sd = sd if sd > 1e-9 else (float(np.std(arr)) or 1.0)

    rows = []
    for dev in devices:
        seq, vseq = traj[dev], vec_traj[dev]
        if len(seq) < 2:
            continue
        best = None
        for ((_, c0), (w1, c1)), ((_, v0), (_, v1)) in zip(
            zip(seq, seq[1:]), zip(vseq, vseq[1:])
        ):
            drift = float(np.linalg.norm(v1 - v0))
            z_drift = (drift - mu) / sd
            r = rarity.get((c0, c1), 0.0)
            score = max(0.0, z_drift) + alpha * r
            if best is None or score > best["score"]:
                best = {"score": score, "window": int(w1), "from_cluster": int(c0),
                        "to_cluster": int(c1), "drift": drift, "z_drift": z_drift,
                        "rarity": r, "cluster_changed": bool(c0 != c1)}
        row = {"device": dev, **best,
               "trajectory": [c for (_, c) in seq]}
        rows.append(row)

    scores = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    if not scores.empty:
        # Аномалия — резкий сдвиг поведения (большой дрейф) ИЛИ редкая смена кластера.
        scores["is_anomaly"] = (scores["score"] >= z_threshold) | (
            scores["cluster_changed"] & (scores["rarity"] >= 2.0))
        # Гарантируем, что топ-N попадёт в выдачу для оператора.
        scores["in_top"] = False
        scores.loc[: top_n - 1, "in_top"] = True

    return {
        "scores": scores,
        "trajectories": traj,
        "n_windows": len(window_feats),
        "baseline_k": int(k),
    }
