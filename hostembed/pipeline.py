"""Сквозной пайплайн: данные -> признаки -> embeddings -> кластеры ->
поиск похожих + объяснения -> аномалии -> артефакты (JSON) для веб-сервиса."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from . import lanl_io
from .anomaly import detect_anomalies
from .clustering import Explainer, Similarity, cluster, cluster_profiles, select_k
from .config import PipelineConfig
from .embeddings import FeatureEmbedder, combined_embedding, graph_embedding
from .evaluation import evaluate_method
from .features import extract_features


def _project_2d(emb_df: pd.DataFrame, seed: int) -> np.ndarray:
    """2D-проекция для UI: t-SNE с откатом на PCA."""
    x = emb_df.values
    n = x.shape[0]
    if n >= 10:
        try:
            from sklearn.manifold import TSNE
            perp = max(5, min(30, n // 4))
            return TSNE(n_components=2, random_state=seed, perplexity=perp,
                        init="pca", learning_rate="auto").fit_transform(x)
        except Exception:
            pass
    return PCA(n_components=2, random_state=seed).fit_transform(x)


def _dominant_role(devices, labels, roles: pd.Series | None) -> dict[int, str]:
    if roles is None:
        return {}
    s = pd.DataFrame({"device": devices, "cluster": labels})
    s["role"] = s["device"].map(roles).fillna("external")
    out = {}
    for c, grp in s.groupby("cluster"):
        out[int(c)] = grp["role"].mode().iat[0]
    return out


def run(cfg: PipelineConfig) -> dict:
    os.makedirs(cfg.artifacts_dir, exist_ok=True)
    windows = lanl_io.discover_windows(cfg.data_dir)
    if not windows:
        raise FileNotFoundError(
            f"В каталоге {cfg.data_dir} нет окон. Сначала сгенерируйте данные "
            f"(python -m hostembed.datagen --out {cfg.data_dir}).")

    labels_df = lanl_io.load_labels(cfg.data_dir)
    roles = (labels_df.set_index("device")["role"] if labels_df is not None else None)
    truth_df = lanl_io.load_anomaly_truth(cfg.data_dir)

    # Анализируем ВНУТРЕННИЕ хосты. Внешние узлы (интернет-пиры) остаются лишь
    # контекстом для признаков/графа. Если разметки нет (реальный LANL) —
    # анализируем все устройства.
    internal = sorted(roles.index) if roles is not None else None

    # Признаки по всем окнам (для аномалий) и «текущее» окно — последнее.
    window_feats = [extract_features(lanl_io.load_window(w), internal) for w in windows]
    cur_feats = window_feats[-1]
    cur_flows = lanl_io.load_window(windows[-1])
    devices = list(cur_feats.index)

    # --- Представления: три метода ---
    fe = FeatureEmbedder(n_components=cfg.pca_dim, seed=cfg.seed)
    feat_emb = fe.fit_transform(cur_feats)
    all_nodes = sorted(set(cur_flows["src_device"]) | set(cur_flows["dst_device"]))
    graph_emb = graph_embedding(cur_flows, all_nodes, cfg.embed_dim, cfg.graph_walk_len, cfg.seed)
    graph_emb = graph_emb.reindex(devices).fillna(0.0)
    comb_emb = combined_embedding(feat_emb, graph_emb)
    embeddings = {"features": feat_emb, "graph": graph_emb, "combined": comb_emb}

    # --- Кластеризация + оценка каждого метода ---
    eval_rows = {}
    clusterings = {}
    for name in cfg.methods:
        emb = embeddings[name]
        k, _scores = select_k(emb.values, cfg.k_min, cfg.k_max, cfg.seed)
        lab, _km = cluster(emb.values, k, cfg.seed)
        clusterings[name] = {"k": k, "labels": lab}
        eval_rows[name] = {"k": k, **evaluate_method(emb, lab, roles, cfg.n_neighbors)}

    # Выбор лучшего метода — по силуэту (без разметки).
    best_method = max(cfg.methods,
                      key=lambda m: (eval_rows[m].get("silhouette") or -1))
    best_emb = embeddings[best_method]
    best_labels = clusterings[best_method]["labels"]
    best_k = clusterings[best_method]["k"]

    # --- Поиск похожих + объяснения (на лучшем представлении) ---
    sim = Similarity(best_emb)
    expl = Explainer(cur_feats, cur_flows)
    coords = _project_2d(best_emb, cfg.seed)
    profiles = cluster_profiles(cur_feats, best_labels)
    dom_roles = _dominant_role(devices, best_labels, roles)

    lab_series = pd.Series(best_labels, index=devices)
    hosts_out = []
    for i, dev in enumerate(devices):
        nbrs = sim.neighbors(dev, cfg.n_neighbors)
        neighbors = [{
            "device": nd, "similarity": round(s, 4),
            "same_cluster": bool(lab_series.get(nd) == lab_series.get(dev)),
            "reasons": expl.explain(dev, nd),
        } for nd, s in nbrs]
        hosts_out.append({
            "device": dev,
            "role": (roles.get(dev, "external") if roles is not None else "unknown"),
            "cluster": int(best_labels[i]),
            "x": float(coords[i, 0]), "y": float(coords[i, 1]),
            "top_features": expl.top_features(dev),
            "stats": {
                "total_flows": int(cur_feats.loc[dev, "total_flows"]),
                "in_peers": int(cur_feats.loc[dev, "in_peers"]),
                "out_peers": int(cur_feats.loc[dev, "out_peers"]),
                "log_total_bytes": round(float(cur_feats.loc[dev, "log_total_bytes"]), 2),
                "initiated_ratio": round(float(cur_feats.loc[dev, "initiated_ratio"]), 3),
            },
            "neighbors": neighbors,
        })

    clusters_out = []
    for c in sorted(set(best_labels)):
        members = [d for d, l in zip(devices, best_labels) if l == c]
        clusters_out.append({
            "cluster": int(c), "size": len(members),
            "profile": profiles.get(int(c), []),
            "dominant_role": dom_roles.get(int(c), "n/a"),
        })

    # --- Аномалии по окнам ---
    anom = detect_anomalies(window_feats, cfg.k_min, cfg.k_max,
                            cfg.anomaly_z_threshold, cfg.anomaly_top_n, seed=cfg.seed)
    anom_scores = anom["scores"]
    anomalies_out = []
    if not anom_scores.empty:
        truth_set = set(truth_df["device"]) if truth_df is not None else set()
        shown = anom_scores[anom_scores["in_top"] | anom_scores["is_anomaly"]]
        for _, r in shown.iterrows():
            anomalies_out.append({
                "device": r["device"],
                "score": round(float(r["score"]), 3),
                "window": int(r["window"]),
                "from_cluster": int(r["from_cluster"]),
                "to_cluster": int(r["to_cluster"]),
                "z_drift": round(float(r["z_drift"]), 2),
                "rarity": round(float(r["rarity"]), 2),
                "cluster_changed": bool(r["cluster_changed"]),
                "is_anomaly": bool(r["is_anomaly"]),
                "trajectory": list(r["trajectory"]),
                "ground_truth": r["device"] in truth_set,
            })

    # --- Сводка и сохранение артефактов ---
    summary = {
        "n_devices": len(devices),
        "n_windows": len(windows),
        "methods": list(cfg.methods),
        "best_method": best_method,
        "best_k": int(best_k),
        "evaluation": eval_rows,
        "anomaly_baseline_k": anom.get("baseline_k"),
        "n_anomalies_flagged": int(anom_scores["is_anomaly"].sum()) if not anom_scores.empty else 0,
        "has_labels": roles is not None,
    }

    artifacts = {
        "summary.json": summary,
        "hosts.json": {"hosts": hosts_out},
        "clusters.json": {"clusters": clusters_out},
        "anomalies.json": {"anomalies": anomalies_out},
    }
    for fname, obj in artifacts.items():
        with open(os.path.join(cfg.artifacts_dir, fname), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    return summary


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Запуск пайплайна host2vec.")
    p.add_argument("--data", default="data")
    p.add_argument("--artifacts", default="artifacts")
    args = p.parse_args()
    cfg = PipelineConfig(data_dir=args.data, artifacts_dir=args.artifacts)
    summary = run(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
