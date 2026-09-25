"""Кластеризация хостов, поиск похожих и объяснение соседства."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .config import SERVICE_PORTS

# Человеко-читаемые расшифровки «базовых» признаков (для объяснений и профилей).
_FEATURE_PHRASES = {
    "total_flows": "высокая сетевая активность",
    "out_flows": "много исходящих потоков",
    "in_flows": "много входящих потоков",
    "out_peers": "много исходящих партнёров",
    "in_peers": "к нему подключается много хостов",
    "initiated_ratio": "преимущественно инициирует соединения (клиент)",
    "fanout_fanin_log": "исходящих связей больше, чем входящих",
    "log_total_bytes": "большой суммарный объём трафика",
    "bytes_per_packet": "крупные пакеты",
    "mean_flow_bytes": "крупные потоки",
    "send_recv_log_ratio": "отдаёт данных больше, чем получает",
    "mean_duration": "длинные соединения",
    "median_duration": "длинные соединения",
    "tcp_frac": "преобладает TCP",
    "udp_frac": "преобладает UDP",
    "icmp_frac": "заметная доля ICMP",
    "consume_port_entropy": "разнообразие потребляемых сервисов",
    "serve_port_entropy": "разнообразие обслуживаемых сервисов",
    "active_hours": "активен много часов в сутки",
    "flows_per_active_hour": "высокая интенсивность в активные часы",
    "business_hours_frac": "активность в рабочее время",
    "peak_hour_share": "активность сконцентрирована по времени",
}


def _humanize(feature: str) -> str:
    if feature in _FEATURE_PHRASES:
        return _FEATURE_PHRASES[feature]
    if feature.startswith("serve_"):
        name = feature.split("_", 1)[1]
        return f"обслуживает сервис {name}"
    if feature.startswith("use_"):
        name = feature.split("_", 1)[1]
        return f"обращается к сервису {name}"
    return feature


def select_k(emb: np.ndarray, k_min: int, k_max: int, seed: int = 42) -> tuple[int, dict]:
    """Подобрать число кластеров по силуэту."""
    n = emb.shape[0]
    k_max = min(k_max, n - 1)
    scores: dict[int, float] = {}
    best_k, best_s = k_min, -1.0
    for k in range(max(2, k_min), max(3, k_max + 1)):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(emb)
        if len(set(labels)) < 2:
            continue
        s = float(silhouette_score(emb, labels))
        scores[k] = s
        if s > best_s:
            best_s, best_k = s, k
    return best_k, scores


def cluster(emb: np.ndarray, k: int, seed: int = 42) -> tuple[np.ndarray, KMeans]:
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(emb)
    return labels, km


class Similarity:
    """Поиск ближайших хостов в пространстве представлений (косинус)."""

    def __init__(self, emb_df: pd.DataFrame):
        self.devices = list(emb_df.index)
        self._pos = {d: i for i, d in enumerate(self.devices)}
        self.mat = emb_df.values
        n_fit = min(len(self.devices), 50)
        self.nn = NearestNeighbors(n_neighbors=n_fit, metric="cosine")
        self.nn.fit(self.mat)

    def neighbors(self, device: str, k: int = 8) -> list[tuple[str, float]]:
        if device not in self._pos:
            return []
        i = self._pos[device]
        k_eff = min(k + 1, len(self.devices))
        dist, idx = self.nn.kneighbors(self.mat[i:i + 1], n_neighbors=k_eff)
        out = []
        for d, j in zip(dist[0], idx[0]):
            if j == i:
                continue
            out.append((self.devices[j], float(1.0 - d)))  # similarity = 1 - cos_dist
        return out[:k]


class Explainer:
    """Объясняет, почему два хоста оказались похожими."""

    def __init__(self, feat_df: pd.DataFrame, flows: pd.DataFrame):
        self.feat = feat_df
        self.columns = list(feat_df.columns)
        self.z = pd.DataFrame(
            StandardScaler().fit_transform(feat_df.values),
            index=feat_df.index, columns=feat_df.columns,
        )
        # Множества партнёров для пересечения «общих соседей».
        self.peers: dict[str, set] = {}
        out = flows.groupby("src_device")["dst_device"].agg(set)
        inc = flows.groupby("dst_device")["src_device"].agg(set)
        for dev in feat_df.index:
            self.peers[dev] = set(out.get(dev, set())) | set(inc.get(dev, set()))

    def top_features(self, device: str, n: int = 5) -> list[str]:
        if device not in self.z.index:
            return []
        row = self.z.loc[device]
        top = row[row > 0.5].sort_values(ascending=False).head(n)
        return [_humanize(f) for f in top.index]

    def explain(self, a: str, b: str, n: int = 5) -> list[str]:
        reasons: list[str] = []
        if a not in self.z.index or b not in self.z.index:
            return reasons
        za, zb = self.z.loc[a], self.z.loc[b]
        # Признаки, по которым ОБА заметно выше среднего и близки между собой.
        both_high = (za > 0.5) & (zb > 0.5)
        cand = []
        for feat in both_high.index[both_high.values]:
            strength = min(za[feat], zb[feat])
            diff = abs(za[feat] - zb[feat])
            cand.append((feat, strength, diff))
        cand.sort(key=lambda x: (-x[1], x[2]))
        for feat, _strength, _diff in cand[:n]:
            reasons.append(_humanize(feat))

        # Общие сервисные порты.
        shared_ports = []
        for port, name in SERVICE_PORTS.items():
            for pref in ("serve", "use"):
                col = [c for c in self.columns if c.startswith(f"{pref}_{name}_{port}")]
                if col and self.feat.loc[a, col[0]] > 0 and self.feat.loc[b, col[0]] > 0:
                    verb = "обслуживают" if pref == "serve" else "обращаются к"
                    shared_ports.append(f"оба {verb} {name}:{port}")
        for s in shared_ports[:3]:
            if s not in reasons:
                reasons.append(s)

        # Общие партнёры по коммуникации.
        common = self.peers.get(a, set()) & self.peers.get(b, set())
        if len(common) >= 2:
            sample = ", ".join(sorted(common)[:3])
            reasons.append(f"{len(common)} общих партнёров по сети (напр. {sample})")

        if not reasons:
            reasons.append("близкие поведенческие профили по совокупности признаков")
        return reasons


def cluster_profiles(feat_df: pd.DataFrame, labels: np.ndarray, n: int = 4) -> dict[int, list[str]]:
    """Характерные черты каждого кластера (признаки с наибольшим средним z)."""
    z = pd.DataFrame(
        StandardScaler().fit_transform(feat_df.values),
        index=feat_df.index, columns=feat_df.columns,
    )
    z["__cluster__"] = labels
    profiles: dict[int, list[str]] = {}
    for c, grp in z.groupby("__cluster__"):
        means = grp.drop(columns="__cluster__").mean().sort_values(ascending=False)
        top = [f for f in means.index if means[f] > 0.4][:n]
        profiles[int(c)] = [_humanize(f) for f in top] or ["типовое поведение"]
    return profiles
