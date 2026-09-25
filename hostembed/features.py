"""Извлечение поведенческих признаков хоста из netflow-потоков.

На вход — поток LANL-подобных записей за окно. На выход — матрица
«хост × признак» (pandas.DataFrame, индекс = имя устройства). Все признаки
вычисляются только из сетевых данных: объёмы, направления, порты/сервисы,
число партнёров, протоколы, временны́е паттерны.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BUSINESS_HOURS, PROTO_ICMP, PROTO_TCP, PROTO_UDP, SERVICE_PORTS


def _entropy(counts: np.ndarray) -> float:
    counts = counts[counts > 0].astype(float)
    if counts.size <= 1:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _port_columns(key: str) -> list[str]:
    """Полный (фиксированный) список колонок-индикаторов сервисных портов."""
    return [f"{key}_{name}_{port}" for port, name in SERVICE_PORTS.items()]


def _port_fractions(sub: pd.DataFrame, key: str) -> pd.DataFrame:
    """Доля потоков на каждый сервисный порт (pivot host × port).

    Возвращает ФИКСИРОВАННЫЙ набор колонок по всем портам из SERVICE_PORTS,
    чтобы схема признаков не зависела от того, какие порты встретились в окне —
    это критично для сравнения представлений разных окон (детекция аномалий).
    """
    cols = _port_columns(key)
    svc = sub[sub["dport"].isin(SERVICE_PORTS.keys())]
    if svc.empty:
        return pd.DataFrame(columns=cols, index=pd.Index([], name="host"))
    tab = (svc.groupby(["host", "dport"]).size()
              .unstack(fill_value=0))
    totals = sub.groupby("host").size()
    tab = tab.div(totals, axis=0).fillna(0.0)
    tab.columns = [f"{key}_{SERVICE_PORTS[int(c)]}_{int(c)}" for c in tab.columns]
    return tab.reindex(columns=cols, fill_value=0.0)


def extract_features(df: pd.DataFrame, hosts: list[str] | None = None) -> pd.DataFrame:
    """Построить матрицу признаков хостов за одно окно.

    Признаки хоста агрегируются по ВСЕМ его потокам (в т.ч. к внешним пирам),
    но если задан ``hosts``, в результат попадают строки только этих устройств
    (например, только внутренние хосты — внешние узлы остаются лишь контекстом).
    """
    if df.empty:
        return pd.DataFrame()

    base = df.copy()
    base["hour"] = ((base["time"] % 86400) // 3600).astype(int)

    # Разворачиваем каждый поток в две «перспективы»: для инициатора и получателя.
    out = pd.DataFrame({
        "host": base["src_device"], "peer": base["dst_device"], "dir": "out",
        "sent_b": base["src_bytes"], "recv_b": base["dst_bytes"],
        "sent_p": base["src_packets"], "recv_p": base["dst_packets"],
        "dur": base["duration"], "proto": base["protocol"],
        "dport": base["dst_port"], "hour": base["hour"],
    })
    inc = pd.DataFrame({
        "host": base["dst_device"], "peer": base["src_device"], "dir": "in",
        "sent_b": base["dst_bytes"], "recv_b": base["src_bytes"],
        "sent_p": base["dst_packets"], "recv_p": base["src_packets"],
        "dur": base["duration"], "proto": base["protocol"],
        "dport": base["dst_port"], "hour": base["hour"],
    })
    ev = pd.concat([out, inc], ignore_index=True)

    all_hosts = sorted(ev["host"].unique())
    if hosts is not None:
        keep = [h for h in hosts if h in set(all_hosts)]
        hosts = pd.Index(keep, name="device")
    else:
        hosts = pd.Index(all_hosts, name="device")
    feats = pd.DataFrame(index=hosts)

    g = ev.groupby("host")
    feats["total_flows"] = g.size()
    feats["out_flows"] = out.groupby("host").size().reindex(hosts).fillna(0)
    feats["in_flows"] = inc.groupby("host").size().reindex(hosts).fillna(0)
    feats["out_peers"] = out.groupby("host")["peer"].nunique().reindex(hosts).fillna(0)
    feats["in_peers"] = inc.groupby("host")["peer"].nunique().reindex(hosts).fillna(0)

    feats["initiated_ratio"] = feats["out_flows"] / feats["total_flows"].clip(lower=1)
    feats["fanout_fanin_log"] = (np.log1p(feats["out_peers"]) - np.log1p(feats["in_peers"]))

    # Объёмы (с точки зрения хоста).
    feats["sent_bytes"] = g["sent_b"].sum()
    feats["recv_bytes"] = g["recv_b"].sum()
    feats["sent_packets"] = g["sent_p"].sum()
    feats["recv_packets"] = g["recv_p"].sum()
    total_bytes = feats["sent_bytes"] + feats["recv_bytes"]
    total_packets = (feats["sent_packets"] + feats["recv_packets"]).clip(lower=1)
    feats["log_total_bytes"] = np.log1p(total_bytes)
    feats["bytes_per_packet"] = total_bytes / total_packets
    feats["mean_flow_bytes"] = total_bytes / feats["total_flows"].clip(lower=1)
    feats["send_recv_log_ratio"] = (np.log1p(feats["sent_bytes"]) - np.log1p(feats["recv_bytes"]))

    # Длительность.
    feats["mean_duration"] = g["dur"].mean()
    feats["median_duration"] = g["dur"].median()

    # Протоколы.
    proto = ev.assign(
        tcp=(ev["proto"] == PROTO_TCP).astype(int),
        udp=(ev["proto"] == PROTO_UDP).astype(int),
        icmp=(ev["proto"] == PROTO_ICMP).astype(int),
    ).groupby("host")[["tcp", "udp", "icmp"]].mean()
    feats["tcp_frac"] = proto["tcp"]
    feats["udp_frac"] = proto["udp"]
    feats["icmp_frac"] = proto["icmp"]

    # Энтропия портов: потребляемых (out) и обслуживаемых (in).
    feats["consume_port_entropy"] = (
        out.groupby("host")["dport"].apply(lambda s: _entropy(s.value_counts().values))
        .reindex(hosts).fillna(0.0))
    feats["serve_port_entropy"] = (
        inc.groupby("host")["dport"].apply(lambda s: _entropy(s.value_counts().values))
        .reindex(hosts).fillna(0.0))

    # Временны́е паттерны.
    feats["active_hours"] = g["hour"].nunique()
    feats["flows_per_active_hour"] = feats["total_flows"] / feats["active_hours"].clip(lower=1)
    lo, hi = BUSINESS_HOURS
    is_biz = ev["hour"].between(lo, hi - 1)
    biz = ev.assign(biz=is_biz.astype(int)).groupby("host")["biz"].mean()
    feats["business_hours_frac"] = biz
    peak = ev.groupby(["host", "hour"]).size().groupby("host").max()
    feats["peak_hour_share"] = (peak / feats["total_flows"].clip(lower=1)).reindex(hosts)

    # Индикаторы сервисных портов (обслуживаемые и потребляемые).
    serve = _port_fractions(inc, "serve").reindex(hosts).fillna(0.0)
    consume = _port_fractions(out, "use").reindex(hosts).fillna(0.0)
    feats = feats.join(serve).join(consume)

    feats = feats.fillna(0.0)
    feats.index.name = "device"
    return feats
