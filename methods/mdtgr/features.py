"""Поведенческий профиль хоста в окне — вход модели и «якорный» блок эмбеддинга.

Профиль повторяет по смыслу публичный поведенческий эталон стенда
(``evaluation.knn_consistency.build_reference_features``): объёмы, число собеседников,
доля исходящего, распределения по портам/протоколам/часам; каждый блок стандартизуется
отдельно и делится на sqrt(ширины), чтобы блоки имели одинаковый вес.

Отличия осознанные и нужны для метода:
- словарь портов и протоколов фиксируется по train-окну (``Schema``), поэтому ширина
  вектора одинакова во всех окнах;
- есть два варианта нормировки: по самому окну (так делает эталон, поэтому блок хорошо
  держит Behavior) и по train-схеме (общая система координат для обучаемой части).

``ts`` во входных потоках — целые секунды Unix; час считается как ``(ts // 3600) % 24``.
Строковые колонки ожидаются категориальными (``pd.Categorical``): так коды словарей и
группировки не материализуют миллионы объектов.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BLOCKS = ("volume", "peers", "ratio", "ports", "proto", "hourly")
TOP_PORTS = 20
WEIGHTS = {"volume": 1.0, "peers": 1.0, "ratio": 1.0, "ports": 1.0, "proto": 0.5, "hourly": 1.0}


@dataclass
class Schema:
    """Словарь и нормировки поведенческого блока (фиксируются по train-окну)."""

    ports: list[str]
    protos: list[str]
    scalers: dict = field(default_factory=dict)

    @property
    def width(self) -> int:
        return 6 + 2 + 2 + len(self.ports) + len(self.protos) + 24

    def to_json(self) -> dict:
        return {"ports": self.ports, "protos": self.protos,
                "scalers": {k: {"mean": v["mean"].tolist(), "scale": v["scale"].tolist()}
                            for k, v in self.scalers.items()}}

    @classmethod
    def from_json(cls, raw: dict) -> "Schema":
        scalers = {k: {"mean": np.asarray(v["mean"], np.float64),
                       "scale": np.asarray(v["scale"], np.float64)}
                   for k, v in raw.get("scalers", {}).items()}
        return cls(ports=list(raw["ports"]), protos=list(raw["protos"]), scalers=scalers)


def _two_way_dist(frame: pd.DataFrame, column: str, hosts: pd.Index, keep: list[str]) -> np.ndarray:
    """Распределение значения по хостам с ОБЕИХ сторон потока (src и dst), как в эталоне."""
    keep_index = pd.Index(keep)
    out = (frame.groupby(["src", column], observed=True).size()
           .unstack(column, fill_value=0.0).reindex(index=hosts, columns=keep_index,
                                                    fill_value=0.0).to_numpy(np.float64))
    inn = (frame.groupby(["dst", column], observed=True).size()
           .unstack(column, fill_value=0.0).reindex(index=hosts, columns=keep_index,
                                                    fill_value=0.0).to_numpy(np.float64))
    both = out + inn
    return both / np.maximum(both.sum(1, keepdims=True), 1.0)


def _as_blocks(frame: pd.DataFrame, hosts: pd.Index, ports: list[str],
               protos: list[str]) -> dict:
    """Сырые блоки признаков до стандартизации. Петли не выкидываются — как в эталоне."""
    byt = frame["bytes"].to_numpy(np.float64)
    pkt = frame["packets"].to_numpy(np.float64)
    hour = ((frame["ts"].to_numpy(np.int64) // 3600) % 24).astype(np.int64)
    local = pd.DataFrame({"src": frame["src"].values, "dst": frame["dst"].values,
                          "bytes": byt, "packets": pkt, "hour": hour,
                          "port": frame["dst_port"].values, "proto": frame["proto"].values},
                         copy=False)
    out = local.groupby("src", observed=True).agg(
        out_flows=("dst", "size"), out_bytes=("bytes", "sum"),
        out_pkts=("packets", "sum"), out_peers=("dst", "nunique"))
    inn = local.groupby("dst", observed=True).agg(
        in_flows=("src", "size"), in_bytes=("bytes", "sum"),
        in_pkts=("packets", "sum"), in_peers=("src", "nunique"))
    base = out.join(inn, how="outer").reindex(hosts).fillna(0.0)

    volume = np.log1p(base[["out_flows", "out_bytes", "out_pkts",
                            "in_flows", "in_bytes", "in_pkts"]].to_numpy(np.float64))
    peers = np.log1p(base[["out_peers", "in_peers"]].to_numpy(np.float64))
    total_flows = (base.out_flows + base.in_flows).replace(0, 1.0)
    total_bytes = (base.out_bytes + base.in_bytes).replace(0, 1.0)
    ratio = np.column_stack([base.out_flows / total_flows, base.out_bytes / total_bytes])
    return {"volume": volume, "peers": peers, "ratio": ratio,
            "ports": _two_way_dist(local, "port", hosts, ports),
            "proto": _two_way_dist(local, "proto", hosts, protos),
            "hourly": _two_way_dist(local, "hour", hosts, list(range(24)))}


def fit_schema(frame: pd.DataFrame, hosts: list[str]) -> Schema:
    """Зафиксировать словари и нормировки по train-окну."""
    ports = [str(p) for p in frame["dst_port"].astype(str).value_counts().head(TOP_PORTS).index]
    protos = sorted(str(p) for p in frame["proto"].astype(str).unique())
    blocks = _as_blocks(frame, pd.Index([str(h) for h in hosts]), ports, protos)
    scalers = {}
    for name, values in blocks.items():
        mean = values.mean(0)
        scale = values.std(0)
        scalers[name] = {"mean": mean, "scale": np.where(scale > 1e-12, scale, 1.0)}
    return Schema(ports=ports, protos=protos, scalers=scalers)


def _standardize(blocks: dict, schema: Schema, *, own_fit: bool) -> np.ndarray:
    parts = []
    for name in BLOCKS:
        values = blocks[name]
        if own_fit:
            mean = values.mean(0)
            scale = values.std(0)
            scale = np.where(scale > 1e-12, scale, 1.0)
        else:
            state = schema.scalers.get(name)
            if state is None:
                raise KeyError(f"в схеме нет нормировки блока {name}")
            mean, scale = state["mean"], state["scale"]
        z = np.nan_to_num((values - mean) / scale)
        z = z / np.sqrt(max(z.shape[1], 1))
        parts.append(z * WEIGHTS[name])
    return np.column_stack(parts).astype(np.float32)


def behavior_pair(frame: pd.DataFrame, hosts: list[str], schema: Schema) -> tuple[np.ndarray, np.ndarray]:
    """Два профиля окна за один проход: (по самому окну, по train-схеме)."""
    index = pd.Index([str(h) for h in hosts])
    blocks = _as_blocks(frame, index, schema.ports, schema.protos)
    return (_standardize(blocks, schema, own_fit=True),
            _standardize(blocks, schema, own_fit=False))


def behavior_matrix(frame: pd.DataFrame, hosts: list[str], schema: Schema, *,
                    own_fit: bool = True) -> np.ndarray:
    """Один профиль: own_fit=True — по окну, False — по train-схеме."""
    index = pd.Index([str(h) for h in hosts])
    blocks = _as_blocks(frame, index, schema.ports, schema.protos)
    return _standardize(blocks, schema, own_fit=own_fit)
