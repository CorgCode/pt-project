"""Чтение netflow-окон.

Поддерживает два источника:
  * синтетические CSV, сгенерированные `datagen` (с заголовком);
  * «сырые» файлы LANL NetFlow (без заголовка, 11 колонок, опц. .gz/.bz2) —
    колонки приводятся к единой схеме `NETFLOW_COLUMNS`.
"""
from __future__ import annotations

import glob
import json
import os

import pandas as pd

from .config import NETFLOW_COLUMNS

_NUMERIC = ["time", "duration", "protocol", "src_port", "dst_port",
            "src_packets", "dst_packets", "src_bytes", "dst_bytes"]


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    for col in _NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["src_device", "dst_device"])
    df[_NUMERIC] = df[_NUMERIC].fillna(0)
    return df.reset_index(drop=True)


def load_window(path: str) -> pd.DataFrame:
    """Прочитать одно окно в DataFrame со схемой NETFLOW_COLUMNS."""
    compression = "infer"
    # Пытаемся прочитать с заголовком (наш синтетический формат).
    head = pd.read_csv(path, nrows=1, compression=compression)
    has_header = set(NETFLOW_COLUMNS).issubset(set(map(str, head.columns)))
    if has_header:
        df = pd.read_csv(path, compression=compression)
        df = df[[c for c in NETFLOW_COLUMNS]]
    else:
        # Сырой LANL: без заголовка, ровно 11 колонок в порядке схемы.
        df = pd.read_csv(path, header=None, names=NETFLOW_COLUMNS, compression=compression)
    return _coerce(df)


def discover_windows(data_dir: str) -> list[str]:
    """Список файлов-окон в порядке времени."""
    meta_path = os.path.join(data_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        files = [os.path.join(data_dir, f) for f in meta.get("window_files", [])]
        if files:
            return files
    patterns = ["netflow_window_*.csv", "*.csv", "*.csv.gz", "*.csv.bz2", "*.txt", "*.txt.gz"]
    found: list[str] = []
    for pat in patterns:
        found = sorted(glob.glob(os.path.join(data_dir, pat)))
        # Исключаем служебные файлы разметки.
        found = [f for f in found if os.path.basename(f) not in
                 ("labels.csv", "anomalies_truth.csv")]
        if found:
            break
    return found


def load_labels(data_dir: str) -> pd.DataFrame | None:
    path = os.path.join(data_dir, "labels.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_anomaly_truth(data_dir: str) -> pd.DataFrame | None:
    path = os.path.join(data_dir, "anomalies_truth.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None
