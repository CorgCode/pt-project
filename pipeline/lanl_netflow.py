"""Чтение NetFlow-файлов LANL Unified Host and Network Dataset (2017).

Источник: https://csr.lanl.gov/data/2017/ (netflow_day-NN.bz2). CSV без заголовка, bzip2. Поля:

    Time, Duration, SrcDevice, DstDevice, Protocol, SrcPort, DstPort,
    SrcPackets, DstPackets, SrcBytes, DstBytes

Единицы и начало отсчёта Time источник не указывает — в дату не переводим, окна задаются в тех же
единицах. Устройства и редкие порты обезличены (Comp132598, Port12597).

Нормализованный поток: src, dst, dst_port, proto, bytes, packets, ts
    src=SrcDevice, dst=DstDevice, dst_port=DstPort, proto=Protocol,
    bytes=SrcBytes, packets=SrcPackets, ts=Time  (направление: инициатор -> получатель).
Остальное сохраняется в доп. колонках: src_port, duration, resp_bytes, resp_packets.
host_id — строка устройства как есть (без перенумерации), поэтому ID одинаковы во всех файлах и окнах.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

RAW_COLUMNS = ["Time", "Duration", "SrcDevice", "DstDevice", "Protocol", "SrcPort",
               "DstPort", "SrcPackets", "DstPackets", "SrcBytes", "DstBytes"]
CANONICAL_COLUMNS = ["src", "dst", "dst_port", "proto", "bytes", "packets", "ts"]
EXTRA_COLUMNS = ["src_port", "duration", "resp_bytes", "resp_packets"]
FLOW_COLUMNS = CANONICAL_COLUMNS + EXTRA_COLUMNS
SOURCE_FORMAT = "lanl-uhn-2017-netflow"


class PipelineError(ValueError):
    """Данные нельзя обработать; в сообщении — почему и где."""


def iter_raw_chunks(path: Path, chunksize: int = 500_000, strict: bool = True,
                    bad_lines: Counter | None = None) -> Iterator[pd.DataFrame]:
    """Читать сырые куски файла (bz2 распаковывает pandas, потоково).

    strict=True: строка с лишними полями — ошибка; strict=False — пропуск и счёт в bad_lines["wrong_field_count"].
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    kwargs: dict = dict(header=None, names=RAW_COLUMNS, dtype=str, chunksize=chunksize,
                        keep_default_na=False)
    if not strict:
        def skip(_line):
            if bad_lines is not None:
                bad_lines["wrong_field_count"] += 1
            return None
        kwargs.update(engine="python", on_bad_lines=skip)
    try:
        yield from pd.read_csv(path, **kwargs)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise PipelineError(f"{path}: malformed netflow data: {exc}") from exc


def canonicalize(raw: pd.DataFrame) -> tuple[pd.DataFrame, Counter]:
    """Сырой кусок -> нормализованные потоки; битые строки выбрасываются и считаются по причинам."""
    bad: Counter = Counter()
    df = raw.copy()
    for col in ("SrcDevice", "DstDevice", "Protocol", "SrcPort", "DstPort"):
        df[col] = df[col].str.strip()
    num = {c: pd.to_numeric(df[c].str.strip(), errors="coerce") for c in
           ("Time", "Duration", "SrcPackets", "DstPackets", "SrcBytes", "DstBytes")}
    keep = pd.Series(True, index=df.index)

    def reject(mask: pd.Series, reason: str) -> None:
        nonlocal keep
        mask = mask & keep
        bad[reason] += int(mask.sum())
        keep &= ~mask

    reject(df["SrcDevice"].eq("") | df["DstDevice"].eq(""), "blank_device")
    reject(~np.isfinite(num["Time"]), "invalid_time")
    reject(~np.isfinite(num["Duration"]) | (num["Duration"] < 0), "invalid_duration")
    for c in ("SrcPackets", "DstPackets", "SrcBytes", "DstBytes"):
        reject(~np.isfinite(num[c]) | (num[c] < 0), "invalid_count")
    reject(df["Protocol"].eq("") | df["DstPort"].eq(""), "blank_protocol_or_port")

    out = pd.DataFrame({
        "src": df["SrcDevice"], "dst": df["DstDevice"], "dst_port": df["DstPort"],
        "proto": df["Protocol"], "bytes": num["SrcBytes"], "packets": num["SrcPackets"],
        "ts": num["Time"], "src_port": df["SrcPort"], "duration": num["Duration"],
        "resp_bytes": num["DstBytes"], "resp_packets": num["DstPackets"],
    })[FLOW_COLUMNS]
    return out[keep].reset_index(drop=True), bad


def read_canonical(path: Path, chunksize: int = 500_000, strict: bool = True,
                   bad: Counter | None = None) -> Iterator[pd.DataFrame]:
    """Читать файл кусками нормализованных потоков; счётчики битых строк — в bad."""
    bad = bad if bad is not None else Counter()
    for raw in iter_raw_chunks(path, chunksize, strict, bad):
        flows, counts = canonicalize(raw)
        bad.update(counts)
        yield flows
