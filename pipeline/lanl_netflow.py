"""Reader for the LANL Unified Host and Network Dataset (2017) NETFLOW files.

Source: https://csr.lanl.gov/data/2017/ (netflow_day-NN.bz2). Documented format:
comma-separated, one record per line, NO header row, bzip2-compressed. Field order:

    Time, Duration, SrcDevice, DstDevice, Protocol, SrcPort, DstPort,
    SrcPackets, DstPackets, SrcBytes, DstBytes

- Time        start of the event, "epoch time format". The unit/origin is NOT stated by
              the source, so this pipeline does not guess a calendar date: Time is kept
              as a number and windows are given in the same units.
- Duration    seconds.
- SrcDevice   "the device that likely initiated the event" (anonymised, e.g. Comp132598).
- DstDevice   the receiving device.
- Protocol    protocol number (kept as an opaque string token).
- SrcPort/DstPort  anonymised as PortNNNN except well-known ports (kept as strings).
- SrcPackets/SrcBytes   sent by SrcDevice; DstPackets/DstBytes sent by DstDevice.

Canonical flow columns used by the benchmark (BENCHMARK_CONTRACT.md):

    src, dst, dst_port, proto, bytes, packets, ts

Mapping (direction = originator -> responder, nothing is invented):
    src=SrcDevice, dst=DstDevice, dst_port=DstPort, proto=Protocol,
    bytes=SrcBytes, packets=SrcPackets, ts=Time.
The responder's counters and other fields are preserved as extra columns:
    src_port, duration, resp_bytes (DstBytes), resp_packets (DstPackets).

Host ID: the device string exactly as in the file (stripped; case-sensitive, never
re-numbered), so IDs are stable across files, windows and methods.
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
    """Input data cannot be processed; the message says why and where."""


def iter_raw_chunks(path: Path, chunksize: int = 500_000, strict: bool = True,
                    bad_lines: Counter | None = None) -> Iterator[pd.DataFrame]:
    """Yield raw string chunks. bz2 is decompressed by pandas (streamed, bounded memory).

    strict=True: a line with too many fields raises PipelineError.
    strict=False: such lines are skipped and counted in bad_lines["wrong_field_count"].
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
    """Convert one raw chunk into canonical flows; drop and count invalid rows by reason."""
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
    """Stream a raw netflow file as canonical-flow chunks; bad-row counts go to `bad`."""
    bad = bad if bad is not None else Counter()
    for raw in iter_raw_chunks(path, chunksize, strict, bad):
        flows, counts = canonicalize(raw)
        bad.update(counts)
        yield flows
