"""Выделить окна нового дня с заранее фиксированной когортой, потоково."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.lanl_netflow import canonicalize, iter_raw_chunks
from pipeline.provenance import classify
from pipeline.windows import window_mask


def prepare_day(raw: Path, base: Path, out: Path, day: int, chunksize=500_000,
                data_kind="lanl", source_note="форма LANL, 2026-10-31, physcorgi@icloud.com"):
    if (out / "manifest.json").exists():
        return json.loads((out / "manifest.json").read_text())
    offset = (day - 2) * 86400
    windows = {"train": [126000 + offset, 129600 + offset],
               "validation": [129600 + offset, 131400 + offset],
               "test": [131400 + offset, 135000 + offset]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "flows").mkdir(exist_ok=True)
    (out / "graph").mkdir(exist_ok=True)
    original = pd.read_csv(base / "cohort.csv", dtype={"host_id": str}).set_index("host_id")
    members = set(original.index)
    events = {w: pd.Series(dtype=float) for w in windows}
    graphs, rows, wrote, bounds = {}, Counter(), set(), {}
    bad, seen = Counter(), 0
    for chunk in iter_raw_chunks(raw, chunksize=chunksize):
        ts = pd.to_numeric(chunk.Time, errors="coerce")
        duration = pd.to_numeric(chunk.Duration, errors="coerce")
        seen += len(chunk)
        for window, interval in windows.items():
            mask = window_mask(ts.to_numpy(), duration.to_numpy(), interval, True)
            if not mask.any():
                continue
            flows, invalid = canonicalize(chunk.loc[mask])
            bad.update(invalid)
            if flows.empty:
                continue
            path = out / "flows" / f"{window}.csv"
            flows.to_csv(path, index=False, mode="a" if window in wrote else "w", header=window not in wrote)
            wrote.add(window)
            rows[window] += len(flows)
            lo, hi = bounds.get(window, (np.inf, -np.inf))
            bounds[window] = min(lo, float(flows.ts.min())), max(hi, float(flows.ts.max()))
            real = flows.loc[flows.src != flows.dst]
            counts = pd.concat([real.src, real.dst]).value_counts()
            events[window] = events[window].add(counts, fill_value=0)
            f = real.loc[real.src.isin(members) & real.dst.isin(members)]
            a = np.where(f.src < f.dst, f.src, f.dst)
            b = np.where(f.src < f.dst, f.dst, f.src)
            graph = pd.DataFrame({"host_a": a, "host_b": b, "flows": 1,
                                  "bytes": (f.bytes + f.resp_bytes).to_numpy(),
                                  "packets": (f.packets + f.resp_packets).to_numpy()})
            graph = graph.groupby(["host_a", "host_b"]).sum()
            graphs[window] = graph if window not in graphs else graphs[window].add(graph, fill_value=0)
        if seen % 10_000_000 < chunksize:
            print(f"day{day:02d}: прочитано {seen:,} строк; сохранено {dict(rows)}", flush=True)
    if wrote != set(windows):
        raise ValueError(f"Не все окна найдены: {set(windows) - wrote}")
    cohort = pd.DataFrame(index=original.index)
    for w in windows:
        cohort[f"events_{w}"] = events[w].reindex(cohort.index).fillna(0).astype(int)
        if w != "train":
            cohort[f"evaluable_{w}"] = cohort[f"events_{w}"] >= 300
        graphs[w].reset_index().to_csv(out / "graph" / f"edges_{w}.csv", index=False)
    cohort.to_csv(out / "cohort.csv")
    provenance = classify([raw], data_kind, source_note)
    manifest = copy.deepcopy(json.loads((base / "manifest.json").read_text()))
    manifest.update(data_kind=provenance["kind"], synthetic=provenance["kind"] == "synthetic_fixture",
                    provenance=provenance, provenance_verified=False,
                    real_lanl_declared=provenance["kind"] == "lanl_declared",
                    source_files=[str(raw)], invalid_rows=dict(bad), raw_rows=seen)
    manifest["config"]["windows"] = windows
    manifest["windows"] = {w: {"start": a, "end": b, "n_flows": rows[w],
                               "min_time": bounds[w][0], "max_time": bounds[w][1]}
                           for w, (a, b) in windows.items()}
    manifest["cohort"] = {"policy": "train_only", "n_hosts": len(cohort),
                          "rule": "fixed cohort from day02 before later-day observations",
                          "source": str(base / "cohort.csv"),
                          "sha256": hashlib.sha256((out / "cohort.csv").read_bytes()).hexdigest()}
    manifest["counts"] = dict(rows)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--day", required=True, type=int)
    args = ap.parse_args()
    prepare_day(args.raw, args.base, args.out, args.day)


if __name__ == "__main__":
    main()
