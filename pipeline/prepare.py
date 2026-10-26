"""LANL NetFlow -> нормализованные потоки, окна, когорта и графы.

    python -m pipeline.prepare describe --netflow data/raw/netflow_day-02.bz2 ...
    python -m pipeline.prepare run --netflow data/raw/netflow_day-*.bz2 \
        --config pipeline/example_config.json --output-dir data/processed

В --output-dir: flows/{train,validation,test}.csv, cohort.csv, graph/edges_<окно>.csv, manifest.json
(схема, конфиг, счётчики, хеши, проверки утечек). Когорта фиксируется здесь — до любых моделей.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.lanl_netflow import (FLOW_COLUMNS, RAW_COLUMNS, SOURCE_FORMAT, PipelineError,
                                   read_canonical)
from pipeline.provenance import SYNTHETIC_MARKER, classify
from pipeline.windows import PipelineConfig, window_mask

SCHEMA_VERSION = 2


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def detect_kind(inputs: list[Path], declared: str | None = None, source_note: str | None = None) -> str:
    """Вид происхождения: synthetic_fixture, unverified (по умолчанию) или lanl_declared."""
    return classify([Path(p) for p in inputs], declared, source_note)["kind"]


SELECTION_NOTE = {
    "train_only": ("Когорта определяется ТОЛЬКО по train-окну. Хост, неактивный в позднем окне, там не оценивается; "
                   "каждый эталон строится по оцениваемым хостам, покрытие пишется в отчёт. «Выживание» хостов "
                   "в поздние окна — часть измеряемого результата, а не определения когорты."),
    "all_windows": ("Когорта — хосты, активные во ВСЕХ окнах, включая будущие validation и test. Будущая активность "
                    "влияет на отбор: это бенчмарк «стабильных хостов». Результаты завышают сохранение, узнаваемость "
                    "и сохранение связей; при этом будущих потоков и рёбер модели не получают."),
}


def describe(inputs: list[Path], chunksize: int = 500_000, strict: bool = True) -> dict:
    """Пройти по файлам (ничего не пишет), чтобы выбрать окна по реальной оси Time."""
    files = []
    for path in inputs:
        bad: Counter = Counter()
        rows, tmin, tmax, hosts, protos = 0, np.inf, -np.inf, set(), Counter()
        for chunk in read_canonical(path, chunksize, strict, bad):
            rows += len(chunk)
            if len(chunk):
                tmin, tmax = min(tmin, chunk["ts"].min()), max(tmax, chunk["ts"].max())
                hosts |= set(chunk["src"]) | set(chunk["dst"])
                protos.update(chunk["proto"].value_counts().to_dict())
        files.append({"file": str(path), "rows": rows, "min_time": None if rows == 0 else float(tmin),
                      "max_time": None if rows == 0 else float(tmax), "n_hosts": len(hosts),
                      "invalid_rows": dict(bad), "protocols_top5": dict(protos.most_common(5))})
    return {"source_format": SOURCE_FORMAT, "raw_columns": RAW_COLUMNS, "files": files}


def _write_chunk(path: Path, df: pd.DataFrame, first: bool) -> None:
    df.to_csv(path, mode="w" if first else "a", header=first, index=False)


def prepare(inputs: list[Path], config: PipelineConfig, out_dir: Path, *, chunksize: int = 500_000,
            strict: bool = True, data_kind: str | None = None, source_note: str | None = None,
            overwrite: bool = False) -> dict:
    inputs = [Path(p) for p in inputs]
    if not inputs:
        raise PipelineError("нет входных файлов")
    provenance = classify(inputs, data_kind, source_note)
    kind = provenance["kind"]
    out_dir = Path(out_dir)
    if (out_dir / "manifest.json").exists() and not overwrite:
        raise PipelineError(f"в {out_dir} уже есть подготовленный датасет; добавьте --overwrite")
    (out_dir / "flows").mkdir(parents=True, exist_ok=True)
    (out_dir / "graph").mkdir(parents=True, exist_ok=True)

    names = config.ordered
    wrote = {w: False for w in names}
    host_events = {w: pd.Series(dtype="float64") for w in names}
    n_rows = {w: 0 for w in names}
    t_range = {w: [np.inf, -np.inf] for w in names}
    bad: Counter = Counter()
    totals = Counter()
    for path in inputs:
        for flows in read_canonical(path, chunksize, strict, bad):
            totals["valid_rows"] += len(flows)
            time, dur = flows["ts"].to_numpy(), flows["duration"].to_numpy()
            assigned = np.zeros(len(flows), dtype=bool)
            for w in names:
                inside = (time >= config.windows[w][0]) & (time < config.windows[w][1])
                keep = window_mask(time, dur, config.windows[w], config.drop_straddling)
                totals[f"straddling_dropped_{w}"] += int((inside & ~keep).sum())
                assigned |= inside
                part = flows[keep]
                if part.empty:
                    continue
                _write_chunk(out_dir / "flows" / f"{w}.csv", part, first=not wrote[w])
                wrote[w] = True
                n_rows[w] += len(part)
                t_range[w] = [min(t_range[w][0], part["ts"].min()), max(t_range[w][1], part["ts"].max())]
                real = part[part["src"] != part["dst"]]
                totals[f"self_flows_{w}"] += len(part) - len(real)
                counts = pd.concat([real["src"], real["dst"]]).value_counts()
                host_events[w] = host_events[w].add(counts, fill_value=0)
            totals["outside_all_windows"] += int((~assigned).sum())
    for w in names:
        if not wrote[w]:
            raise PipelineError(f"в окне {w} {config.windows[w]} нет потоков; "
                                "проверьте диапазон Time командой describe")

    # когорта фиксируется до моделей (см. PipelineConfig.cohort_policy)
    table = pd.DataFrame({w: host_events[w] for w in names}).fillna(0)
    active = table >= config.min_events_per_window
    later = [w for w in names if w != "train"]
    keep = active.all(axis=1) if config.cohort_policy == "all_windows" else active["train"]
    cohort = table[keep].sort_index().astype(int)
    evaluable = active.loc[cohort.index, later]
    problems = []
    if len(cohort) < config.min_cohort_size:
        problems.append(f"cohort has {len(cohort)} hosts")
    for w in later:
        if int(evaluable[w].sum()) < config.min_cohort_size:
            problems.append(f"only {int(evaluable[w].sum())} cohort hosts are evaluable in {w}")
    if problems:
        raise PipelineError(
            f"когорта слишком мала ({'; '.join(problems)}) при >= {config.min_events_per_window} событий в окне "
            f"и политике {config.cohort_policy!r}; нужно >= {config.min_cohort_size}. "
            "Возьмите окна длиннее или порог ниже.")
    stable = active.all(axis=1)
    train_events = table["train"]
    selection = {
        "policy": config.cohort_policy, "note": SELECTION_NOTE[config.cohort_policy],
        "n_cohort": int(len(cohort)),
        "n_evaluable": {w: int(evaluable[w].sum()) for w in later},
        "coverage": {w: float(evaluable[w].mean()) for w in later},
        "n_hosts_active_in_every_window": int(stable.sum()),
        "median_train_events": {
            "active_in_every_window": float(train_events[stable].median()) if stable.any() else None,
            "not_active_in_every_window": float(train_events[active["train"] & ~stable].median())
            if (active["train"] & ~stable).any() else None},
        "interpretation": "если медианы различаются, выживание хоста связано с его активностью в train, "
                          "и ограничение «стабильными» хостами меняет оцениваемую совокупность",
    }
    cohort.columns = [f"events_{w}" for w in cohort.columns]
    for w in later:
        cohort[f"evaluable_{w}"] = evaluable[w]
    cohort.index.name = "host_id"
    cohort.to_csv(out_dir / "cohort.csv")
    members = set(cohort.index)

    graph_sizes = {}
    for w in names:
        acc = None
        for chunk in pd.read_csv(out_dir / "flows" / f"{w}.csv", chunksize=chunksize,
                                 dtype={"src": "string", "dst": "string"},
                                 usecols=["src", "dst", "bytes", "packets", "resp_bytes", "resp_packets"]):
            chunk = chunk[(chunk.src != chunk.dst) & chunk.src.isin(members) & chunk.dst.isin(members)]
            if chunk.empty:
                continue
            a = np.where(chunk.src < chunk.dst, chunk.src, chunk.dst)
            b = np.where(chunk.src < chunk.dst, chunk.dst, chunk.src)
            g = pd.DataFrame({"host_a": a, "host_b": b, "flows": 1,
                              "bytes": chunk.bytes + chunk.resp_bytes,
                              "packets": chunk.packets + chunk.resp_packets}
                             ).groupby(["host_a", "host_b"]).sum()
            acc = g if acc is None else acc.add(g, fill_value=0)
        edges = (acc if acc is not None else pd.DataFrame(columns=["flows", "bytes", "packets"])).reset_index()
        edges.to_csv(out_dir / "graph" / f"edges_{w}.csv", index=False)
        graph_sizes[w] = int(len(edges))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "data_kind": kind,
        "synthetic": kind == "synthetic_fixture",
        "real_lanl_declared": kind == "lanl_declared",
        "provenance_verified": False,
        "provenance": provenance,
        "source_format": SOURCE_FORMAT,
        "raw_columns": RAW_COLUMNS,
        "canonical_columns": FLOW_COLUMNS,
        "canonical_mapping": {"src": "SrcDevice", "dst": "DstDevice", "dst_port": "DstPort",
                              "proto": "Protocol", "bytes": "SrcBytes", "packets": "SrcPackets",
                              "ts": "Time", "src_port": "SrcPort", "duration": "Duration",
                              "resp_bytes": "DstBytes", "resp_packets": "DstPackets"},
        "time_axis": "Time column units; calendar origin not specified by the source",
        "host_id_rule": "device string as in the file, stripped, case-sensitive",
        "config": config.to_dict(),
        "inputs": [{"file": p.name, "bytes": p.stat().st_size, "sha256": _sha256(p)} for p in inputs],
        "invalid_rows": dict(bad),
        "counts": dict(totals),
        "windows": {w: {"start": config.windows[w][0], "end": config.windows[w][1],
                        "n_flows": n_rows[w], "min_time": float(t_range[w][0]),
                        "max_time": float(t_range[w][1]), "graph_edges": graph_sizes[w]}
                    for w in names},
        "cohort": {"n_hosts": int(len(cohort)), "policy": config.cohort_policy,
                   "rule": (f"active (as src or dst, excluding self flows) >= {config.min_events_per_window} "
                            + ("events in the train window" if config.cohort_policy == "train_only"
                               else "events in EVERY window")),
                   "evaluable": selection["n_evaluable"], "selection": selection,
                   "sha256": _sha256(out_dir / "cohort.csv")},
        "graph_rule": "undirected, one edge per unordered cohort host pair; flows/bytes/packets "
                      "aggregated over the window (bytes/packets include both directions); self loops removed",
        "leakage_checks": {
            "windows_strictly_ordered": True,
            "max_time_train_lt_min_time_test":
                bool(all(t_range["train"][1] < t_range[w][0] for w in names if w != "train")),
            "boundary_straddling_flows_dropped": config.drop_straddling,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("describe", help="scan raw files: Time range, hosts, invalid rows")
    d.add_argument("--netflow", nargs="+", type=Path, required=True)
    d.add_argument("--lenient", action="store_true", help="skip lines with a wrong field count")
    r = sub.add_parser("run", help="build windows, cohort and graphs")
    r.add_argument("--netflow", nargs="+", type=Path, required=True)
    r.add_argument("--config", type=Path, required=True)
    r.add_argument("--output-dir", type=Path, required=True)
    r.add_argument("--chunksize", type=int, default=500_000)
    r.add_argument("--lenient", action="store_true")
    r.add_argument("--data-kind", choices=["lanl", "synthetic_fixture"], default=None,
                   help="DECLARE the origin; default is 'unverified'. 'lanl' also needs --source-note")
    r.add_argument("--source-note", help="where/when the LANL files were obtained (required with --data-kind lanl)")
    r.add_argument("--cohort-policy", choices=["train_only", "all_windows"], default=None,
                   help="override the config's cohort policy")
    r.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "describe":
            print(json.dumps(describe(args.netflow, strict=not args.lenient), indent=2))
        else:
            cfg = PipelineConfig.from_json(args.config)
            if args.cohort_policy:
                cfg = PipelineConfig(**{**cfg.__dict__, "cohort_policy": args.cohort_policy})
            m = prepare(args.netflow, cfg, args.output_dir, chunksize=args.chunksize, strict=not args.lenient,
                        data_kind=args.data_kind, source_note=args.source_note, overwrite=args.overwrite)
            print(json.dumps({k: m[k] for k in ("data_kind", "provenance_verified", "windows", "cohort",
                                                "invalid_rows")}, indent=2))
    except (PipelineError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
