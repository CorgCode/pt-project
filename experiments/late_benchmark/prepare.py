"""Extend a frozen source cohort with the predeclared late test windows."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.lanl_netflow import canonicalize, iter_raw_chunks
from pipeline.prepare import _sha256, SELECTION_NOTE
from pipeline.windows import PipelineConfig, window_mask

PROTOCOL = Path(__file__).with_name("protocol.json")


def load_protocol(path=PROTOCOL):
    spec = json.loads(Path(path).read_text())
    PipelineConfig.from_json(path)
    if set(spec["test_days"].values()) != {88, 89, 90}:
        raise ValueError("the final benchmark must include days 88, 89 and 90")
    for name, day in spec["test_days"].items():
        if spec["windows"][name] != [(day - 1) * 86400 + 45000, (day - 1) * 86400 + 48600]:
            raise ValueError("test windows must be predeclared 12:30–13:30")
    if spec["cohort_policy"] != "train_only" or spec["windows"]["train"] != [126000, 129600]:
        raise ValueError("source cohort and training window must remain frozen")
    return spec


def extract_window(raw, output, interval, hosts, chunksize=250_000):
    """Scan through EOF (including bzip2 CRC); keep no uncompressed full-day copy."""
    raw, output = Path(raw), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial.csv")
    if temporary.exists():
        temporary.unlink()  # only an incomplete output of this extraction
    members = set(hosts)
    events, bad, totals = Counter(), Counter(), Counter()
    graph = None
    lower, upper = np.inf, -np.inf
    wrote = False
    for chunk in iter_raw_chunks(raw, chunksize=chunksize):
        totals["raw_rows"] += len(chunk)
        ts = pd.to_numeric(chunk.Time, errors="coerce").to_numpy()
        duration = pd.to_numeric(chunk.Duration, errors="coerce").to_numpy()
        inside = (ts >= interval[0]) & (ts < interval[1])
        keep = window_mask(ts, duration, interval, True)
        totals["straddling_dropped"] += int((inside & ~keep).sum())
        if keep.any():
            flows, invalid = canonicalize(chunk.loc[keep])
            bad.update(invalid)
            if not flows.empty:
                flows.to_csv(temporary, index=False, mode="a" if wrote else "w", header=not wrote)
                wrote = True
                totals["n_flows"] += len(flows)
                lower, upper = min(lower, float(flows.ts.min())), max(upper, float(flows.ts.max()))
                real = flows.loc[flows.src.ne(flows.dst)]
                totals["self_flows"] += len(flows) - len(real)
                events.update(pd.concat([real.src, real.dst]).value_counts().to_dict())
                f = real.loc[real.src.isin(members) & real.dst.isin(members)]
                a, b = np.minimum(f.src.to_numpy(), f.dst.to_numpy()), np.maximum(f.src.to_numpy(), f.dst.to_numpy())
                grouped = pd.DataFrame({"host_a": a, "host_b": b, "flows": 1,
                    "bytes": (f.bytes + f.resp_bytes).to_numpy(),
                    "packets": (f.packets + f.resp_packets).to_numpy()}).groupby(["host_a", "host_b"]).sum()
                graph = grouped if graph is None else graph.add(grouped, fill_value=0)
        if totals["raw_rows"] % 10_000_000 < chunksize:
            print(f"{raw.name}: scanned {totals['raw_rows']:,}, kept {totals['n_flows']:,}", flush=True)
    if not wrote:
        raise ValueError(f"no flows in the predeclared window of {raw.name}")
    temporary.replace(output)
    if graph is None:
        graph = pd.DataFrame(columns=["flows", "bytes", "packets"],
                             index=pd.MultiIndex.from_tuples([], names=["host_a", "host_b"]))
    return {"start": interval[0], "end": interval[1], "n_flows": totals["n_flows"],
            "min_time": lower, "max_time": upper, "graph_edges": len(graph),
            "raw_rows": totals["raw_rows"], "straddling_dropped": totals["straddling_dropped"],
            "self_flows": totals["self_flows"], "invalid_rows_in_window": dict(bad),
            "n_hosts_in_window": len(events), "n_hosts_outside_source_cohort": len(set(events) - members),
            "raw_sha256": _sha256(raw), "flows_sha256": _sha256(output)}, events, graph


def prepare_target(raw, output, name, interval, hosts, cohort_sha, chunksize):
    raw, output = Path(raw), Path(output)
    if not raw.is_file():
        raise FileNotFoundError(f"missing completed archive: {raw.name}")
    receipt = output / f"extraction_{name}.json"
    target = output / "flows" / f"{name}.csv"
    graph_path = output / "graph" / f"edges_{name}.csv"
    if receipt.exists() and target.exists() and graph_path.exists():
        saved = json.loads(receipt.read_text())
        if saved["interval"] != interval or saved["base_cohort_sha256"] != cohort_sha:
            raise ValueError("extraction cache belongs to a different protocol/cohort")
        if saved["stats"]["flows_sha256"] != _sha256(target) or saved["stats"]["raw_sha256"] != _sha256(raw):
            raise ValueError("raw archive or extracted window changed")
        return saved["stats"], saved["events"]
    stats, events, graph = extract_window(raw, target, interval, hosts, chunksize)
    graph.reset_index().to_csv(graph_path, index=False)
    receipt.write_text(json.dumps({"interval": interval, "stats": stats,
        "base_cohort_sha256": cohort_sha, "events": dict(events)}, indent=2))
    return stats, events


def prepare(base, raw_dir, output, protocol=PROTOCOL, chunksize=250_000, workers=1):
    base, raw_dir, output = Path(base), Path(raw_dir), Path(output)
    spec = load_protocol(protocol)
    original = json.loads((base / "manifest.json").read_text())
    if original["cohort"]["policy"] != "train_only":
        raise ValueError("base cohort must have been selected on train only")
    for name in ("train", "validation"):
        span = original["windows"][name]
        if [span["start"], span["end"]] != spec["windows"][name]:
            raise ValueError("base train/validation windows do not match the frozen protocol")
    source = pd.read_csv(base / "cohort.csv", dtype={"host_id": str}).set_index("host_id")
    if source.index.has_duplicates:
        raise ValueError("duplicate hosts in the source cohort")
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved.get("benchmark_id") != spec["benchmark_id"] or saved.get("protocol_sha256") != _sha256(Path(protocol)):
            raise ValueError("output belongs to a different benchmark")
        if saved.get("base_manifest_sha256") != _sha256(base / "manifest.json"):
            raise ValueError("frozen source manifest changed")
        if saved["cohort"]["sha256"] != _sha256(output / "cohort.csv"):
            raise ValueError("prepared cohort changed")
        expected = {**saved["source_files_sha256"],
                    **{f"flows/{w}.csv": saved["windows"][w]["flows_sha256"] for w in spec["test_days"]}}
        for relative, digest in expected.items():
            if _sha256(output / relative) != digest:
                raise ValueError(f"frozen processed window changed: {relative}")
        return saved
    (output / "flows").mkdir(parents=True, exist_ok=True)
    (output / "graph").mkdir(exist_ok=True)
    cohort = source[["events_train", "events_validation", "evaluable_validation"]].copy()
    windows = {name: copy.deepcopy(original["windows"][name]) for name in ("train", "validation")}
    base_hashes = {}
    for name in windows:
        for relative in (f"flows/{name}.csv", f"graph/edges_{name}.csv"):
            origin = (base / relative).resolve()
            destination = output / relative
            if destination.is_symlink() and destination.resolve() != origin:
                raise ValueError("existing source link points to a different dataset")
            if not destination.exists():
                destination.symlink_to(origin)
        base_hashes[f"flows/{name}.csv"] = _sha256(base / "flows" / f"{name}.csv")
    receipts = {}
    if not 1 <= workers <= 3:
        raise ValueError("workers must be between one and three")
    cohort_sha = _sha256(base / "cohort.csv")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {name: pool.submit(prepare_target, raw_dir / f"netflow_day-{day:02d}.bz2", output,
                    name, spec["windows"][name], source.index.tolist(), cohort_sha, chunksize)
                for name, day in spec["test_days"].items()}
        for name, day in spec["test_days"].items():
            raw = raw_dir / f"netflow_day-{day:02d}.bz2"
            stats, events = jobs[name].result()
            counts = pd.Series(events, dtype=float).reindex(cohort.index, fill_value=0).astype(int)
            cohort[f"events_{name}"] = counts
            cohort[f"evaluable_{name}"] = counts >= spec["min_events_per_window"]
            if cohort[f"evaluable_{name}"].sum() < spec["min_cohort_size"]:
                raise ValueError(f"day {day}: too few source-cohort hosts; predeclared thresholds remain unchanged")
            windows[name] = stats
            receipts[name] = {"file": raw.name, "sha256": stats["raw_sha256"], "bytes": raw.stat().st_size}
    cohort.to_csv(output / "cohort.csv")
    coverage = {name: float(cohort[f"evaluable_{name}"].mean()) for name in windows if name != "train"}
    evaluable = {name: int(cohort[f"evaluable_{name}"].sum()) for name in coverage}
    manifest = copy.deepcopy(original)
    manifest.update(benchmark_id=spec["benchmark_id"], protocol_sha256=_sha256(Path(protocol)),
        windows=windows, inputs=list(receipts.values()), counts={}, invalid_rows={},
        base_manifest_sha256=_sha256(base / "manifest.json"), source_files_sha256=base_hashes,
        config=PipelineConfig.from_json(protocol).to_dict(),
        cohort={"n_hosts": len(cohort), "policy": "train_only", "rule": "fixed source-day02 cohort, >=300 events; late activity only determines per-window evaluability",
                "sha256": _sha256(output / "cohort.csv"), "source_cohort_sha256": _sha256(base / "cohort.csv"),
                "evaluable": evaluable, "selection": {"policy": "train_only", "note": SELECTION_NOTE["train_only"],
                "n_cohort": len(cohort), "n_evaluable": evaluable, "coverage": coverage}},
        provenance={"kind": original["data_kind"], "source_note": "LANL download form; user-provided access link; train/validation reused unchanged, test days 88–90",
                    "verified": False}, provenance_verified=False)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("PT_PROJECT_ML/late_test/processed"))
    parser.add_argument("--raw", type=Path, default=Path("data/late_benchmark/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/late_benchmark/processed"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    result = prepare(args.base, args.raw, args.output, workers=args.workers)
    print(json.dumps({"benchmark_id": result["benchmark_id"], "cohort": result["cohort"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
