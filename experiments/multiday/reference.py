"""Общий эталон нового дня и фиксированная по train дня02 когорта."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import pandas as pd

from pipeline.data import _read_cached


def pair_windows(base: Path, target: Path, out: Path, window: str):
    if window not in ("validation", "test"):
        raise ValueError("Ожидается validation или test")
    (out / "flows").mkdir(parents=True, exist_ok=True)
    (out / "graph").mkdir(exist_ok=True)
    source = pd.read_csv(base / "cohort.csv", dtype={"host_id": str}).set_index("host_id")
    later = pd.read_csv(target / "cohort.csv", dtype={"host_id": str}).set_index("host_id")
    if not set(source.index) <= set(later.index):
        raise ValueError("В новом дне отсутствует часть фиксированной когорты")
    cohort = source[["events_train"]].copy()
    cohort[f"events_{window}"] = later.loc[cohort.index, "events_test"]
    cohort[f"evaluable_{window}"] = later.loc[cohort.index, "evaluable_test"]
    cohort.to_csv(out / "cohort.csv")
    for src, dst in [(base / "flows/train.csv", out / "flows/train.csv"),
                     (target / "flows/test.csv", out / "flows" / f"{window}.csv"),
                     (base / "graph/edges_train.csv", out / "graph/edges_train.csv")]:
        if not dst.exists():
            dst.symlink_to(src.resolve())
    m = copy.deepcopy(json.loads((base / "manifest.json").read_text()))
    t = json.loads((target / "manifest.json").read_text())
    m["windows"] = {"train": m["windows"]["train"], window: t["windows"]["test"]}
    m["config"]["windows"] = {w: [v["start"], v["end"]] for w, v in m["windows"].items()}
    m["cohort"] = {"policy": "train_only", "n_hosts": len(cohort), "source": str(base / "cohort.csv"),
                   "sha256": hashlib.sha256((out / "cohort.csv").read_bytes()).hexdigest()}
    m["later_day_provenance"] = t.get("provenance")
    (out / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=2) + "\n")


def build_reference(base: Path, target: Path, out: Path, reference: Path, window: str):
    from pipeline import reference as pipeline_reference
    pair_windows(base, target, out, window)
    completed = reference / window / "manifest.json"
    if not completed.exists():
        original_loader = pipeline_reference.load_flows
        def compact_load(path):
            flows = _read_cached(path)
            flows["ts"] = pd.to_datetime(flows.ts, unit="s")
            return flows
        try:
            pipeline_reference.load_flows = compact_load
            pipeline_reference.build_reference(out, reference, pairs_per_anchor=10)
        finally:
            pipeline_reference.load_flows = original_loader
    manifest = json.loads(completed.read_text())
    # Это та же нормировка относительно persistence, что в исходной методологии.
    # Для новых дней порог берётся из замороженного эталона до оценки кандидатов.
    ideal = float(manifest["reference_neighbor_stability"]["consistency@10"])
    chance = 10 / (manifest["n_hosts"] - 1)
    if not chance < ideal <= 1.0:
        raise ValueError("Persistence не выше случайного уровня; нормировка Neighbors неприменима")
    manifest["score_neighbors_ideal"] = ideal
    manifest["score_normalization_source"] = "frozen source-behavior persistence; independent of candidate embeddings"
    completed.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Готов общий эталон {window}: {manifest['n_hosts']} хостов; persistence {ideal:.5f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--window", choices=["validation", "test"], required=True)
    args = ap.parse_args()
    build_reference(args.base, args.target, args.out, args.reference, args.window)


if __name__ == "__main__":
    main()
