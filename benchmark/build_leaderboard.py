"""Build one CSV leaderboard from benchmark result JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("benchmark/results"))
    parser.add_argument("--output", type=Path, default=Path("benchmark/leaderboard.csv"))
    args = parser.parse_args()

    rows = []
    reference_keys = set()
    fit_keys = set()
    host_counts = set()
    kinds, splits = set(), set()
    for path in sorted(args.results_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        info = data.get("reference_info") or {}
        files = info.get("files_sha256")
        fit_files = info.get("fit_reference_files_sha256")
        # New results fingerprint EVERY reference file; legacy ones only features/neighbors.
        fingerprint = (
            tuple(sorted(files.items())) if files else
            (info.get("features_sha256"), info.get("neighbors_sha256"))
        )
        if fit_files and any(k.startswith("link_learned") for k in data.get("metrics", {})):
            fit_keys.add(tuple(sorted(fit_files.items())))      # learned scores must share ONE fit reference
        if not any(fingerprint):
            raise ValueError(
                f"{path}: no reference fingerprint; re-evaluate this submission "
                "with the current common evaluator before building the leaderboard"
            )
        # Including None prevents silent mixing of legacy / unverified scores.
        reference_keys.add(fingerprint)
        host_counts.add(data.get("n_hosts"))
        kinds.add((info.get("data_kind", "unspecified"), info.get("synthetic")))
        splits.add(data.get("split"))
        row = {"method": data.get("method", path.stem), "n_hosts": data.get("n_hosts")}
        row.update(data.get("metrics", {}))
        rows.append(row)

    if len(reference_keys) > 1:
        raise ValueError(
            "Leaderboard mixes different or unknown reference datasets. "
            "Evaluate every method against the SAME reference directory before comparing."
        )
    if len(fit_keys) > 1:
        raise ValueError("Leaderboard mixes learned link scores fitted on different reference files.")
    if len(host_counts) > 1:
        raise ValueError("Leaderboard mixes different host cohorts.")
    if len(kinds) > 1:
        raise ValueError(f"Leaderboard mixes data kinds (e.g. synthetic and real): {sorted(map(str, kinds))}")
    if len(splits) > 1:
        raise ValueError(f"Leaderboard mixes evaluation splits: {sorted(map(str, splits))}")

    if not rows:
        print("No result JSON files found.")
        return

    df = pd.DataFrame(rows).sort_values("method")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
