"""Compatibility CLI for canonical single-file and multi-window submissions.

Canonical columns are host_id,emb_0,... as documented in BENCHMARK_CONTRACT.md.
This command validates file format and cohort consistency, not model provenance.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from benchmark.validate_submission import validate_submission


def validate_manifest(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(path)
    manifest = pd.read_csv(path, dtype="string")
    required = {"method", "window_id", "path"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Missing manifest columns: {sorted(required - set(manifest.columns))}")
    if manifest.empty or manifest[list(required)].isna().any().any():
        raise ValueError("Empty manifest or blank required entries")
    for column in required:
        if manifest[column].str.strip().eq("").any():
            raise ValueError(f"Blank manifest field: {column}")
    if manifest.duplicated(["method", "window_id"]).any():
        raise ValueError("Duplicate (method,window_id) in manifest")
    cohorts: dict[str, set[str]] = {}
    for row in manifest.itertuples(index=False):
        file = (path.parent / row.path).resolve()
        validate_submission(str(file))
        ids = pd.read_csv(file, dtype={"host_id": "string"})["host_id"]
        current = set(ids.astype(str))
        if row.window_id in cohorts and current != cohorts[row.window_id]:
            raise ValueError(
                f"Window {row.window_id}: incompatible host cohorts across methods; "
                f"missing={sorted(cohorts[row.window_id] - current)[:5]}, "
                f"extra={sorted(current - cohorts[row.window_id])[:5]}"
            )
        cohorts[row.window_id] = current
    print(f"OK: {len(manifest)} method-window submissions; {len(cohorts)} windows")
    return len(manifest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("submission", nargs="?", type=Path)
    ap.add_argument("--manifest", type=Path)
    args = ap.parse_args()
    if (args.submission is None) == (args.manifest is None):
        ap.error("Provide either a submission file or --manifest, not both")
    if args.manifest:
        validate_manifest(args.manifest)
    else:
        validate_submission(str(args.submission))


if __name__ == "__main__":
    main()
