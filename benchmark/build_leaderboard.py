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
    for path in sorted(args.results_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        row = {"method": data.get("method", path.stem), "n_hosts": data.get("n_hosts")}
        row.update(data.get("metrics", {}))
        rows.append(row)

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
