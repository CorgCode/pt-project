"""Write a SYNTHETIC reference/features.csv for calibrating baselines and CI.

Not real data and not a substitute for the project reference artifacts (#19).

    python -m benchmark.baselines.make_synthetic_reference --output-dir /tmp/synthetic_reference
"""
from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.demo_knn_consistency import make_flows
from evaluation.knn_consistency import build_reference_features


def make(output_dir: Path, n_hosts: int = 300, seed: int = 0) -> Path:
    flows, roles = make_flows(n_hosts=n_hosts, flows_per_host=120, seed=seed)
    feats = build_reference_features(flows, hosts=list(roles.index))
    feats.columns = [f"ref_{i}" for i in range(feats.shape[1])]
    feats.index.name = "host_id"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "features.csv"
    feats.to_csv(path)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-hosts", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(make(a.output_dir, a.n_hosts, a.seed))
