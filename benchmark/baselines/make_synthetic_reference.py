"""СИНТЕТИЧЕСКИЙ reference/features.csv для калибровки бейзлайнов и CI. Не настоящие данные.

    python -m benchmark.baselines.make_synthetic_reference --output-dir /tmp/synthetic_reference
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.knn_consistency import build_reference_features


def make_flows(n_hosts=300, n_roles=5, flows_per_host=200, seed=0):
    """Синтетические потоки: у каждой роли свои порты, протокол, направление, объём и время."""
    rng = np.random.default_rng(seed)
    hosts = [f"h{i}" for i in range(n_hosts)]
    roles = rng.integers(0, n_roles, n_hosts)
    role_ports = [rng.choice([22, 25, 53, 80, 123, 389, 443, 445, 3389, 8080, 1433, 3306],
                             size=3, replace=False) for _ in range(n_roles)]
    role_hour = rng.integers(0, 24, n_roles)
    role_out = np.linspace(0.15, 0.85, n_roles)
    recs = []
    for i, h in enumerate(hosts):
        r = roles[i]
        for _ in range(flows_per_host):
            peer = hosts[rng.integers(n_hosts)]
            if peer == h:
                continue
            outgoing = rng.random() < role_out[r]
            src, dst = (h, peer) if outgoing else (peer, h)
            recs.append((src, dst, rng.choice(role_ports[r]) if rng.random() < .9 else rng.integers(1, 65535),
                         "tcp" if r % 2 == 0 else "udp",
                         rng.lognormal(6 + r, 1), rng.integers(1, 50) * (r + 1),
                         ((role_hour[r] + rng.integers(-2, 3)) % 24) * 3600 + rng.integers(0, 3600)))
    cols = ["src", "dst", "dst_port", "proto", "bytes", "packets", "ts"]
    return pd.DataFrame(recs, columns=cols), pd.Series(roles, index=hosts)


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
