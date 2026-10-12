"""Generate a SYNTHETIC netflow fixture in the exact LANL file format.

For tests, CI and pipeline smoke runs ONLY. The files are written in the documented
LANL layout (headerless, 11 comma-separated fields, bzip2), together with a
SYNTHETIC_FIXTURE marker so downstream artifacts are always labelled synthetic.
Nothing here represents LANL traffic statistics.

    python -m pipeline.fixtures --output-dir data/fixture --n-hosts 120 --seed 0
"""
from __future__ import annotations

import argparse
import bz2
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.provenance import SYNTHETIC_MARKER

ROLE_PORTS = [(443, 80), (22, 3389), (53, 123), (445, 389), (25, 587)]
WELL_KNOWN = {22, 25, 53, 80, 123, 389, 443, 445, 587, 3389}


def _port(p: int) -> str:
    return str(p) if p in WELL_KNOWN else f"Port{p}"


def generate(output_dir: Path, n_hosts: int = 300, hours: float = 12.0, files: int = 3,
             seed: int = 0, flows_per_host_hour: float = 30.0) -> dict:
    """Write netflow_day-NN.bz2 files covering [0, hours*3600) seconds; return metadata."""
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / SYNTHETIC_MARKER).write_text("synthetic data, not LANL traffic\n")
    hosts = np.array([f"Comp{100000 + i * 7:06d}" for i in range(n_hosts)])
    n_roles = len(ROLE_PORTS)
    roles = rng.integers(0, n_roles, n_hosts)
    out_share = np.linspace(0.2, 0.9, n_roles)[roles]
    volume_mu = (5 + 0.7 * roles)
    # Each role has a few shared "servers"; hosts talk mostly to their role's servers plus a
    # few private peers (persistent pools, with some turnover halfway through).
    servers = [rng.choice(n_hosts, 8, replace=False) for _ in range(n_roles)]
    pools = np.stack([np.concatenate([rng.choice(servers[roles[i]], 5, replace=False),
                                      rng.integers(0, n_hosts, 3)]) for i in range(n_hosts)])
    pools2 = pools.copy()
    swap = rng.random(pools.shape) < 0.25
    pools2[swap] = rng.integers(0, n_hosts, swap.sum())
    total_s = hours * 3600.0
    rows = []
    for i in range(n_hosts):
        n = rng.poisson(flows_per_host_hour * hours * (0.6 + 0.2 * (roles[i] % 3)))
        t = rng.uniform(0, total_s, n)
        # role-specific activity phase within a 4h cycle
        keep = rng.random(n) < (0.5 + 0.5 * np.cos(2 * np.pi * (t / 14400.0 - roles[i] / n_roles)) ** 2)
        t = t[keep]
        n = len(t)
        pool = np.where((t < total_s / 2)[:, None], pools[i], pools2[i])
        peer = np.where(rng.random(n) < 0.9, pool[np.arange(n), rng.integers(0, 8, n)],
                        rng.integers(0, n_hosts, n))
        outgoing = rng.random(n) < out_share[i]
        src = np.where(outgoing, i, peer)
        dst = np.where(outgoing, peer, i)
        dp = np.where(rng.random(n) < 0.9, np.array(ROLE_PORTS[roles[i]])[rng.integers(0, 2, n)],
                      rng.integers(1024, 65535, n))
        sb = np.rint(rng.lognormal(volume_mu[i], 1.0, n))
        rows.append(pd.DataFrame({
            "Time": np.floor(t), "Duration": np.rint(rng.exponential(2.0, n)),
            "SrcDevice": hosts[src], "DstDevice": hosts[dst],
            "Protocol": np.where(roles[i] % 2 == 0, "6", "17"),
            "SrcPort": [f"Port{p}" for p in rng.integers(1024, 65535, n)],
            "DstPort": [_port(int(p)) for p in dp],
            "SrcPackets": np.maximum(1, np.rint(sb / 500)), "DstPackets": np.maximum(1, np.rint(sb / 800)),
            "SrcBytes": sb, "DstBytes": np.rint(sb * 0.6)}))
    flows = pd.concat(rows).sort_values(["Time", "SrcDevice", "DstDevice"], kind="stable")
    edges = np.linspace(0, total_s, files + 1)
    for k in range(files):
        part = flows[(flows.Time >= edges[k]) & (flows.Time < edges[k + 1] + (k == files - 1))]
        text = part.to_csv(header=False, index=False, float_format="%.0f")
        with bz2.open(output_dir / f"netflow_day-{k + 1:02d}.bz2", "wt") as f:
            f.write(text)
    roles_df = pd.DataFrame({"host_id": hosts, "target": [f"role_{r}" for r in roles]})
    roles_df.to_csv(output_dir / "fixture_roles.csv", index=False)
    return {"n_hosts": n_hosts, "hours": hours, "files": files, "rows": int(len(flows)), "seed": seed}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-hosts", type=int, default=300)
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--files", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(generate(a.output_dir, a.n_hosts, a.hours, a.files, a.seed))


if __name__ == "__main__":
    main()
