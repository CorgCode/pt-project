"""Synthetic check: a good embedding must beat a random one. Run: python demo_knn_consistency.py"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from knn_consistency import build_reference_features, evaluate_embedding


def make_flows(n_hosts=300, n_roles=5, flows_per_host=200, seed=0):
    """Hosts with role-specific port/proto/direction/volume/time profiles."""
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


if __name__ == "__main__":
    flows, roles = make_flows()
    ref = build_reference_features(flows, hosts=list(roles.index))
    rng = np.random.default_rng(1)
    onehot = np.eye(roles.max() + 1)[roles.values]
    embeddings = {
        "good (role + noise)": pd.DataFrame(onehot + rng.normal(0, .3, onehot.shape), index=roles.index),
        "weak (role + heavy noise)": pd.DataFrame(onehot + rng.normal(0, 1.5, onehot.shape), index=roles.index),
        "oracle (PCA of reference; leaks by design, ceiling check)":
            pd.DataFrame(PCA(16, random_state=0).fit_transform(ref.values), index=roles.index),
        "random": pd.DataFrame(rng.normal(size=(len(roles), 16)), index=roles.index),
    }
    results = {name: evaluate_embedding(e, ref) for name, e in embeddings.items()}
    for name, res in results.items():
        print(f"\n== {name} ==")
        print(res.round(3).to_string())
    c10 = {n: r.loc[(10, "consistency"), "mean"] for n, r in results.items()}
    names = list(c10)
    good, weak, oracle, rand = (c10[names[j]] for j in (0, 1, 2, 3))
    chance = 10 / (len(roles) - 1)
    assert oracle > good > weak > rand, c10
    assert good > 3 * rand and abs(rand - chance) < 0.03, (good, rand, chance)
    print(f"\nOK: Consistency@10 oracle={oracle:.3f} > good={good:.3f} > weak={weak:.3f} "
          f"> random={rand:.3f} (chance ~ {chance:.3f})")
