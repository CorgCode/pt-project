"""Create a deterministic temporal identity-pair reference shared by all methods.

Flow CSVs must contain src,dst (one row per flow). Embeddings use the canonical
host_id,emb_* schema described in BENCHMARK_CONTRACT.md.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from evaluation.io import load_embeddings

COLUMNS = ["window_a", "host_a", "window_b", "host_b", "label", "split"]
SPLITS = ("train", "validation", "test")


def flow_counts(path: Path) -> pd.Series:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = {"src", "dst"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: expected columns src,dst; missing {sorted(missing)}")
    ids = pd.concat([df["src"], df["dst"]], ignore_index=True)
    ids = ids[ids.str.strip().ne("")]
    return ids.value_counts()


def common_cohort(flows_a: Path, flows_b: Path, embeddings_a: dict[str, Path],
                  embeddings_b: dict[str, Path], min_flows: int) -> list[str]:
    if min_flows < 1:
        raise ValueError("min_flows must be >= 1")
    if not embeddings_a or set(embeddings_a) != set(embeddings_b):
        raise ValueError("supply the same non-empty set of methods for both windows")
    ca, cb = flow_counts(flows_a), flow_counts(flows_b)
    cohort = set(ca[ca >= min_flows].index) & set(cb[cb >= min_flows].index)
    for method in sorted(embeddings_a):
        a, b = load_embeddings(embeddings_a[method]), load_embeddings(embeddings_b[method])
        cohort &= set(a.index) & set(b.index)
    return sorted(cohort)


def build_pairs(hosts: list[str], window_a: str, window_b: str, *,
                negatives_per_positive: int = 5, seed: int = 41,
                train_fraction: float = 0.5, validation_fraction: float = 0.2) -> pd.DataFrame:
    """Split hosts first, then make same-ID positives and different-ID negatives.

    The exact output should be generated once and reused for every embedding method.
    Split cohorts are disjoint, avoiding host leakage between train/validation/test.
    """
    hosts = sorted(set(map(str, hosts)))
    if len(hosts) < 10:
        raise ValueError(f"need at least 10 eligible hosts; got {len(hosts)}")
    if negatives_per_positive < 1:
        raise ValueError("negatives_per_positive must be >= 1")
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1
            and train_fraction + validation_fraction < 1):
        raise ValueError("train_fraction and validation_fraction must be in (0,1) and sum to < 1")
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(hosts, dtype=object)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = max(2, int(n * train_fraction))
    n_val = max(2, int(n * validation_fraction))
    if n_train + n_val >= n:
        n_train, n_val = max(2, int(n * .6)), max(2, int(n * .2))
    cohorts = {"train": list(shuffled[:n_train]),
               "validation": list(shuffled[n_train:n_train+n_val]),
               "test": list(shuffled[n_train+n_val:])}
    rows = []
    for split in SPLITS:
        cohort = sorted(cohorts[split])
        if len(cohort) < 2:
            raise ValueError(f"split {split!r} has fewer than two hosts; use a larger cohort")
        for host in cohort:
            rows.append((window_a, host, window_b, host, 1, split))
            others = [x for x in cohort if x != host]
            take = min(negatives_per_positive, len(others))
            for other in rng.choice(np.asarray(others, dtype=object), size=take, replace=False):
                rows.append((window_a, host, window_b, str(other), 0, split))
    return pd.DataFrame(rows, columns=COLUMNS).sort_values(
        ["split", "host_a", "label", "host_b"], ascending=[True, True, False, True],
        kind="stable").reset_index(drop=True)


def method_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected METHOD=PATH")
    method, path = value.split("=", 1)
    if not method.strip() or not path.strip():
        raise argparse.ArgumentTypeError("expected non-empty METHOD=PATH")
    return method.strip(), Path(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flows-a", type=Path, required=True)
    ap.add_argument("--flows-b", type=Path, required=True)
    ap.add_argument("--embeddings-a", action="append", type=method_path, required=True, metavar="METHOD=PATH")
    ap.add_argument("--embeddings-b", action="append", type=method_path, required=True, metavar="METHOD=PATH")
    ap.add_argument("--window-a", required=True)
    ap.add_argument("--window-b", required=True)
    ap.add_argument("--min-flows", type=int, default=1)
    ap.add_argument("--negatives-per-positive", type=int, default=5)
    ap.add_argument("--train-fraction", type=float, default=.5)
    ap.add_argument("--validation-fraction", type=float, default=.2)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--output", type=Path, default=Path("reference/temporal_pairs.csv"))
    args = ap.parse_args()
    ea, eb = dict(args.embeddings_a), dict(args.embeddings_b)
    if len(ea) != len(args.embeddings_a) or len(eb) != len(args.embeddings_b):
        ap.error("duplicate method name")
    hosts = common_cohort(args.flows_a, args.flows_b, ea, eb, args.min_flows)
    pairs = build_pairs(hosts, args.window_a, args.window_b,
                        negatives_per_positive=args.negatives_per_positive, seed=args.seed,
                        train_fraction=args.train_fraction, validation_fraction=args.validation_fraction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output, index=False)
    print(f"Wrote {len(pairs)} pairs for {len(hosts)} eligible hosts to {args.output}")
    print("hosts by split:", pairs.groupby("split")["host_a"].nunique().to_dict())

if __name__ == "__main__":
    main()
