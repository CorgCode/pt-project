"""Future-link prediction with ONE frozen candidate pool shared by all methods.

This evaluates presence/absence of a communication edge in a *later* window.
It is NOT a classification of host functional roles.

Usage:
  python -m evaluation.future_links build --source-flows train.csv \
    --target-flows test.csv --hosts embeddings.csv --output reference/edges.csv
  python -m evaluation.future_links score --embeddings embeddings.csv \
    --edges reference/edges.csv --output results/link_prediction.json

Required canonical flows: src,dst,dst_port,proto,bytes,packets,ts.
Undirected edge = >=1 observed flow in either direction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from evaluation.temporal_reference import load_flows
from evaluation.evaluate import _load_matrix


FIELDS = ["host_a", "host_b", "source_window", "target_window", "label", "split"]


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def observed_edges(flows: pd.DataFrame, cohort: set[str]) -> set[tuple[str, str]]:
    return {
        _pair(str(a), str(b))
        for a, b in zip(flows["src"], flows["dst"])
        if a != b and a in cohort and b in cohort
    }


def build_edges(
    source: pd.DataFrame,
    target: pd.DataFrame,
    hosts: list[str],
    *,
    seed: int = 0,
    negatives_per_positive: int = 1,
    split: str = "test",
) -> pd.DataFrame:
    """Generate deterministic balanced-ish candidate pairs for an explicit cohort.

    No per-method candidate generation. Uses rejection sampling among all
    possible unordered host pairs to avoid O(n^2) candidate enumeration.
    Sampling is conditional on this population; PR-AUC is NOT deployment AP.
    """
    if not 0 < negatives_per_positive <= 100:
        raise ValueError("negatives_per_positive must be from 1 to 100")
    if len(hosts) != len(set(hosts)) or len(hosts) < 3 or any(not h for h in hosts):
        raise ValueError("At least 3 unique, nonblank hosts required")
    if target["ts"].min() <= source["ts"].max():
        raise ValueError("Source/target windows overlap or touch")
    if split not in {"test", "validation"}:
        raise ValueError("Use validation or test as split")
    hosts = sorted(hosts)
    population = set(hosts)
    positives = sorted(observed_edges(target, population))
    if not positives:
        raise ValueError("No positive target edges for the cohort")
    all_pairs_count = len(hosts) * (len(hosts) - 1) // 2
    remaining = all_pairs_count - len(positives)
    requested = len(positives) * negatives_per_positive
    if remaining < requested:
        raise ValueError(
            f"Insufficient negative pairs: requested={requested}, possible={remaining}; "
            "reduce negatives_per_positive or adjust cohort"
        )
    rng = np.random.default_rng(seed)
    negatives: set[tuple[str, str]] = set()
    pos_set = set(positives)
    # Rejection sampling is fast when candidate negatives are not too dense.
    # Sampling never relies on embedding scores.
    retries = 0
    max_retries = max(10000, requested * 100)
    while len(negatives) < requested:
        a, b = rng.choice(len(hosts), size=2, replace=False)
        pair = _pair(hosts[int(a)], hosts[int(b)])
        if pair not in pos_set:
            negatives.add(pair)
        retries += 1
        if retries > max_retries:
            # Near-exhausted space: an explicit enumeration is only acceptable
            # for sufficiently small networks.
            if all_pairs_count > 1_000_000:
                raise RuntimeError(
                    "Rejection sampling exhausted. Reduce negative ratio or "
                    "use a sparse candidate-pool strategy."
                )
            for i, a in enumerate(hosts):
                for b in hosts[i + 1:]:
                    pair = (a, b)
                    if pair not in pos_set:
                        negatives.add(pair)
                    if len(negatives) >= requested:
                        break
                if len(negatives) >= requested:
                    break
            break

    source_window = (
        source["ts"].min().isoformat() + "Z/" +
        source["ts"].max().isoformat() + "Z"
    )
    target_window = (
        target["ts"].min().isoformat() + "Z/" +
        target["ts"].max().isoformat() + "Z"
    )
    rows = [
        (a, b, source_window, target_window, 1, split)
        for a, b in positives
    ] + [
        (a, b, source_window, target_window, 0, split)
        for a, b in sorted(negatives)
    ]
    out = pd.DataFrame(rows, columns=FIELDS)
    # Persisting links (already seen in the source window) are far easier to predict
    # than genuinely new ones; keep the flag so both can be reported separately.
    seen = observed_edges(source, population)
    out["seen_in_source"] = [
        _pair(a, b) in seen for a, b in zip(out["host_a"], out["host_b"])
    ]
    return out


def load_candidate_edges(file: Path) -> pd.DataFrame:
    df = pd.read_csv(
        file, dtype={"host_a": "string", "host_b": "string",
                     "source_window": "string", "target_window": "string",
                     "split": "string", "label": "string"},
    )
    missing = set(FIELDS) - set(df.columns)
    if missing:
        raise ValueError(f"edges.csv missing columns {sorted(missing)}")
    if df.empty or df[FIELDS].isna().any().any():
        raise ValueError("edges.csv empty or has missing values")
    if not df["label"].isin(["0", "1"]).all():
        raise ValueError("Edge labels must be 0/1")
    if not df["split"].isin(["test", "validation"]).all():
        raise ValueError("Unexpected split")
    if not (df.host_a < df.host_b).all():
        raise ValueError("Pairs must be canonical, distinct and undirected")
    if df.duplicated(["host_a", "host_b", "source_window", "target_window", "split"]).any():
        raise ValueError("Duplicate candidate edges")
    return df


def score_embeddings(
    embedding: pd.DataFrame, edges: pd.DataFrame,
    *, split: str = "test",
) -> dict:
    """Score EXACT saved candidates. Unknown hosts are an error, not filtered."""
    edges = edges[edges["split"] == split]
    if edges.empty:
        raise ValueError(f"No candidate edges for split={split}")
    if edges[["source_window", "target_window"]].drop_duplicates().shape[0] != 1:
        raise ValueError("Do not mix source/target window pairs in an evaluation")
    if embedding.index.has_duplicates:
        raise ValueError("Duplicate host IDs")
    known = set(embedding.index)
    used = set(edges["host_a"]) | set(edges["host_b"])
    if not used.issubset(known):
        raise ValueError(f"Embedding lacks candidate hosts: {sorted(used - known)[:5]}")
    labels = edges["label"].astype(int).to_numpy()
    if len(np.unique(labels)) < 2:
        raise ValueError("Need both positive and negative candidates")
    emb = embedding.to_numpy(dtype=float)
    if not np.isfinite(emb).all():
        raise ValueError("Embedding contains nonfinite coordinates")
    if (np.linalg.norm(emb, axis=1) == 0).any():
        raise ValueError("Zero-norm host vectors make cosine undefined")
    index = {host: j for j, host in enumerate(embedding.index)}
    a = np.array([index[x] for x in edges["host_a"]])
    b = np.array([index[x] for x in edges["host_b"]])
    # Vectorized per-pair cosine: avoids constructing a full N*N similarity matrix.
    norms = np.linalg.norm(emb, axis=1)
    cosine = np.einsum("ij,ij->i", emb[a], emb[b]) / (norms[a] * norms[b])
    window = edges.iloc[0]
    return {
        "roc_auc": float(roc_auc_score(labels, cosine)),
        "pr_auc": float(average_precision_score(labels, cosine)),
        "n_examples": int(len(labels)),
        "n_positive": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "n_candidate_hosts": len(used),
        "n_embedding_hosts": len(known),
        "host_coverage": float(len(used) / len(known)),
        "split": split,
        "source_window": window["source_window"],
        "target_window": window["target_window"],
        "interpretation": (
            "PR-AUC depends on the frozen sampled candidate distribution; "
            "this is NOT full-population deployment average precision."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sp = ap.add_subparsers(dest="command", required=True)
    build = sp.add_parser("build")
    build.add_argument("--source-flows", type=Path, required=True)
    build.add_argument("--target-flows", type=Path, required=True)
    build.add_argument("--hosts", type=Path, required=True,
                       help="Canonical embeddings.csv used ONLY to define fixed host cohort")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--negatives-per-positive", type=int, default=1)
    score = sp.add_parser("score")
    score.add_argument("--embeddings", type=Path, required=True)
    score.add_argument("--edges", type=Path, required=True)
    score.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.command == "build":
        cohort = _load_matrix(args.hosts, "host_id", "emb_").index.astype(str).tolist()
        df = build_edges(load_flows(args.source_flows), load_flows(args.target_flows),
                         cohort, seed=args.seed,
                         negatives_per_positive=args.negatives_per_positive)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Saved {len(df)} candidate edges to {args.output}")
    else:
        emb = _load_matrix(args.embeddings, "host_id", "emb_")
        edges = load_candidate_edges(args.edges)
        report = score_embeddings(emb, edges)
        report["edges_sha256"] = hashlib.sha256(args.edges.read_bytes()).hexdigest()
        report["ground_truth_host_roles"] = False
        output = json.dumps(report, indent=2, ensure_ascii=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output + "\n", encoding="utf-8")
        print(output)


if __name__ == "__main__":
    main()
