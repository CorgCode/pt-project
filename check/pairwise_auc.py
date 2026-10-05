"""Pairwise ROC-AUC / Average Precision evaluator for host embeddings.

Integration contract: ``embeddings.csv`` has ``device_id,dim_0,...`` and the
pre-built ``reference/pairs.csv`` has ``host_a,host_b,label,split``.  This
module never constructs or re-samples pairs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PAIR_COLUMNS = ("host_a", "host_b", "label", "split")


def _load_embeddings(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "device_id" not in df.columns:
        raise ValueError("embeddings.csv must contain a 'device_id' column")
    if df["device_id"].isna().any() or df["device_id"].duplicated().any():
        raise ValueError("device_id must be present and unique")
    columns = [c for c in df.columns if c != "device_id"]
    if not columns:
        raise ValueError("embeddings.csv must contain at least one vector column")
    try:
        matrix = df[columns].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("all embedding dimensions must be numeric") from exc
    if not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError("embedding dimensions must not contain NaN or infinity")
    return matrix.set_axis(df["device_id"].astype(str), axis="index")


def _load_pairs(path: str | Path, split: str) -> pd.DataFrame:
    pairs = pd.read_csv(path)
    missing = set(PAIR_COLUMNS) - set(pairs.columns)
    if missing:
        raise ValueError(f"pairs.csv is missing columns: {sorted(missing)}")
    pairs = pairs.loc[pairs["split"].astype(str) == str(split), list(PAIR_COLUMNS)].copy()
    if pairs.empty:
        raise ValueError(f"pairs.csv has no rows for split={split!r}")
    pairs["host_a"] = pairs["host_a"].astype(str)
    pairs["host_b"] = pairs["host_b"].astype(str)
    pairs["label"] = pd.to_numeric(pairs["label"], errors="raise")
    if not pairs["label"].isin([0, 1]).all():
        raise ValueError("pair labels must be binary: 0 (different) or 1 (similar)")
    if (pairs["host_a"] == pairs["host_b"]).any():
        raise ValueError("self-pairs are not allowed")
    canonical = np.sort(pairs[["host_a", "host_b"]].to_numpy(str), axis=1)
    if pd.Series(list(map(tuple, canonical))).duplicated().any():
        raise ValueError("pairs.csv contains duplicate undirected pairs in the selected split")
    if pairs["label"].nunique() != 2:
        raise ValueError("the selected split must contain both positive and negative pairs")
    return pairs


def _scores(embeddings: pd.DataFrame, pairs: pd.DataFrame, metric: str) -> np.ndarray:
    a = embeddings.loc[pairs["host_a"]].to_numpy(float)
    b = embeddings.loc[pairs["host_b"]].to_numpy(float)
    if metric == "cosine":
        a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
        b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
        return np.einsum("ij,ij->i", a, b)
    if metric == "euclidean":
        return -np.linalg.norm(a - b, axis=1)
    raise ValueError(f"unsupported metric: {metric!r}")


def evaluate_pairwise(
    embeddings_path: str | Path, pairs_path: str | Path, split: str = "test",
    metric: str = "cosine", *, require_full_coverage: bool = False,
) -> dict:
    """Evaluate one embedding against canonical labelled pairs.

    Missing hosts are reported and excluded by default.  Strict benchmark runs
    should use ``require_full_coverage=True``.
    """
    embeddings = _load_embeddings(embeddings_path)
    pairs = _load_pairs(pairs_path, split)
    covered = pairs["host_a"].isin(embeddings.index) & pairs["host_b"].isin(embeddings.index)
    dropped = int((~covered).sum())
    if require_full_coverage and dropped:
        raise ValueError(f"embedding misses hosts used by {dropped} canonical pairs")
    evaluated = pairs.loc[covered].copy()
    if evaluated.empty or evaluated["label"].nunique() != 2:
        raise ValueError("covered pairs must contain both positive and negative labels")
    scores = _scores(embeddings, evaluated, metric)
    labels = evaluated["label"].to_numpy(int)
    return {
        "split": str(split), "score": metric,
        "n_canonical_pairs": int(len(pairs)), "n_evaluated_pairs": int(len(evaluated)),
        "n_dropped_missing_host": dropped, "coverage": float(len(evaluated) / len(pairs)),
        "positive_rate": float(labels.mean()),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--metric", choices=("cosine", "euclidean"), default="cosine")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output", help="optional JSON result path")
    args = parser.parse_args()
    result = evaluate_pairwise(args.embeddings, args.pairs, args.split, args.metric,
                               require_full_coverage=args.strict)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
