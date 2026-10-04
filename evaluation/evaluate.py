"""Common CLI for evaluating host embeddings with the shared benchmark contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluation.knn_consistency import evaluate_embedding


def _load_matrix(path: Path, id_col: str, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if id_col not in df.columns:
        raise ValueError(f"{path}: missing required column {id_col!r}")
    if df[id_col].duplicated().any():
        raise ValueError(f"{path}: duplicate {id_col} values")
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"{path}: no columns with prefix {prefix!r}")
    matrix = df.set_index(id_col)[cols]
    if matrix.isna().any().any():
        raise ValueError(f"{path}: NaN values are not allowed")
    return matrix


def run(embeddings_path: Path, reference_dir: Path, method: str) -> dict:
    emb = _load_matrix(embeddings_path, "host_id", "emb_")
    result = {
        "method": method,
        "submission": str(embeddings_path),
        "n_hosts": int(len(emb)),
        "metrics": {},
        "unavailable_metrics": [],
    }

    features = reference_dir / "features.csv"
    if features.exists():
        ref = _load_matrix(features, "host_id", "ref_")
        summary = evaluate_embedding(emb, ref, ks=(5, 10, 20))
        for (k, metric), row in summary.iterrows():
            for stat in ("mean", "median", "std"):
                result["metrics"][f"{metric}@{k}_{stat}"] = float(row[stat])
    else:
        result["unavailable_metrics"].append("knn_consistency: reference/features.csv missing")

    # Other metric implementations are added by issues #22-#25. Keep them explicit
    # rather than silently inventing scores or per-method reference data.
    expected = {
        "triplet_accuracy": reference_dir / "triplets.csv",
        "pairwise_roc_auc": reference_dir / "pairs.csv",
        "pairwise_pr_auc": reference_dir / "pairs.csv",
        "precision_at_k": reference_dir / "pairs.csv",
        "recall_at_k": reference_dir / "pairs.csv",
        "linear_probe": reference_dir / "labels.csv",
    }
    for metric, artifact in expected.items():
        result["unavailable_metrics"].append(
            f"{metric}: implementation pending" if artifact.exists()
            else f"{metric}: implementation pending; {artifact.name} missing"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a host embedding submission")
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run(args.embeddings, args.reference_dir, args.method)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
