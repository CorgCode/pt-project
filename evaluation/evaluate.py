"""Common evaluator: ONE entry point that computes every applicable metric for an embedding.

    python -m evaluation.evaluate --embeddings E.csv --reference-dir reference/test \
        --method NAME --output benchmark/results/NAME.json

Which metric runs is decided by the files present in the reference directory (all built once
by `python -m pipeline.reference`, shared by every method):

  neighbors.csv|features.csv -> kNN Consistency / Jaccard / nDCG   (behavioural proxy)
  edges.csv                  -> future link ROC-AUC / PR-AUC        (predictive, link-level)
  pairs.csv                  -> pair ROC-AUC / PR-AUC, Precision@k / Recall@k  (structural proxy)
  triplets.csv               -> Triplet Accuracy                    (structural proxy)
  labels.csv (+ labels_source in manifest.json) -> linear probe     (external labels only)

Missing artifacts are reported under `unavailable_metrics` / `metric_status`; a PRESENT but
malformed artifact is an error, never silently skipped. Identity retrieval needs two windows
and has its own command (`python -m evaluation.identity_retrieval`).
Definitions and interpretation: evaluation/METRICS.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.frozen_neighbors import evaluate_frozen_neighbors
from evaluation.future_links import load_candidate_edges, score_embeddings
from evaluation.io import load_embeddings
from evaluation.knn_consistency import evaluate_embedding
from evaluation.linear_probe import linear_probe, load_labels
from evaluation.retrieval import load_pairs, pair_auc, precision_recall_at_k
from evaluation.triplet_accuracy import load_triplets, triplet_accuracy
from evaluation.uncertainty import bootstrap_ci

KS = (5, 10, 20)
RETRIEVAL_KS = (1, 3, 5)
REFERENCE_FILES = ("features.csv", "neighbors.csv", "edges.csv", "pairs.csv", "triplets.csv", "labels.csv")
REQUIRED = ("knn_consistency", "future_link_prediction", "pairwise_roc_auc", "pairwise_pr_auc",
            "precision_at_k", "recall_at_k", "triplet_accuracy", "linear_probe")


def _load_matrix(path: Path, id_col: str, prefix: str) -> pd.DataFrame:
    """Compatibility wrapper (also used for reference/features.csv)."""
    return load_embeddings(path, id_col, prefix)


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _ci(result: dict, name: str, ci) -> None:
    if ci is not None:
        result["metrics_ci"][name] = [float(ci[0]), float(ci[1])]


def run(embeddings_path: Path, reference_dir: Path, method: str, split: str | None = None,
        n_boot: int = 0, seed: int = 0) -> dict:
    emb_full = load_embeddings(embeddings_path, "host_id", "emb_")
    emb = emb_full
    reference_dir = Path(reference_dir)
    eval_file = reference_dir / "eval_hosts.csv"
    if eval_file.exists():
        # The reference covers the hosts evaluable in its target window; score exactly those hosts.
        eval_hosts = pd.read_csv(eval_file, dtype={"host_id": str})["host_id"].tolist()
        absent = sorted(set(eval_hosts) - set(emb_full.index))
        if absent:
            raise ValueError(f"embedding lacks {len(absent)} evaluable hosts (e.g. {absent[:5]}); "
                             "every method must embed the whole cohort")
        emb = emb_full.loc[eval_hosts]
    manifest_file = reference_dir / "manifest.json"
    protocol = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists() else {}
    split = split or protocol.get("split", "test")
    result: dict = {
        "method": method, "submission": str(embeddings_path), "n_hosts": int(len(emb)),
        "n_embedded_hosts": int(len(emb_full)),
        "split": split, "n_bootstrap": n_boot, "seed": seed,
        "metrics": {}, "metrics_ci": {}, "metric_details": {},
        "metric_status": {}, "unavailable_metrics": [],
    }
    neighbors, features = reference_dir / "neighbors.csv", reference_dir / "features.csv"
    result["reference_info"] = {
        "features_sha256": _sha(features), "neighbors_sha256": _sha(neighbors),
        "files_sha256": {f: _sha(reference_dir / f) for f in REFERENCE_FILES if (reference_dir / f).exists()},
        "kind": protocol.get("kind", "unspecified_reference"),
        "ground_truth": protocol.get("ground_truth"),
        "provenance_verified": protocol.get("provenance_verified", False),
        "train_end_utc": protocol.get("train_end_utc"), "test_start_utc": protocol.get("test_start_utc"),
        "data_kind": protocol.get("data_kind", "unspecified"),
        "synthetic": protocol.get("synthetic"),
        "real_lanl_declared": protocol.get("real_lanl_declared"),
        "provenance_verified_by_pipeline": protocol.get("provenance_verified", False),
        "cohort_policy": protocol.get("cohort_policy"),
        "n_cohort": protocol.get("n_cohort"), "n_evaluable": protocol.get("n_evaluable"),
        "evaluable_coverage": protocol.get("coverage"),
        "target_window": protocol.get("target_window"), "source_window": protocol.get("source_window"),
        "cohort_sha256": protocol.get("cohort_sha256"),
        "labels_source": protocol.get("labels_source"),
    }

    def unavailable(metric: str, reason: str) -> None:
        result["metric_status"][metric] = f"unavailable: {reason}"
        result["unavailable_metrics"].append(f"{metric}: {reason}")

    # ---- behavioural neighbourhood agreement (proxy) -------------------------------------
    per_host = None
    if neighbors.exists():
        summary, per_host = evaluate_frozen_neighbors(emb, neighbors, KS, split=split, return_per_host=True)
    elif features.exists():
        summary, per_host = evaluate_embedding(emb, load_embeddings(features, "host_id", "ref_"),
                                               ks=KS, return_per_host=True)
    else:
        summary = None
    if summary is not None:
        for (k, metric), row in summary.iterrows():
            for stat in ("mean", "median", "std"):
                result["metrics"][f"{metric}@{k}_{stat}"] = float(row[stat])
        for k, ph in per_host.items():
            for metric in ph.columns:
                v = ph[metric].to_numpy()
                _ci(result, f"{metric}@{k}_mean",
                    bootstrap_ci(lambda i, v=v: float(v[i].mean()), len(v), n_boot=n_boot, seed=seed))
        result["metric_status"]["knn_consistency"] = "ok"
    else:
        unavailable("knn_consistency", "reference/neighbors.csv and reference/features.csv missing")

    # ---- future link prediction ----------------------------------------------------------
    edges_file = reference_dir / "edges.csv"
    if edges_file.exists():
        r = score_embeddings(emb, load_candidate_edges(edges_file), split=split, n_boot=n_boot, seed=seed)
        m = result["metrics"]
        m["link_roc_auc"], m["link_pr_auc"] = r["roc_auc"], r["pr_auc"]
        m["link_pr_auc_lift"] = r["pr_auc_lift"]
        if r.get("pr_auc_at_natural_prevalence") is not None:
            m["link_pr_auc_at_natural_prevalence"] = r["pr_auc_at_natural_prevalence"]
        if "new_links" in r:
            m["link_new_roc_auc"], m["link_new_pr_auc"] = r["new_links"]["roc_auc"], r["new_links"]["pr_auc"]
        _ci(result, "link_roc_auc", r["roc_auc_ci"])
        _ci(result, "link_pr_auc", r["pr_auc_ci"])
        result["metric_details"]["future_link_prediction"] = {
            k: r[k] for k in ("n_examples", "n_positive", "positive_rate", "natural_prevalence",
                              "share_positives_seen_in_source", "source_window", "target_window", "split",
                              "interpretation", "new_links") if k in r}
        result["metric_status"]["future_link_prediction"] = "ok"
    else:
        unavailable("future_link_prediction", "edges.csv missing")

    # ---- pair judgments: pairwise AUC and retrieval --------------------------------------
    pairs_file = reference_dir / "pairs.csv"
    if pairs_file.exists():
        pairs = load_pairs(pairs_file)
        j = protocol.get("judgments") or {}
        n_pos, n_neg = j.get("n_positive_pairs", 0), j.get("n_negative_pairs", 0)
        judged_prev = n_pos / (n_pos + n_neg) if n_pos and n_neg else None
        a = pair_auc(emb, pairs, split, judged_prevalence=judged_prev, n_boot=n_boot, seed=seed)
        m = result["metrics"]
        m["pair_roc_auc"], m["pair_pr_auc"], m["pair_pr_auc_lift"] = a["roc_auc"], a["pr_auc"], a["pr_auc_lift"]
        if "pr_auc_at_judged_prevalence" in a:
            m["pair_pr_auc_at_judged_prevalence"] = a["pr_auc_at_judged_prevalence"]
        _ci(result, "pair_roc_auc", a["roc_auc_ci"])
        _ci(result, "pair_pr_auc", a["pr_auc_ci"])
        n_possible = j.get("n_possible_pairs")
        result["metric_details"]["pairwise_auc"] = {
            **{k: a[k] for k in ("n_pairs", "n_positive", "positive_rate", "split")},
            "judged_population_prevalence": judged_prev,
            "population_positive_rate": (n_pos / n_possible) if n_possible else None,
            "unjudged_share": (j.get("n_unjudged_pairs", 0) / n_possible) if n_possible else None,
            "interpretation": (
                "Relevance is judged only for pairs with peer-set Jaccard >= tau (positive) or == 0 (negative). "
                "judged_population_prevalence is the positive rate AMONG JUDGED pairs; the unjudged pairs "
                "(0 < J < tau) are excluded, so it is not the positive rate of all host pairs "
                "(population_positive_rate) and pair_pr_auc_at_judged_prevalence is a conditional AP, "
                "not a population AP.")}
        pr = precision_recall_at_k(emb, pairs, RETRIEVAL_KS, split, n_boot=n_boot, seed=seed)
        for k, e in pr["by_k"].items():
            if e["eligible_anchors"]:
                m[f"precision@{k}"], m[f"recall@{k}"] = e["precision"], e["recall"]
                m[f"precision_lift@{k}"] = e["precision_lift"]
                _ci(result, f"precision@{k}", e["precision_ci"])
                _ci(result, f"recall@{k}", e["recall_ci"])
        result["metric_details"]["retrieval"] = pr
        for name in ("pairwise_roc_auc", "pairwise_pr_auc", "precision_at_k", "recall_at_k"):
            result["metric_status"][name] = "ok"
    else:
        for name in ("pairwise_roc_auc", "pairwise_pr_auc", "precision_at_k", "recall_at_k"):
            unavailable(name, "pairs.csv missing")

    # ---- triplets ------------------------------------------------------------------------
    triplets_file = reference_dir / "triplets.csv"
    if triplets_file.exists():
        t = triplet_accuracy(emb, load_triplets(triplets_file), split, n_boot=n_boot, seed=seed)
        result["metrics"]["triplet_accuracy"] = t["accuracy"]
        _ci(result, "triplet_accuracy", t["accuracy_ci"])
        result["metric_details"]["triplet_accuracy"] = {k: t[k] for k in
                                                        ("n_triplets", "n_ties", "chance", "split")}
        result["metric_status"]["triplet_accuracy"] = "ok"
    else:
        unavailable("triplet_accuracy", "triplets.csv missing")

    # ---- linear probe (external labels with declared provenance only) -----------------------
    labels_file = reference_dir / "labels.csv"
    source = protocol.get("labels_source")
    if not labels_file.exists():
        unavailable("linear_probe", "labels.csv missing (no external host labels exist; none are fabricated)")
    elif not source:
        unavailable("linear_probe", "labels.csv has no declared external provenance (manifest labels_source)")
    else:
        pr = linear_probe(emb, load_labels(labels_file), labels_source=source,
                          eval_split="test" if split == "test" else split, n_boot=n_boot, seed=seed)
        for k in ("accuracy", "macro_f1", "balanced_accuracy"):
            result["metrics"][f"probe_{k}"] = pr[k]
        result["metrics"]["probe_majority_baseline_accuracy"] = pr["majority_baseline_accuracy"]
        _ci(result, "probe_accuracy", pr["accuracy_ci"])
        result["metric_details"]["linear_probe"] = {k: v for k, v in pr.items() if k != "accuracy_ci"}
        result["metric_status"]["linear_probe"] = "ok"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a host embedding submission")
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--split", default=None, help="default: the split declared by the reference manifest")
    parser.add_argument("--bootstrap", type=int, default=200, help="bootstrap replicates for 95%% CIs (0 = off)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run(args.embeddings, args.reference_dir, args.method, args.split, args.bootstrap, args.seed)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
