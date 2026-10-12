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
from evaluation.io import load_embeddings as _load_matrix
from evaluation.uncertainty import auc_with_ci, host_index, reweighted_average_precision


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
    *, split: str = "test", n_boot: int = 0, seed: int = 0,
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
    auc = auc_with_ci(labels, cosine, n_boot=n_boot, seed=seed, hosts=host_index(edges.host_a, edges.host_b))
    n_hosts = len(known)
    natural = float(labels.sum() / (n_hosts * (n_hosts - 1) / 2))
    extra: dict = {
        "roc_auc_ci": auc["roc_auc_ci"], "pr_auc_ci": auc["pr_auc_ci"], "ci_method": auc["ci_method"],
        "pr_auc_lift": auc["pr_auc"] / float(labels.mean()),
        # PR-AUC as if the pool had the natural positive rate (observed target edges over
        # all cohort pairs); removes the class-prior effect of 1:1 sampling only.
        "natural_prevalence": natural,
        "pr_auc_at_natural_prevalence": (reweighted_average_precision(labels, cosine, natural)
                                         if 0 < natural < 1 else None),
    }
    if "seen_in_source" in edges.columns:
        seen = edges["seen_in_source"].astype(str).str.lower().eq("true").to_numpy()
        extra["share_positives_seen_in_source"] = float(seen[labels == 1].mean())
        new = ~seen
        if new.any() and len(np.unique(labels[new])) == 2:
            extra["new_links"] = {
                "roc_auc": float(roc_auc_score(labels[new], cosine[new])),
                "pr_auc": float(average_precision_score(labels[new], cosine[new])),
                "n_examples": int(new.sum()), "n_positive": int(labels[new].sum()),
                "positive_rate": float(labels[new].mean()),
                "note": "candidates not observed in the source window (new links vs unseen non-links)",
            }
    return {
        **extra,
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


def _window_bounds(window: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    a, b = window.split("/")
    return pd.Timestamp(a.rstrip("Z")), pd.Timestamp(b.rstrip("Z"))


def score_learned(embedding: pd.DataFrame, fit_edges: pd.DataFrame, eval_edges: pd.DataFrame, *,
                  fit_split: str = "validation", eval_split: str = "test", seed: int = 0,
                  n_boot: int = 0) -> dict:
    """Edge classifier on frozen embeddings: LogisticRegression on [|e_a - e_b|, e_a * e_b].

    Fit on the candidates of an EARLIER target window (validation), scored on the later one
    (test); the embedding is the same source-window embedding in both cases. This is the
    standard edge-classification protocol and does not use cosine, which is the wrong score for
    heterophilous traffic (clients link to servers, not to look-alike clients). The fit window
    must end before the evaluation window starts and use the same source window; otherwise it is
    refused (no tuning on, or peeking at, the evaluation window). C=1 fixed, never tuned on test.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    fit = fit_edges[fit_edges["split"] == fit_split]
    ev = eval_edges[eval_edges["split"] == eval_split]
    if fit.empty or ev.empty:
        raise ValueError(f"need candidates for fit split {fit_split!r} and eval split {eval_split!r}")
    fw, ew = fit.iloc[0], ev.iloc[0]
    if fw["source_window"] != ew["source_window"]:
        raise ValueError("fit and evaluation candidates use different source windows")
    if _window_bounds(fw["target_window"])[1] > _window_bounds(ew["target_window"])[0]:
        raise ValueError("fit window must end before the evaluation window starts (temporal leakage)")
    used = set(fit.host_a) | set(fit.host_b) | set(ev.host_a) | set(ev.host_b)
    if not used <= set(embedding.index):
        raise ValueError(f"embedding lacks candidate hosts: {sorted(used - set(embedding.index))[:5]}")
    scaler = StandardScaler().fit(embedding.to_numpy(float))
    z = pd.DataFrame(scaler.transform(embedding.to_numpy(float)), index=embedding.index)

    def feats(df):
        a, b = z.loc[df.host_a].to_numpy(), z.loc[df.host_b].to_numpy()
        return np.hstack([np.abs(a - b), a * b])

    yf, ye = fit["label"].astype(int).to_numpy(), ev["label"].astype(int).to_numpy()
    if len(np.unique(yf)) < 2 or len(np.unique(ye)) < 2:
        raise ValueError("both classes are required in the fit and evaluation candidates")
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(feats(fit), yf)
    score = clf.predict_proba(feats(ev))[:, 1]
    out = auc_with_ci(ye, score, n_boot=n_boot, seed=seed, hosts=host_index(ev.host_a, ev.host_b))
    out.update(n_fit=int(len(fit)), n_eval=int(len(ev)), fit_split=fit_split, eval_split=eval_split,
               fit_target_window=fw["target_window"], eval_target_window=ew["target_window"],
               pr_auc_lift=out["pr_auc"] / float(ye.mean()))
    return out


def persistence_baseline(edges: pd.DataFrame, split: str = "test") -> dict | None:
    """Embedding-free reference point: score = 1 if the pair already communicated in the source window."""
    e = edges[edges["split"] == split]
    if "seen_in_source" not in e.columns:
        return None
    y = e["label"].astype(int).to_numpy()
    seen = e["seen_in_source"].astype(str).str.lower().eq("true").to_numpy().astype(float)
    if len(np.unique(y)) < 2:
        return None
    return {"roc_auc": float(roc_auc_score(y, seen)), "pr_auc": float(average_precision_score(y, seen)),
            "note": "trivial predictor: pair seen in the source window; embeddings should be read against it"}


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
