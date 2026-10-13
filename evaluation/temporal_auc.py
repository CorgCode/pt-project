"""Evaluate a frozen temporal identity-pair reference with ROC-AUC and PR-AUC."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics import average_precision_score, roc_auc_score
from evaluation.io import load_embeddings, ReferenceError
from evaluation.uncertainty import auc_with_ci, host_index

REQUIRED = {"window_a", "host_a", "window_b", "host_b", "label", "split"}
SPLITS = {"train", "validation", "test"}


def load_temporal_pairs(path: Path) -> pd.DataFrame:
    p = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = REQUIRED - set(p.columns)
    if missing: raise ReferenceError(f"{path}: missing columns {sorted(missing)}")
    if p.empty: raise ReferenceError(f"{path}: no pairs")
    for c in REQUIRED:
        if p[c].str.strip().eq("").any(): raise ReferenceError(f"{path}: blank values in {c!r}")
    if not p.label.isin(["0", "1"]).all(): raise ReferenceError(f"{path}: label must be 0 or 1")
    if not p.split.isin(SPLITS).all(): raise ReferenceError(f"{path}: split must be train, validation, or test")
    if p.window_a.nunique() != 1 or p.window_b.nunique() != 1:
        raise ReferenceError(f"{path}: each file must represent one pair of windows")
    same = p.host_a.eq(p.host_b)
    if not p.label.astype(int).eq(same.astype(int)).all():
        raise ReferenceError(f"{path}: label must be 1 exactly when host_a == host_b")
    # Prevent duplicate rows from overweighting particular examples.
    if p.duplicated(["window_a", "host_a", "window_b", "host_b", "split"]).any():
        raise ReferenceError(f"{path}: duplicate pair in a split")
    return p


def fit_procrustes(a: pd.DataFrame, b: pd.DataFrame, train: pd.DataFrame):
    positives = train[(train.label == "1") & (train.host_a == train.host_b)]
    hosts = sorted(set(positives.host_a) & set(a.index) & set(b.index))
    if len(hosts) < 2: raise ValueError("Procrustes needs at least two train positive pairs")
    A, B = a.loc[hosts].to_numpy(float), b.loc[hosts].to_numpy(float)
    if A.shape[1] != B.shape[1]: raise ValueError("Procrustes requires equal embedding dimensions")
    mu_a, mu_b = A.mean(axis=0), B.mean(axis=0)
    R, _ = orthogonal_procrustes(B - mu_b, A - mu_a)
    return mu_a, mu_b, R, hosts


def cosine_scores(a: pd.DataFrame, b: pd.DataFrame, p: pd.DataFrame, transform=None) -> np.ndarray:
    missing = (set(p.host_a) - set(a.index)) | (set(p.host_b) - set(b.index))
    if missing: raise ReferenceError(f"embedding missing hosts used in pairs: {sorted(missing)[:5]}")
    A, B = a.loc[p.host_a].to_numpy(float), b.loc[p.host_b].to_numpy(float)
    if transform is not None:
        mu_a, mu_b, R, _ = transform
        A, B = A - mu_a, (B - mu_b) @ R
    na, nb = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
    if (na == 0).any() or (nb == 0).any(): raise ValueError("zero-norm vector: cosine is undefined")
    return np.einsum("ij,ij->i", A, B) / (na * nb)


def nearest_candidate_accuracy(a: pd.DataFrame, b: pd.DataFrame, hosts: list[str], transform=None) -> tuple[float, int]:
    """Fraction of queries whose same-ID vector is tied for nearest in the candidate window.

    Queries and candidates are restricted to the current validation/test cohort. Ties at
    the maximum cosine similarity count as a hit; this avoids arbitrary host-ID tie breaks.
    """
    hosts = sorted(set(map(str, hosts)))
    if len(hosts) < 2:
        raise ValueError("nearest-candidate evaluation needs at least two hosts")
    missing = (set(hosts) - set(a.index)) | (set(hosts) - set(b.index))
    if missing:
        raise ReferenceError(f"embedding missing candidate hosts: {sorted(missing)[:5]}")
    A, B = a.loc[hosts].to_numpy(float), b.loc[hosts].to_numpy(float)
    if transform is not None:
        mu_a, mu_b, R, _ = transform
        A, B = A - mu_a, (B - mu_b) @ R
    na, nb = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
    if (na == 0).any() or (nb == 0).any():
        raise ValueError("zero-norm vector: cosine is undefined")
    similarities = (A @ B.T) / (na[:, None] * nb[None, :])
    # The diagonal is the same-host candidate. Count ties within numerical tolerance.
    hits = np.diag(similarities) >= similarities.max(axis=1) - 1e-12
    return float(hits.mean()), len(hosts)

def evaluate_method(a: pd.DataFrame, b: pd.DataFrame, pairs: pd.DataFrame, *, method: str,
                    shared_coordinates: bool = False, n_boot: int = 200, seed: int = 41) -> list[dict]:
    transform = fit_procrustes(a, b, pairs[pairs.split == "train"])
    align_hosts = transform[3]
    rows = []
    for split in ("validation", "test"):
        p = pairs[pairs.split == split].copy()
        if p.empty: raise ValueError(f"no pairs for split={split}")
        y = p.label.astype(int).to_numpy()
        if len(np.unique(y)) != 2: raise ValueError(f"split={split} must contain both labels")
        modes = {}
        if shared_coordinates: modes["cosine_raw"] = cosine_scores(a, b, p)
        modes["cosine_procrustes"] = cosine_scores(a, b, p, transform)
        # Host-cluster bootstrap because pair rows share endpoints.
        host_clusters = host_index(p.host_a, p.host_b)
        win = f"{p.window_a.iloc[0]}_to_{p.window_b.iloc[0]}"
        for score_name, scores in modes.items():
            metrics = auc_with_ci(y, scores, n_boot=n_boot, seed=seed, hosts=host_clusters)
            details = {
                "score": score_name, "roc_auc_ci": metrics["roc_auc_ci"],
                "pr_auc_ci": metrics["pr_auc_ci"], "ci_method": metrics["ci_method"],
                "alignment_fit_hosts": len(align_hosts),
                "alignment_hosts_sha256": hashlib.sha256("\n".join(align_hosts).encode()).hexdigest(),
                "shared_coordinate_system_declared": shared_coordinates,
            }
            base = {"method": method, "score_name": score_name, "split": split, "window_id": win,
                    "n_examples": len(p), "n_positive": int(y.sum()), "positive_rate": float(y.mean()),
                    "coverage": 1.0, "details": json.dumps(details, sort_keys=True)}
            for metric in ("roc_auc", "pr_auc"):
                rows.append({**base, "metric": metric, "score_value": float(metrics[metric])})

        # Retrieval uses the full candidate cohort, not the sampled pair rows.
        cohort = sorted(set(p.loc[p.label == "1", "host_a"]) & set(p.loc[p.label == "1", "host_b"]))
        retrieval_modes = {}
        if shared_coordinates:
            retrieval_modes["cosine_raw"] = None
        retrieval_modes["cosine_procrustes"] = transform
        for score_name, alignment in retrieval_modes.items():
            accuracy, n_hosts = nearest_candidate_accuracy(a, b, cohort, alignment)
            details = {
                "score": score_name,
                "candidate_pool_size": n_hosts,
                "tie_policy": "ties at maximum cosine count as hits (tolerance 1e-12)",
                "alignment_fit_hosts": len(align_hosts),
                "alignment_hosts_sha256": hashlib.sha256("\n".join(align_hosts).encode()).hexdigest(),
                "shared_coordinate_system_declared": shared_coordinates,
            }
            rows.append({
                "method": method, "metric": "nearest_candidate_accuracy", "score_name": score_name,
                "score_value": accuracy, "split": split, "window_id": win,
                "n_examples": n_hosts, "n_positive": n_hosts, "positive_rate": 1.0 / n_hosts,
                "coverage": 1.0, "details": json.dumps(details, sort_keys=True),
            })
    return rows


def method_path(value: str) -> tuple[str, Path]:
    if "=" not in value: raise argparse.ArgumentTypeError("expected METHOD=PATH")
    method, path = value.split("=", 1)
    if not method.strip() or not path.strip(): raise argparse.ArgumentTypeError("expected METHOD=PATH")
    return method.strip(), Path(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--embeddings-a", action="append", type=method_path, required=True, metavar="METHOD=PATH")
    ap.add_argument("--embeddings-b", action="append", type=method_path, required=True, metavar="METHOD=PATH")
    ap.add_argument("--shared-coordinates", action="append", default=[], metavar="METHOD")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--output", type=Path, default=Path("runs/temporal/results.csv"))
    args = ap.parse_args()
    ea, eb = dict(args.embeddings_a), dict(args.embeddings_b)
    if len(ea) != len(args.embeddings_a) or len(eb) != len(args.embeddings_b): ap.error("duplicate method name")
    if not ea or set(ea) != set(eb): ap.error("methods must match and be non-empty across both windows")
    pairs = load_temporal_pairs(args.pairs)
    rows = []
    for method in sorted(ea):
        a, b = load_embeddings(ea[method]), load_embeddings(eb[method])
        rows.extend(evaluate_method(a, b, pairs, method=method,
                                    shared_coordinates=method in args.shared_coordinates,
                                    n_boot=args.bootstrap, seed=args.seed))
    out = pd.DataFrame(rows, columns=["method", "metric", "score_name", "score_value", "split", "window_id",
                                     "n_examples", "n_positive", "positive_rate", "coverage", "details"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} metric rows for {len(ea)} methods to {args.output}")

if __name__ == "__main__": main()
