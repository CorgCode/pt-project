"""Compare archived source samples with refits without reading any test window."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from evaluation.io import load_embeddings
from pipeline.prepare import _sha256
from .download import atomic_json
from .models import ROOT, WORK


def source_geometry(original, candidate, cuts):
    if original.index.has_duplicates or candidate.index.has_duplicates:
        raise ValueError("duplicate source host")
    if original.shape[1] != candidate.shape[1] or cuts[-1] != original.shape[1]:
        raise ValueError("source dimensions differ")
    if not original.index.isin(candidate.index).all():
        raise ValueError("source host is missing")
    a = original.to_numpy(dtype=float)
    b = candidate.loc[original.index].to_numpy(dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite source vectors")
    errors, start = [], 0
    for stop in cuts:
        ga, gb = a[:, start:stop] @ a[:, start:stop].T, b[:, start:stop] @ b[:, start:stop].T
        errors.append({"columns": [start, stop], "relative_gram_error":
                       float(np.linalg.norm(ga - gb) / max(np.linalg.norm(ga), 1e-30))})
        start = stop
    return errors


def audit(samples, work=WORK):
    samples, work = Path(samples), Path(work)
    results = []
    for name, cuts in (("ref_x", [57, 185, 249, 505]), ("ref_x_f", [57, 185, 249, 505]),
                       ("gnn_tfidf_new", [66, 322, 450, 642])):
        source = samples / f"{name}.csv"
        meta = json.loads((samples / f"{name}.json").read_text())
        candidate = work / "models" / name / "embeddings.csv"
        training = json.loads(candidate.with_name("training.json").read_text())
        if _sha256(candidate) != training["source_sha256"]:
            raise ValueError("source changed during audit")
        original, refit = load_embeddings(source), load_embeddings(candidate)
        commit = subprocess.check_output(["git", "rev-parse", meta["commit"]], cwd=ROOT, text=True).strip()
        blob = subprocess.check_output(["git", "ls-tree", commit, meta["path"]], cwd=ROOT, text=True).split()[2]
        results.append({"method": name, "original_commit": commit, "original_blob": blob,
                        "original_path": meta["path"], "sample_sha256": _sha256(source),
                        "sample_hosts": len(original), "selection": meta["sample"],
                        "candidate_source_sha256": _sha256(candidate), "blocks": source_geometry(original, refit, cuts)})
    report = {"scope": "source-vector samples only; no test vectors or test-based parameter selection",
              "interpretation": "Gram matrices are invariant to SVD basis rotations. Agreement on this sample supports source-block parity; "
                                "it does not prove complete inference or historical-checkpoint equivalence. Learned GNN refits may drift.",
              "source_checks": results}
    atomic_json(work / "quality_audit.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=ROOT / "data/late_benchmark/audit/source_samples")
    parser.add_argument("--work", type=Path, default=WORK)
    args = parser.parse_args()
    print(json.dumps(audit(args.samples, args.work), indent=2))
