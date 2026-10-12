"""Build FROZEN evaluation references from a processed dataset (once, for all methods).

    python -m pipeline.reference --processed-dir data/processed --output-dir reference

For each target window (validation, test) a complete reference directory is written,
all built from that later window and the fixed cohort - never from any embedding:

    reference/<target>/features.csv    behavioural profile of the target window
    reference/<target>/neighbors.csv   frozen top-k behavioural neighbours      (kNN proxy)
    reference/<target>/edges.csv       future-link candidates, source=train     (link prediction)
    reference/<target>/pairs.csv       shared-peer pair judgments               (structural proxy)
    reference/<target>/triplets.csv    shared-peer triplets                     (structural proxy)
    reference/<target>/labels.csv      ONLY if --labels is supplied (external, with provenance)
    reference/<target>/eval_hosts.csv  hosts evaluable in the target window (the evaluator subsets to them)
    reference/<target>/manifest.json   provenance, cohort policy/coverage, windows, rules, file hashes

Use reference/validation to tune; report final numbers on reference/test only.
No role labels are fabricated: labels.csv exists only when supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from evaluation.future_links import build_edges
from evaluation.structural_pairs import build_judgments
from evaluation.temporal_reference import generate_reference, load_flows
from pipeline.data import ProcessedData

TARGETS = ("validation", "test")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_reference(processed_dir: Path, output_dir: Path, *, seed: int = 0,
                    ks: tuple[int, ...] = (5, 10, 20), negatives_per_positive: int = 1,
                    tau_pos: float | None = None, pairs_per_anchor: int = 5,
                    labels: Path | None = None, labels_source: str | None = None) -> dict:
    data = ProcessedData(processed_dir)
    if labels is not None and not labels_source:
        raise ValueError("--labels requires --labels-source describing where the labels come from")
    train = load_flows(processed_dir / "flows" / "train.csv")
    summary = {}
    for target in (t for t in TARGETS if t in data.windows):
        # hosts evaluable in THIS target window (== the cohort under the all_windows policy)
        hosts = data.evaluable(target)
        out = Path(output_dir) / target
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"host_id": hosts}).to_csv(out / "eval_hosts.csv", index=False)
        flows = load_flows(processed_dir / "flows" / f"{target}.csv")
        manifest = generate_reference(train, flows, hosts, out, ks, split=target)

        edges = build_edges(train, flows, hosts, seed=seed, split=target,
                            negatives_per_positive=negatives_per_positive)
        edges.to_csv(out / "edges.csv", index=False)

        pairs, triplets, judge = build_judgments(flows, hosts, tau_pos=tau_pos, seed=seed,
                                                 pairs_per_anchor=pairs_per_anchor)
        pairs.to_csv(out / "pairs.csv", index=False)
        triplets.to_csv(out / "triplets.csv", index=False)

        label_info = None
        if labels is not None:
            lab = pd.read_csv(labels, dtype={"host_id": "string", "target": "string", "split": "string"})
            if not {"host_id", "target", "split"} <= set(lab.columns):
                raise ValueError("labels file needs columns host_id,target,split")
            unknown = sorted(set(lab.host_id) - set(data.cohort))
            if unknown:
                raise ValueError(f"labels contain hosts outside the cohort: {unknown[:5]}")
            lab.to_csv(out / "labels.csv", index=False)
            label_info = {"source": labels_source, "n_labelled": int(len(lab)),
                          "coverage": len(lab) / len(hosts)}

        files = {f.name: _sha(f) for f in sorted(out.glob("*.csv"))}   # includes eval_hosts.csv
        manifest.update({
            "data_kind": data.manifest["data_kind"],
            "synthetic": data.synthetic,
            "real_lanl_declared": bool(data.manifest.get("real_lanl_declared")),
            "provenance_verified": False,
            "provenance": data.manifest.get("provenance"),
            "cohort_policy": data.cohort_policy,
            "n_cohort": len(data.cohort), "n_evaluable": len(hosts),
            "coverage": len(hosts) / len(data.cohort),
            "cohort_selection": data.manifest["cohort"].get("selection"),
            "pipeline_manifest_sha256": _sha(processed_dir / "manifest.json"),
            "cohort_sha256": data.manifest["cohort"]["sha256"],
            "source_window": "train", "target_window": target,
            "windows": data.manifest["windows"],
            "seed": seed,
            "files_sha256": files,
            "reference_views": {
                "neighbors.csv": "behavioural-profile proxy (target window)",
                "edges.csv": "future communication (source=train, target window); "
                             f"negatives_per_positive={negatives_per_positive}",
                "pairs.csv / triplets.csv": "structural proxy: shared communication peers",
                "labels.csv": "external labels, only if supplied",
            },
            "judgments": judge,
            "labels": label_info,
            "labels_source": labels_source if labels is not None else None,
        })
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False,
                                                      sort_keys=True) + "\n", encoding="utf-8")
        summary[target] = {"n_hosts": len(hosts), "coverage": len(hosts) / len(data.cohort), "edges": int(len(edges)),
                           "positive_rate": float(edges["label"].mean()),
                           "pairs": judge["n_pairs"], "triplets": judge["n_triplets"],
                           "labels": label_info}
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20])
    ap.add_argument("--negatives-per-positive", type=int, default=1)
    ap.add_argument("--tau-pos", type=float, default=None,
                    help="fixed Jaccard threshold; default: max(0.1, 99th percentile)")
    ap.add_argument("--pairs-per-anchor", type=int, default=5)
    ap.add_argument("--labels", type=Path, help="external host_id,target,split file (optional)")
    ap.add_argument("--labels-source", help="where the labels come from (required with --labels)")
    a = ap.parse_args()
    try:
        out = build_reference(a.processed_dir, a.output_dir, seed=a.seed, ks=tuple(a.ks),
                              negatives_per_positive=a.negatives_per_positive, tau_pos=a.tau_pos,
                              pairs_per_anchor=a.pairs_per_anchor, labels=a.labels,
                              labels_source=a.labels_source)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
