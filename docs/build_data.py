"""Собрать docs/data/synthetic_demo.json для сайта из синтетического прогона demo/synthetic_run.

    python -m experiments.make_demo_bundle      # сначала пересобрать прогон
    python docs/build_data.py                   # затем данные сайта

Числа берутся из сохранённых файлов прогона через тот же слой данных, что у дашборда (app/results.py),
поэтому сайт и дашборд не расходятся.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.results import Bundle  # noqa: E402

KEYS = ["consistency@5_mean", "consistency@10_mean", "consistency@20_mean", "jaccard@10_mean", "ndcg@10_mean",
        "link_roc_auc", "link_learned_roc_auc", "link_new_roc_auc", "link_pr_auc_lift", "partner_new_mrr",
        "pair_roc_auc", "triplet_accuracy", "precision@3", "recall@3"]


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def build(bundle_dir: Path, projection_method: str = "behavioral") -> dict:
    b = Bundle(bundle_dir)
    if not b.synthetic:
        raise SystemExit("сайт показывает только синтетический прогон")
    results = {t: {m: {k: r["metrics"][k] for k in KEYS if k in r["metrics"]}
                   for m, r in b.results(t).items()} for t in b.targets}
    identity = {}
    for m, v in b.identity("test").items():
        identity[m] = {"raw_top1": v.get("none", {}).get("top1"), "aligned_top1": v.get("procrustes", {}).get("top1"),
                       "chance_top1": v.get("procrustes", {}).get("chance_top1")}
    audit = b.audit() or {}
    proj = b.projection(projection_method)
    m = b.manifest
    return {
        "meta": {"synthetic": True, "claim": "Synthetic fixture only; not LANL results",
                 "source_commit": _git_sha(), "source_path": "demo/synthetic_run",
                 "hosts": len(proj), "seed": m["config"]["seed"],
                 "methods": sorted(results["test"]), "missing": m.get("missing_expected_methods", []),
                 "cohort_policy": (m.get("cohort") or {}).get("policy")},
        "results": results,
        "identity": identity,
        "audit": {"prevalence": audit.get("prevalence"), "random_null": audit.get("random_null")},
        "projection": {"method": projection_method,
                       "description": "PCA of train-window behavioral embeddings, colored by agreement with held-out test neighbors",
                       "per_host_mean_agreement": float(proj["agreement"].mean()),
                       "points": [{"id": h, "x": round(float(r.PC1), 5), "y": round(float(r.PC2), 5),
                                   "agreement": float(r.agreement)} for h, r in proj.iterrows()]},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, default=ROOT / "demo" / "synthetic_run")
    ap.add_argument("--output", type=Path, default=ROOT / "docs" / "data" / "synthetic_demo.json")
    a = ap.parse_args()
    data = build(a.bundle)
    a.output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{a.output}: методов {len(data['meta']['methods'])}, хостов {data['meta']['hosts']}")


if __name__ == "__main__":
    main()
