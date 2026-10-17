"""Собрать docs/data/results.json для сайта из прогона (папка с run/, reference/ и processed/).

    python docs/build_data.py --bundle /путь/к/lanl

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
from evaluation import scoreboard as sb  # noqa: E402

KEYS = ["consistency@5_mean", "consistency@10_mean", "consistency@20_mean", "jaccard@10_mean", "ndcg@10_mean",
        "link_roc_auc", "link_learned_roc_auc", "link_new_roc_auc", "link_pr_auc_lift", "partner_new_mrr",
        "pair_roc_auc", "triplet_accuracy", "precision@3", "recall@3"]


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


DETAIL = ["consistency@5_mean", "consistency@10_mean", "consistency@20_mean", "link_roc_auc", "link_learned_roc_auc",
          "link_new_roc_auc", "partner_new_mrr", "pair_roc_auc", "identity_top1"]


def team_table(results_dir: Path = ROOT / "scoreboard" / "results") -> dict:
    """Общая таблица команды: балл 0–100, размерность и главные метрики каждого метода."""
    results = sb.load_results(results_dir)
    cols, values, order = sb.ranking(results)
    score = sb.overall_score(results, cols)
    by = {r["method"]: r for r in results}
    rows = []
    for m in list(order.index) + sorted(sb.SERVICE & set(by)):
        r = by[m]
        rows.append({"method": m, "service": m in sb.SERVICE, "score": None if score[m] != score[m] else round(float(score[m]), 1),
                     "dim": r.get("dim"), "values": {k: r["metrics"].get(k) for k in cols},
                     "tasks": {g: round(v, 1) for g, v in sb.task_scores(r, cols).items()},
                     "detail": {k: r["metrics"].get(k) for k in DETAIL}})
    chance = {}
    for r in results:
        if r["method"] in sb.SERVICE:
            continue
        for s in r.get("summary", []):
            if s.get("chance") is not None:
                chance.setdefault(s["metric"], []).append(s["chance"])
    pers = next((r["metric_details"]["future_link_prediction"].get("persistence_baseline", {}).get("roc_auc")
                 for r in results if r.get("metric_details", {}).get("future_link_prediction")), None)
    groups: dict[str, list[str]] = {}                       # задача -> её колонки
    for r in results:
        for x in r.get("summary", []):
            ms = groups.setdefault(x.get("group") or x["metric"], [])
            if x["metric"] not in ms:
                ms.append(x["metric"])
    return {"columns": cols, "rows": rows, "persistence_auc": pers, "tasks": groups,
            "chance": {k: [min(v), max(v)] for k, v in chance.items()}}


def dataset_facts(bundle_dir: Path) -> dict:
    pm = Bundle(bundle_dir).processed_manifest or {}
    w = pm.get("windows", {})
    coh = pm.get("cohort", {})
    return {"valid_rows": pm.get("counts", {}).get("valid_rows"), "invalid_rows": sum(pm.get("invalid_rows", {}).values()),
            "windows": {k: {"minutes": round((v["end"] - v["start"]) / 60), "flows": v.get("n_flows"),
                            "edges": v.get("graph_edges")} for k, v in w.items()},
            "cohort": coh.get("n_hosts"), "evaluated": (coh.get("evaluable") or {}).get("test") if isinstance(coh.get("evaluable"), dict) else None,
            "rule": coh.get("rule")}


def build(bundle_dir: Path, projection_method: str = "behavioral") -> dict:
    b = Bundle(bundle_dir)
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
        "meta": {"synthetic": b.synthetic, "kind": b.kind, "label": b.label,
                 "source_commit": _git_sha(), "source_path": "results/lanl_day02",
                 "hosts": len(proj), "seed": m["config"]["seed"],
                 "methods": sorted(results["test"]), "missing": m.get("missing_expected_methods", []),
                 "cohort_policy": (m.get("cohort") or {}).get("policy")},
        "results": results,
        "team": team_table(),
        "dataset": dataset_facts(bundle_dir),
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
    ap.add_argument("--bundle", type=Path, required=True, help="папка прогона с run/, reference/, processed/")
    ap.add_argument("--output", type=Path, default=ROOT / "docs" / "data" / "results.json")
    a = ap.parse_args()
    data = build(a.bundle)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{a.output}: методов {len(data['meta']['methods'])}, хостов {data['meta']['hosts']}")


if __name__ == "__main__":
    main()
