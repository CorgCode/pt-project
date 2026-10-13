"""Run all methods on the SAME cohort, windows, references and candidates.

    python -m experiments.run_experiment --processed-dir data/processed \
        --reference-dir data/reference --out-dir runs/lanl_smoke [--submissions-dir submissions]

Methods: built-in train-window-only baselines (random, behavioral, behavioral_pca,
behavioral_svd, graph_spectral), the reference ceiling (leaks by design), and any
`<submissions-dir>/<method>/embeddings.csv` (source-window embedding; optional
`<method>/<window>.csv` files enable identity retrieval). Expected-but-absent methods are
listed in the report, never simulated.

Outputs: results/<target>/<method>.json, identity/<target>/<method>.json,
leaderboard_<target>.csv, audit.json, report.md, figures/, manifest.json.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from benchmark import methods as graph_methods
from benchmark.baselines.temporal import Baseline, default_baselines
from evaluation.evaluate import run
from evaluation.identity_retrieval import identity_retrieval
from evaluation.io import load_embeddings
from experiments import audit as A
from experiments import report as R
from pipeline.data import ProcessedData
from pipeline.provenance import BADGE

TARGETS = ("validation", "test")
SCOPE = {
    "synthetic_fixture": ("The numbers come from generated data in the LANL netflow layout. They demonstrate the pipeline "
                          "and the metrics only and say nothing about real network behaviour."),
    "unverified": ("The origin of the input files is not established, so nothing here is presented as a LANL result. "
                   "Declare the source (`--data-kind lanl --source-note ...`) to label a run as LANL."),
    "lanl_declared": ("Findings apply to the declared LANL netflow files under the protocol in this report "
                      "(windows, cohort policy, references). They are not claims about other data, other windows, or host roles."),
}
TAGS = {"synthetic_fixture": "SYNTHETIC FIXTURE", "unverified": "UNVERIFIED INPUT", "lanl_declared": "LANL (declared)"}
EXPECTED_EXTERNAL = ["node2vec", "deepwalk", "graphsage", "struc2vec", "rolx"]   # issue #20 / research plan


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def _read_training_windows(submission_dir: Path) -> list[str] | None:
    """Windows an external method declares to have been trained on (provenance.json); None if undeclared."""
    f = Path(submission_dir) / "provenance.json"
    if not f.exists():
        return None
    try:
        w = json.loads(f.read_text(encoding="utf-8")).get("trained_on_windows")
    except json.JSONDecodeError as exc:
        raise ValueError(f"{f}: not valid JSON") from exc
    if not isinstance(w, list) or not all(isinstance(x, str) for x in w):
        raise ValueError(f'{f}: needs {{"trained_on_windows": ["train", ...]}}')
    return w


def _write(path: Path, df: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return path


def run_experiment(processed_dir: Path, reference_dir: Path, out_dir: Path, *, dim: int = 16, seed: int = 0,
                   n_boot: int = 200, submissions_dir: Path | None = None, do_audit: bool = True,
                   audit_configs: int = 12, null_draws: int = 30, methods: list[str] | None = None,
                   method_config: dict | None = None, fast_methods: bool = False,
                   port_free_baselines: bool = False) -> dict:
    processed_dir, reference_dir, out_dir = Path(processed_dir), Path(reference_dir), Path(out_dir)
    data = ProcessedData(processed_dir)
    targets = [t for t in TARGETS if (reference_dir / t).exists() and t in data.windows]
    if "test" not in targets:
        raise ValueError(f"{reference_dir}/test not found; run `python -m pipeline.reference` first")
    ref_manifest = json.loads((reference_dir / "test" / "manifest.json").read_text())
    if ref_manifest.get("cohort_sha256") != data.manifest["cohort"]["sha256"]:
        raise ValueError("reference was built from a different cohort than the processed dataset")
    for t in targets:                       # the saved reference must cover exactly the evaluable hosts
        saved = pd.read_csv(reference_dir / t / "eval_hosts.csv", dtype=str)["host_id"].tolist()
        if saved != data.evaluable(t):
            raise ValueError(f"reference/{t}/eval_hosts.csv does not match the evaluable hosts of the dataset")
    hosts_by_target = {t: data.evaluable(t) for t in targets}

    # ---- collect embeddings: method -> {window: DataFrame}; 'train' = scored source embedding
    emb: dict[str, dict[str, pd.DataFrame]] = {}
    meta: dict[str, dict] = {}
    graph_methods_run = graph_methods.create(methods or [], dim, seed, method_config, fast_methods)
    for b in [*default_baselines(dim, seed, port_free_baselines), *graph_methods_run]:
        b.fit(data)
        emb[b.name] = {"train": b.embed(data, "train"),
                       **{t: b.embed(data, t, hosts_by_target[t]) for t in targets}}
        meta[b.name] = b.info()
    external, external_provenance = [], {}
    if submissions_dir and Path(submissions_dir).exists():
        for d in sorted(p for p in Path(submissions_dir).iterdir() if (p / "embeddings.csv").exists()):
            declared = _read_training_windows(d)
            if declared is not None and set(declared) - {"train"}:
                raise ValueError(f"submission {d.name!r} declares training on {sorted(set(declared) - {'train'})}; "
                                 "methods may only be fitted on the train window (temporal leakage)")
            external_provenance[d.name] = ({"trained_on_windows": declared, "verified": False} if declared is not None
                                           else {"trained_on_windows": None, "verified": False,
                                                 "warning": "no provenance.json: training windows are UNDECLARED"})
            e = {"train": load_embeddings(d / "embeddings.csv")}
            for t in targets:
                if (d / f"{t}.csv").exists():
                    e[t] = load_embeddings(d / f"{t}.csv")
            emb[d.name] = e
            meta[d.name] = {"name": d.name, "description": "external submission",
                            "declared_training_windows": declared,
                            "shared_coordinate_system": (d / "SHARED_COORDINATES").exists()}
            external.append(d.name)
    implemented = [b.name for b in graph_methods_run]
    missing = [m for m in EXPECTED_EXTERNAL if m not in external and m not in implemented]

    # ---- evaluate on every target reference
    results: dict[str, dict[str, dict]] = {t: {} for t in targets}
    identity: dict[str, dict[str, dict]] = {t: {} for t in targets}
    for t in targets:
        ref_t = reference_dir / t
        feats = load_embeddings(ref_t / "features.csv", "host_id", "ref_")
        per_target = dict(emb)
        per_target["reference_ceiling"] = {"train": feats.rename(columns=lambda c: c.replace("ref_", "emb_"))}
        for name, windows in per_target.items():
            path = _write(out_dir / "embeddings" / name / "train.csv", windows["train"])
            # The ceiling only embeds the target-evaluable hosts, so it cannot take part in the learned link
            # scorer (its fit candidates come from the earlier validation window); it is a leaky object anyway.
            fit_ref = (reference_dir / "validation" if t == "test" and name != "reference_ceiling"
                       and (reference_dir / "validation").exists() else None)
            r = run(path, ref_t, name, n_boot=n_boot, seed=seed, fit_reference_dir=fit_ref)
            r["method_info"] = meta.get(name, {"name": name, "description": "target-window reference features",
                                                 "fit_windows": ["target (leaks by design)"]})
            results[t][name] = r
            (out_dir / "results" / t).mkdir(parents=True, exist_ok=True)
            (out_dir / "results" / t / f"{name}.json").write_text(
                json.dumps(r, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if name != "reference_ceiling" and t in windows:
                shared = meta[name].get("shared_coordinate_system", False)
                ident = {"shared_coordinate_system_declared": shared}
                hosts_t = hosts_by_target[t]
                for mode in ("none", "procrustes"):
                    try:
                        ident[mode] = identity_retrieval(windows["train"].loc[hosts_t], windows[t].loc[hosts_t],
                                                         alignment=mode,
                                                         seed=seed, n_boot=n_boot)
                    except ValueError as exc:
                        ident[mode] = {"unavailable": str(exc)}
                identity[t][name] = ident
                (out_dir / "identity" / t).mkdir(parents=True, exist_ok=True)
                (out_dir / "identity" / t / f"{name}.json").write_text(
                    json.dumps(ident, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        subprocess.run([sys.executable, "-m", "benchmark.build_leaderboard", "--results-dir",
                        str(out_dir / "results" / t), "--output", str(out_dir / f"leaderboard_{t}.csv")],
                       check=True, capture_output=True)

    # ---- audit
    audit = None
    if do_audit:
        hosts = hosts_by_target["test"]
        stable = [h for h in hosts if all(h in set(hosts_by_target[t]) for t in targets)]
        scored = {m: w["train"] for m, w in emb.items()}
        audit = {
            "weight_sensitivity": A.weight_sensitivity(processed_dir, hosts, "test", scored, n_configs=audit_configs, seed=seed),
            "reference_stability": A.reference_stability(processed_dir, stable, data.windows),
            "random_null": A.random_null(reference_dir / "test", data.cohort, out_dir / "_null", dim=dim, n=null_draws, seed=seed,
                                    fit_reference_dir=reference_dir / "validation" if "validation" in targets else None),
            "split_agreement": A.split_agreement(results, ["consistency@10_mean", "link_roc_auc",
                                                           "pair_roc_auc", "triplet_accuracy"]),
            "prevalence": A.prevalence(results["test"]),
            "information_overlap": {**A.OVERLAP, **{m: meta[m]["overlap"] for m in meta if meta[m].get("overlap")}},
        }
        A.dump(out_dir / "audit.json", audit)
        import shutil
        shutil.rmtree(out_dir / "_null", ignore_errors=True)

    # ---- manifest, figures, report
    manifest = {
        "data_kind": data.kind, "synthetic": data.synthetic,
        "real_lanl_declared": bool(data.manifest.get("real_lanl_declared")),
        "provenance_verified": False,
        "provenance": {k: (data.manifest.get("provenance") or {}).get(k)
                       for k in ("kind", "basis", "declared_kind", "source_note", "verified", "verification")},
        "badge": BADGE[data.kind],
        "processed_manifest": str(processed_dir / "manifest.json"),
        "cohort_sha256": data.manifest["cohort"]["sha256"], "n_hosts": len(data.cohort),
        "cohort": {"policy": data.cohort_policy, "n_cohort": len(data.cohort),
                   "n_evaluable": {t: len(hosts_by_target[t]) for t in targets},
                   "coverage": {t: len(hosts_by_target[t]) / len(data.cohort) for t in targets},
                   "selection": data.manifest["cohort"].get("selection")},
        "external_provenance": external_provenance,
        "reference_files_sha256": {t: results[t]["random"]["reference_info"]["files_sha256"] for t in targets},
        "config": {"dim": dim, "seed": seed, "bootstrap": n_boot, "targets": targets},
        "methods": sorted(results["test"]), "external_methods": external, "missing_expected_methods": missing,
        "graph_methods": {m: meta[m] for m in implemented},
        "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                        "numpy": np.__version__, "sklearn": sklearn.__version__, "git_sha": _git_sha()},
    }
    A.dump(out_dir / "manifest.json", manifest)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    tag = TAGS[data.kind]
    for t in targets:
        R.scorecard(results[t], fig_dir / f"scorecard_{t}.png", f"{tag} - {t}")
    R.knn_by_k(results["test"], fig_dir / "knn_by_k.png", f"{tag} - kNN Consistency vs k (test)")
    R.identity_plot(identity["test"], fig_dir / "identity_test.png", f"{tag} - identity retrieval (train -> test)")
    if audit:
        R.weights_plot(audit["weight_sensitivity"], fig_dir / "weight_sensitivity.png")
    (out_dir / "report.md").write_text(build_report(manifest, results, identity, audit), encoding="utf-8")
    return manifest


def build_report(manifest: dict, results: dict, identity: dict, audit: dict | None) -> str:
    kind = manifest["data_kind"]
    L = []
    L.append("# Benchmark run report\n")
    L.append(f"> **{BADGE[kind]}.**\n>\n> " + SCOPE[kind] + "\n")
    prov = manifest.get("provenance") or {}
    if kind == "lanl_declared":
        L.append(f"* source declared by the operator: *{prov.get('source_note')}*. The pipeline cannot verify it "
                 "(`provenance_verified: false`).")
    c = manifest["config"]
    L.append(f"* data kind: `{kind}` (basis: {prov.get('basis')}); cohort hosts: **{manifest['n_hosts']}**; "
             f"dim={c['dim']}, seed={c['seed']}, bootstrap={c['bootstrap']}; targets: {', '.join(c['targets'])}")
    L.append(f"* methods evaluated: {', '.join(manifest['methods'])}")
    L.append(f"* **not available in this run** (no embeddings supplied, nothing simulated): "
             f"{', '.join(manifest['missing_expected_methods']) or 'none'}")
    coh = manifest.get("cohort")
    if coh:
        cov = ", ".join(f"{t}: {coh['n_evaluable'][t]}/{coh['n_cohort']} ({coh['coverage'][t]:.0%})" for t in coh["n_evaluable"])
        L.append(f"* cohort policy `{coh['policy']}`; evaluable hosts per window: {cov}")
        if coh.get("selection"):
            L.append(f"  * {coh['selection']['note']}")
    for name, pv in (manifest.get("external_provenance") or {}).items():
        L.append(f"* external method `{name}`: trained on {pv['trained_on_windows']} (declared, unverified)"
                 if pv.get("trained_on_windows") is not None else
                 f"* external method `{name}`: **training windows undeclared** - leakage cannot be excluded")
    L.append("")
    cols = ["consistency@10_mean", "ndcg@10_mean", "link_roc_auc", "link_learned_roc_auc", "link_new_roc_auc",
            "pair_roc_auc", "triplet_accuracy", "precision@3", "recall@3"]
    for t in c["targets"]:
        rows = {m: {k: r["metrics"].get(k) for k in cols} for m, r in results[t].items()}
        df = pd.DataFrame(rows).T
        df.index.name = "method"
        L.append(f"\n## Scores - {t} reference\n")
        L.append(R.md_table(df))
        pers = next(iter(results[t].values()))["metric_details"].get("future_link_prediction", {}).get("persistence_baseline")
        if pers:
            L.append(f"\nLink-prediction reference point (no embedding): *pair already seen in the source window* "
                     f"scores ROC-AUC **{pers['roc_auc']:.3f}**. `link_learned_*` fits an edge classifier on the "
                     f"validation candidates and scores the test candidates; cosine is shown for completeness.")
        L.append(f"\n![scorecard]({'figures/scorecard_' + t + '.png'})\n")
    unavail = sorted({u for r in results["test"].values() for u in r["unavailable_metrics"]})
    L.append("\n## Metrics not available\n")
    L.extend(f"* {u}" for u in unavail)
    ci_rows = []
    for m, r in results["test"].items():
        for k in ("consistency@10_mean", "link_roc_auc", "pair_roc_auc", "triplet_accuracy"):
            ci = r["metrics_ci"].get(k)
            if ci:
                ci_rows.append({"method": m, "metric": k, "estimate": r["metrics"][k], "ci_low": ci[0], "ci_high": ci[1]})
    if ci_rows:
        L.append("\n## 95% bootstrap intervals (test)\n")
        t = pd.DataFrame(ci_rows).set_index("method")
        L.append(R.md_table(t))
    L.append("\n## Identity retrieval (train -> test)\n")
    rows = {}
    for m, v in identity["test"].items():
        rows[m] = {"shared coords declared": str(v["shared_coordinate_system_declared"]),
                   "Top-1 raw": v.get("none", {}).get("top1"), "Top-1 Procrustes": v.get("procrustes", {}).get("top1"),
                   "MRR Procrustes": v.get("procrustes", {}).get("mrr"),
                   "chance Top-1": v.get("procrustes", {}).get("chance_top1")}
    if rows:
        L.append(R.md_table(pd.DataFrame(rows).T))
        L.append("\n![identity](figures/identity_test.png)\n")
        L.append("Raw-coordinate scores are only interpretable for methods that declare a shared coordinate system.")
    if audit:
        w, s = audit["weight_sensitivity"], audit["reference_stability"]
        L.append("\n## Validity audit\n")
        L.append(f"* **Weight sensitivity** ({len(w['configs'])} perturbed behavioural references, k={w['k']}): "
                 f"neighbour agreement with the canonical reference {w['mean_reference_agreement']:.2f} on average "
                 f"(min {w['min_reference_agreement']:.2f}; chance {w['chance_agreement']:.3f}); "
                 f"method-ranking Kendall tau mean {w['mean_kendall_tau']:.2f}, min {w['min_kendall_tau']:.2f}.")
        for pair, v in s["pairs"].items():
            L.append(f"* **Reference stability** {pair}: Consistency@{s['k']} = {v['consistency']:.2f} "
                     f"(chance {s['chance_consistency']:.3f}) - behavioural neighbourhoods drift between windows.")
        if audit["split_agreement"]:
            L.append("* **Validation vs test ranking agreement** (Kendall tau): " +
                     ", ".join(f"{k}: {v:.2f}" for k, v in audit["split_agreement"].items() if v is not None))
        nm = audit["random_null"]["metrics"]
        L.append("* **Chance band (random embeddings, test reference)**: " + ", ".join(
            f"{k} {nm[k]['mean']:.3f} ± {nm[k]['std']:.3f}" for k in
            ("consistency@10_mean", "link_roc_auc", "link_learned_roc_auc", "pair_roc_auc", "triplet_accuracy") if k in nm))
        pv = audit["prevalence"]["future_links"]
        L.append(f"* **Link-prediction prevalence**: sampled positive rate {pv['sampled_positive_rate']:.2f} vs natural "
                 f"{pv['natural_prevalence']:.3f}; {pv['share_positives_seen_in_source']:.0%} of positive links were already "
                 f"present in the source window (compare 'New-link ROC-AUC').")
        L.append("\n![weights](figures/weight_sensitivity.png)\n")
        L.append("\n### Information shared by construction (method input vs reference source)\n")
        ov = pd.DataFrame(audit["information_overlap"]).T
        ov.index.name = "method"
        L.append(R.md_table(ov))
    L.append("\n## Reading these numbers\n")
    L.append("* **Predictive:** future links (outcome observed later) and identity stability. "
             "**Agreement with a proxy we chose:** kNN consistency (behavioural profile), pair/triplet/P@k (shared peers).")
    L.append("* `reference_ceiling` uses the target-window features themselves and leaks by design; "
             "baselines share feature definitions or peer structure with the references (table above), so they "
             "are calibration points, not independent evidence.")
    L.append("* Intervals: kNN and retrieval metrics resample hosts/anchors; link, pair and triplet metrics use a "
             "host-cluster bootstrap (pairs share hosts). They cover evaluation-set sampling only, not model seeds, "
             "reference choice or window choice.")
    L.append("* " + {"synthetic_fixture": "Because the data are synthetic, no statement about LANL follows from any number above.",
                    "unverified": "Because the origin is unverified, treat these as results on an unlabelled dataset.",
                    "lanl_declared": "Results hold for the declared files only; the cohort policy decides which hosts "
                                     "they describe (see above)."}[manifest["data_kind"]])
    return "\n".join(L) + "\n"


def _load_config(text: str | None) -> dict | None:
    if not text:
        return None
    p = Path(text)
    return json.loads(p.read_text(encoding="utf-8") if p.exists() else text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed-dir", type=Path, required=True)
    ap.add_argument("--reference-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--submissions-dir", type=Path)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--methods", nargs="+", default=None, choices=sorted(graph_methods.available()) + ["all"],
                    help="graph-embedding methods to fit on the train window in addition to the baselines")
    ap.add_argument("--method-config", default=None,
                    help='JSON (or a path to a JSON file) mapping a method to hyper-parameters, e.g. \'{"deepwalk": {"num_walks": 5}}\'')
    ap.add_argument("--fast-methods", action="store_true",
                    help="tiny hyper-parameters for every selected method (smoke tests only; results not meaningful)")
    ap.add_argument("--port-free-baselines", action="store_true",
                    help="also run the port-free behavioural baselines (feature ablation, issue #43)")
    ap.add_argument("--no-audit", action="store_true")
    ap.add_argument("--audit-configs", type=int, default=12)
    ap.add_argument("--null-draws", type=int, default=30)
    a = ap.parse_args()
    m = run_experiment(a.processed_dir, a.reference_dir, a.out_dir, dim=a.dim, seed=a.seed, n_boot=a.bootstrap,
                       submissions_dir=a.submissions_dir, do_audit=not a.no_audit,
                       audit_configs=a.audit_configs, null_draws=a.null_draws, methods=a.methods,
                       method_config=_load_config(a.method_config), fast_methods=a.fast_methods, port_free_baselines=a.port_free_baselines)
    print(json.dumps({k: m[k] for k in ("data_kind", "methods", "missing_expected_methods")}, indent=2))
    print(f"report: {a.out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
