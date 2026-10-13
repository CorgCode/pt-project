"""Feature-block ablation of the behavioural baseline, with the port-free variant first (issue #43).

    python -m experiments.feature_ablation --processed-dir data/processed --reference-dir data/reference \
        --out-dir runs/ablation_ports

What is compared. Every variant is the train-window behavioural profile (volume, peers, ratio, ports, proto,
hourly) with some blocks removed ENTIRELY: `full`, `no_ports` (the issue's port-free baseline) and one
leave-one-block-out variant per remaining block. All variants use the SAME hosts, windows and frozen references
and are scored by the common evaluator (`evaluation.evaluate.run`); the files' SHA-256 are recorded to prove it.
For `full` vs `no_ports` the report adds PAIRED differences with confidence intervals (hosts are resampled
together for both variants: per-host kNN consistency; host-cluster bootstrap for pair/link/triplet metrics).

A separate DIAGNOSTIC (not a leaderboard reference): the target-window kNN reference rebuilt without ports. It
shows how much of the standard behavioural proxy is carried by ports and how both variants fare when ports are
removed from the reference as well. It has a different fingerprint and must not be mixed with leaderboard scores.

Everything is fitted on the TRAIN window only; the test window only provides the references.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from benchmark.baselines.temporal import BehavioralBaseline
from evaluation.evaluate import run
from evaluation.frozen_neighbors import evaluate_frozen_neighbors, save_frozen_neighbors
from evaluation.future_links import load_candidate_edges
from evaluation.io import cosine_pairs, load_embeddings
from evaluation.knn_consistency import BLOCKS, build_reference_features, evaluate_embedding
from evaluation.retrieval import load_pairs
from evaluation.temporal_reference import load_flows
from evaluation.triplet_accuracy import load_triplets
from evaluation.uncertainty import bootstrap_ci, dyadic_bootstrap_ci, host_index
from pipeline.data import ProcessedData
from pipeline.provenance import BADGE

KS = (5, 10, 20)
HEADLINE = [("consistency@10_mean", "kNN Consistency@10"), ("ndcg@10_mean", "nDCG@10"),
            ("link_roc_auc", "Link ROC-AUC (cosine)"), ("link_learned_roc_auc", "Link ROC-AUC (learned)"),
            ("pair_roc_auc", "Pair ROC-AUC"), ("triplet_accuracy", "Triplet accuracy")]


def _embedding(data: ProcessedData, excluded: tuple[str, ...]) -> pd.DataFrame:
    return BehavioralBaseline(exclude_blocks=excluded).fit(data).embed(data, "train")


def _variants(leave_one_out: bool) -> dict[str, tuple[str, ...]]:
    v = {"full": (), "no_ports": ("ports",)}
    if leave_one_out:
        v.update({f"no_{b}": (b,) for b in BLOCKS if b != "ports"})
    return v


def _paired_ci(stat_a_minus_b, n_items: int, n_boot: int, seed: int):
    return bootstrap_ci(stat_a_minus_b, n_items, n_boot=n_boot, seed=seed)


def _diff_record(estimate: float, ci) -> dict:
    return {"difference": float(estimate), "ci95": None if ci is None else [float(ci[0]), float(ci[1])],
            "ci_excludes_zero": None if ci is None else bool(ci[0] > 0 or ci[1] < 0)}


def paired_differences(emb_a: pd.DataFrame, emb_b: pd.DataFrame, reference_dir: Path, split: str,
                       n_boot: int, seed: int) -> dict:
    """a minus b on the standard reference, with paired bootstrap intervals."""
    ev = pd.read_csv(reference_dir / "eval_hosts.csv", dtype=str)["host_id"].tolist() \
        if (reference_dir / "eval_hosts.csv").exists() else list(emb_a.index)
    ea, eb = emb_a.loc[ev], emb_b.loc[ev]
    out: dict = {}
    nn = reference_dir / "neighbors.csv"
    if nn.exists():
        _, pa = evaluate_frozen_neighbors(ea, nn, KS, split=split, return_per_host=True)
        _, pb = evaluate_frozen_neighbors(eb, nn, KS, split=split, return_per_host=True)
        for k in KS:
            d = pa[k]["consistency"].to_numpy() - pb[k]["consistency"].to_numpy()
            out[f"consistency@{k}"] = _diff_record(d.mean(), _paired_ci(lambda i, d=d: float(d[i].mean()), len(d), n_boot, seed))
    pf = reference_dir / "pairs.csv"
    if pf.exists():
        p = load_pairs(pf)
        p = p[p["split"] == split]
        y = p["label"].astype(int).to_numpy()
        sa, sb = (cosine_pairs(e, p.host_a.tolist(), p.host_b.tolist()) for e in (ea, eb))
        cols, n = host_index(p.host_a, p.host_b)
        f = lambda w: roc_auc_score(y, sa, sample_weight=w) - roc_auc_score(y, sb, sample_weight=w)
        out["pair_roc_auc"] = _diff_record(roc_auc_score(y, sa) - roc_auc_score(y, sb), dyadic_bootstrap_ci(f, cols, n, n_boot=n_boot, seed=seed))
    ef = reference_dir / "edges.csv"
    if ef.exists():
        e = load_candidate_edges(ef)
        e = e[e["split"] == split]
        y = e["label"].astype(int).to_numpy()
        sa, sb = (cosine_pairs(m, e.host_a.tolist(), e.host_b.tolist()) for m in (emb_a, emb_b))
        cols, n = host_index(e.host_a, e.host_b)
        f = lambda w: roc_auc_score(y, sa, sample_weight=w) - roc_auc_score(y, sb, sample_weight=w)
        out["link_roc_auc"] = _diff_record(roc_auc_score(y, sa) - roc_auc_score(y, sb), dyadic_bootstrap_ci(f, cols, n, n_boot=n_boot, seed=seed))
    tf = reference_dir / "triplets.csv"
    if tf.exists():
        t = load_triplets(tf)
        t = t[t["split"] == split]

        def correct(emb):
            sp = cosine_pairs(emb, t.anchor.tolist(), t.positive.tolist())
            sn = cosine_pairs(emb, t.anchor.tolist(), t.negative.tolist())
            return (sp > sn).astype(float) + 0.5 * np.isclose(sp, sn, rtol=0, atol=1e-12)
        ca, cb = correct(ea), correct(eb)
        cols, n = host_index(t.anchor, t.positive, t.negative)
        f = lambda w: float(np.average(ca, weights=w) - np.average(cb, weights=w))
        out["triplet_accuracy"] = _diff_record(ca.mean() - cb.mean(), dyadic_bootstrap_ci(f, cols, n, n_boot=n_boot, seed=seed))
    return out


def port_free_reference_diagnostic(data: ProcessedData, reference_dir: Path, embeddings: dict[str, pd.DataFrame],
                                   n_boot: int, seed: int) -> dict:
    """kNN agreement against a target-window reference rebuilt WITHOUT ports (analysis only, not a leaderboard)."""
    test_flows = load_flows(data.root / "flows" / "test.csv")
    hosts = pd.read_csv(reference_dir / "eval_hosts.csv", dtype=str)["host_id"].tolist() \
        if (reference_dir / "eval_hosts.csv").exists() else data.cohort
    feats_full = build_reference_features(test_flows, hosts=hosts)
    feats_free = build_reference_features(test_flows, hosts=hosts, exclude_blocks=("ports",))
    self_agree = evaluate_embedding(feats_free, feats_full, ks=KS)          # port-free vs full reference neighbours
    out: dict = {
        "note": ("reference rebuilt from the same target window WITHOUT the port block; different fingerprint, "
                 "NOT comparable with leaderboard scores"),
        "reference_self_agreement": {f"consistency@{k}": float(self_agree.loc[(k, "consistency"), "mean"]) for k in KS},
        "chance": {f"consistency@{k}": k / (len(hosts) - 1) for k in KS},
    }
    with tempfile.TemporaryDirectory() as tmp:
        ref = feats_free.copy()
        ref.index.name = "host_id"
        save_frozen_neighbors(ref, Path(tmp) / "neighbors.csv", reference_window="port-free diagnostic", k_max=max(KS), split="test")
        per = {}
        for name, emb in embeddings.items():
            s, ph = evaluate_frozen_neighbors(emb.loc[hosts], Path(tmp) / "neighbors.csv", KS, split="test", return_per_host=True)
            per[name] = ph
            out[name] = {f"consistency@{k}": float(s.loc[(k, "consistency"), "mean"]) for k in KS}
        d = per["full"][10]["consistency"].to_numpy() - per["no_ports"][10]["consistency"].to_numpy()
        out["paired_full_minus_no_ports@10"] = _diff_record(d.mean(), _paired_ci(lambda i: float(d[i].mean()), len(d), n_boot, seed))
    return out


def run_ablation(processed_dir: Path, reference_dir: Path, out_dir: Path, *, leave_one_out: bool = True,
                 n_boot: int = 200, seed: int = 0) -> dict:
    processed_dir, reference_dir, out_dir = Path(processed_dir), Path(reference_dir), Path(out_dir)
    data = ProcessedData(processed_dir)
    ref = reference_dir / "test"
    if not ref.exists():
        raise ValueError(f"{ref} not found; run `python -m pipeline.reference` first")
    fit_ref = reference_dir / "validation" if (reference_dir / "validation").exists() else None
    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings = {name: _embedding(data, ex) for name, ex in _variants(leave_one_out).items()}
    results = {}
    for name, emb in embeddings.items():
        path = out_dir / "embeddings" / name / "train.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        emb.to_csv(path)
        results[name] = run(path, ref, f"behavioral_{name}", n_boot=n_boot, seed=seed, fit_reference_dir=fit_ref)
    fingerprints = {n: json.dumps(r["reference_info"]["files_sha256"], sort_keys=True) for n, r in results.items()}
    identical = len(set(fingerprints.values())) == 1 and len({r["n_hosts"] for r in results.values()}) == 1
    if not identical:                                                       # cannot happen unless run() changes
        raise RuntimeError("variants were not evaluated on identical references and hosts")
    report = {
        "data_kind": data.kind, "badge": BADGE[data.kind], "synthetic": data.synthetic, "seed": seed, "n_bootstrap": n_boot,
        "protocol": {"split": results["full"]["split"], "n_hosts_scored": results["full"]["n_hosts"],
                     "reference_files_sha256": results["full"]["reference_info"]["files_sha256"],
                     "identical_for_all_variants": True, "fit_window": "train", "cohort_policy": data.cohort_policy,
                     "variants": {n: list(ex) for n, ex in _variants(leave_one_out).items()}},
        "metrics": {n: {k: r["metrics"].get(k) for k, _ in HEADLINE} for n, r in results.items()},
        "metrics_ci": {n: {k: r["metrics_ci"].get(k) for k, _ in HEADLINE} for n, r in results.items()},
        "paired_full_minus_no_ports": paired_differences(embeddings["full"], embeddings["no_ports"], ref,
                                                         results["full"]["split"], n_boot, seed),
        "port_free_reference_diagnostic": port_free_reference_diagnostic(
            data, ref, {"full": embeddings["full"], "no_ports": embeddings["no_ports"]}, n_boot, seed),
        "port_block_share_of_profile_weight": 1.0 / (1.0 + 1.0 + 1.0 + 1.0 + 0.5 + 1.0),
    }
    (out_dir / "ablation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "ablation.md").write_text(render(report), encoding="utf-8")
    return report


SCOPE = {"synthetic_fixture": "Synthetic data: the fixture's roles use role-specific ports BY CONSTRUCTION, so ports are informative here; "
                              "nothing follows for LANL.",
         "unverified": "Origin of the data is not established; nothing here is presented as a LANL result.",
         "lanl_declared": "Findings apply to the declared LANL files under this protocol only."}


def _f(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}"


def _diff_cell(rec: dict | None) -> str:
    """'+0.012 [+0.003, +0.021] *' ; the interval is omitted when no bootstrap was run (n_bootstrap = 0)."""
    if rec is None:
        return "not paired"
    if rec["ci95"] is None:
        return f"{rec['difference']:+.3f} (no interval: bootstrap off)"
    return f"{rec['difference']:+.3f} [{rec['ci95'][0]:+.3f}, {rec['ci95'][1]:+.3f}]" + (" *" if rec["ci_excludes_zero"] else "")


def render(rep: dict) -> str:
    L = ["# Port-free behavioural baseline: feature ablation\n", f"> **{rep['badge']}.** {SCOPE[rep['data_kind']]}\n"]
    pr = rep["protocol"]
    L.append(f"Identical for every variant (checked by file hash): split `{pr['split']}`, {pr['n_hosts_scored']} scored hosts, the same "
             f"frozen reference files, fit on the `{pr['fit_window']}` window only, cohort policy `{pr['cohort_policy']}`.\n")
    L.append("## Standard reference: full profile vs port-free\n")
    L.append("| metric | full | port-free | full - port-free (paired 95% CI) |\n|---|---|---|---|")
    keymap = {"consistency@10_mean": "consistency@10", "link_roc_auc": "link_roc_auc", "pair_roc_auc": "pair_roc_auc",
              "triplet_accuracy": "triplet_accuracy"}
    for k, label in HEADLINE:
        a, b = rep["metrics"]["full"][k], rep["metrics"]["no_ports"][k]
        d = rep["paired_full_minus_no_ports"].get(keymap.get(k, ""), None)
        cell = _diff_cell(d)
        L.append(f"| {label} | {_f(a)} | {_f(b)} | {cell} |")
    L.append("\n`*` the paired interval excludes 0. Intervals resample hosts jointly for both variants (host-cluster bootstrap for "
             "pair/link/triplet metrics); they cover evaluation-set sampling only.\n")
    L.append("## Leave-one-block-out (same protocol)\n")
    L.append("| variant (blocks removed) | " + " | ".join(l for _, l in HEADLINE) + " |\n|---|" + "---|" * len(HEADLINE))
    for n, ex in pr["variants"].items():
        L.append(f"| {n} ({', '.join(ex) or 'none'}) | " + " | ".join(_f(rep["metrics"][n][k]) for k, _ in HEADLINE) + " |")
    dg = rep["port_free_reference_diagnostic"]
    L.append("\n## Diagnostic: reference rebuilt WITHOUT ports (analysis only)\n")
    L.append(f"_{dg['note']}_\n")
    L.append("| kNN Consistency@k | full | port-free | chance |\n|---|---|---|---|")
    for k in KS:
        L.append(f"| @{k} | {_f(dg['full'][f'consistency@{k}'])} | {_f(dg['no_ports'][f'consistency@{k}'])} | {_f(dg['chance'][f'consistency@{k}'])} |")
    p = dg["paired_full_minus_no_ports@10"]
    L.append(f"\nPaired full - port-free at k=10: {_diff_cell(p)}. "
             f"Agreement of the port-free reference's neighbours with the standard reference's: "
             f"Consistency@10 = {_f(dg['reference_self_agreement']['consistency@10'])} (chance {_f(dg['chance']['consistency@10'])}).\n")
    L.append("## Methodological implications\n")
    std = rep["paired_full_minus_no_ports"].get("consistency@10")
    L.append(f"1. **The standard behavioural reference contains the port block** (about {rep['port_block_share_of_profile_weight']:.0%} of the profile's "
             "squared weight). The full baseline therefore shares that information with the reference **by construction**; the "
             "port-free baseline removes this overlap. A gap on the kNN metrics measures information overlap, not representation quality"
             + (f" (here the paired kNN@10 difference is {std['difference']:+.3f})." if std else "."))
    L.append("2. **Only the kNN reference involves ports.** The shared-peer pair/triplet references and the future-link candidates are "
             "built from communication partners, so the port ablation on those metrics measures how much *service identity* adds to "
             "structural/predictive signal, not overlap with the reference.")
    L.append("3. **The port-free reference diagnostic isolates what is left.** If the full-minus-port-free gap shrinks or vanishes when "
             "ports are removed from the reference too, the standard-reference gap was port overlap; if it persists, other blocks drive it. "
             "The diagnostic is not a leaderboard reference and must not be mixed with leaderboard scores.")
    L.append("4. **What ports are in LANL netflow.** Well-known ports are kept and others are anonymised to stable `PortNNNN` tokens, so the "
             "port block partly acts as a service identifier. A port-free profile tests reliance on service identity versus traffic shape "
             "(volume, direction, peers, protocols, time of day).")
    L.append("5. **The leaderboard is unchanged.** The standard references, hosts and windows are the same; the port-free variants are added "
             "rows (`--port-free-baselines`), not a new protocol. Removing a block does not rescale the others (each block is standardised on "
             "its own), so the kept columns are identical to the full profile's.")
    L.append("6. **Limits.** One window pair, one cohort, no host-role labels; the ablation says which information the *proxy* rewards, not "
             "which information identifies roles.")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed-dir", type=Path, required=True)
    ap.add_argument("--reference-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--no-leave-one-out", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rep = run_ablation(a.processed_dir, a.reference_dir, a.out_dir, leave_one_out=not a.no_leave_one_out,
                       n_boot=a.bootstrap, seed=a.seed)
    print((a.out_dir / "ablation.md").read_text())


if __name__ == "__main__":
    main()
