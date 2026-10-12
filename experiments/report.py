"""Markdown report and figures for a benchmark run (matplotlib, no styling tricks)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HEADLINE = [("consistency@10_mean", "kNN Consistency@10", 0.0), ("ndcg@10_mean", "nDCG@10", 0.0),
            ("link_roc_auc", "Link ROC-AUC (cosine)", 0.5), ("link_learned_roc_auc", "Link ROC-AUC (learned)", 0.5),
            ("pair_roc_auc", "Pair ROC-AUC", 0.5), ("triplet_accuracy", "Triplet accuracy", 0.5)]
PALETTE = {"random": "#9aa0a6", "behavioral": "#1a73e8", "behavioral_pca": "#12b5cb",
           "behavioral_svd": "#7baaf7", "graph_spectral": "#e8710a", "reference_ceiling": "#188038"}


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, [df.index.name or "method", *cols])) + " |",
           "|" + "|".join(["---"] * (len(cols) + 1)) + "|"]
    for idx, row in df.iterrows():
        cells = []
        for v in row:
            cells.append("n/a" if v is None or (isinstance(v, float) and np.isnan(v))
                         else floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
        out.append("| " + " | ".join([str(idx), *cells]) + " |")
    return "\n".join(out)


def _color(name: str) -> str:
    return PALETTE.get(name, "#5f6368")


def scorecard(results: dict[str, dict], path: Path, title: str) -> None:
    methods = list(results)
    fig, axes = plt.subplots(1, len(HEADLINE), figsize=(3.0 * len(HEADLINE), 3.6), sharey=False)
    for ax, (key, label, chance) in zip(axes, HEADLINE):
        vals = [results[m]["metrics"].get(key) for m in methods]
        err = []
        for m, v in zip(methods, vals):
            ci = results[m].get("metrics_ci", {}).get(key)
            err.append([max(0, v - ci[0]), max(0, ci[1] - v)] if (ci and v is not None) else [0, 0])
        y = [v if v is not None else np.nan for v in vals]
        ax.bar(range(len(methods)), y, color=[_color(m) for m in methods],
               yerr=np.array(err).T, capsize=2, ecolor="#3c4043")
        ax.axhline(chance, color="#d93025", lw=1, ls="--")
        ax.set_title(label, fontsize=9)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=60, ha="right", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title + "  (dashed = chance; bars show 95% bootstrap CI)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def knn_by_k(results: dict[str, dict], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ks = (5, 10, 20)
    for m, r in results.items():
        ax.plot(ks, [r["metrics"].get(f"consistency@{k}_mean", np.nan) for k in ks], marker="o",
                color=_color(m), label=m)
    n = next(iter(results.values()))["n_hosts"]
    ax.plot(ks, [k / (n - 1) for k in ks], color="#d93025", ls="--", label="chance")
    ax.set_xlabel("k")
    ax.set_ylabel("Consistency@k")
    ax.set_xticks(ks)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def identity_plot(identity: dict[str, dict], path: Path, title: str) -> None:
    methods = [m for m in identity if any("top1" in identity[m].get(a, {}) for a in ("none", "procrustes"))]
    if not methods:
        return
    fig, ax = plt.subplots(figsize=(6, 3.6))
    w = 0.38
    for j, mode in enumerate(("none", "procrustes")):
        # raw coordinates are only meaningful when the method declares ONE coordinate system
        v = [np.nan if (mode == "none" and not identity[m].get("shared_coordinate_system_declared"))
             else identity[m].get(mode, {}).get("top1", np.nan) for m in methods]
        ax.bar(np.arange(len(methods)) + (j - 0.5) * w, v, w, label=("raw (shared-coordinate methods only)" if mode == "none" else "Procrustes-aligned"),
               color=("#1a73e8" if mode == "none" else "#e8710a"), alpha=0.9)
    any_m = next(iter(methods))
    chance = identity[any_m].get("procrustes", identity[any_m].get("none", {})).get("chance_top1")
    if chance:
        ax.axhline(chance, color="#d93025", ls="--", lw=1, label="chance")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Top-1 identity retrieval")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def weights_plot(audit: dict, path: Path) -> None:
    cfgs = audit["configs"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    axes[0].hist([c["reference_agreement"] for c in cfgs], bins=8, color="#1a73e8")
    axes[0].axvline(audit["chance_agreement"], color="#d93025", ls="--")
    axes[0].set_xlabel("neighbour agreement of reweighted vs canonical reference")
    axes[1].bar(range(len(cfgs)), [c["kendall_tau"] if c["kendall_tau"] is not None else np.nan for c in cfgs],
                color=["#e8710a" if c["dropped_block"] else "#12b5cb" for c in cfgs])
    axes[1].set_ylim(-1, 1)
    axes[1].set_xlabel("weight configuration (orange: one block removed)")
    axes[1].set_ylabel("Kendall tau of method ranking")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
