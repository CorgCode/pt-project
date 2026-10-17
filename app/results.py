"""Чтение результата прогона только на чтение (без Streamlit, легко тестировать).

Прогон — это demo/synthetic_run (run/, reference/, processed_manifest.json) или --workdir от
experiments.end_to_end (run/, reference/, processed/). Дашборд показывает только сохранённые файлы;
пересчитываются лишь виды по хостам из сохранённых эмбеддингов и замороженных соседей.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from evaluation.frozen_neighbors import read_frozen_neighbors
from evaluation.io import load_embeddings
from evaluation.knn_consistency import neighborhood_metrics, top_k_neighbors
from pipeline.provenance import BADGE

TARGET_ORDER = ("test", "validation")
PALETTE = {"random": "#9aa0a6", "behavioral": "#1a73e8", "behavioral_pca": "#12b5cb",
           "behavioral_svd": "#7baaf7", "graph_spectral": "#e8710a", "reference_ceiling": "#188038"}
LEAKY = {"reference_ceiling"}

METRIC_GROUPS: dict[str, list[tuple[str, str]]] = {
    "Behavioural proxy (kNN)": [("consistency@5_mean", "Consistency@5"), ("consistency@10_mean", "Consistency@10"),
                                ("consistency@20_mean", "Consistency@20"), ("jaccard@10_mean", "Jaccard@10"),
                                ("ndcg@10_mean", "nDCG@10")],
    "Future links (predictive)": [("link_roc_auc", "ROC-AUC (cosine)"), ("link_learned_roc_auc", "ROC-AUC (learned)"),
                                  ("link_new_roc_auc", "ROC-AUC new links"), ("link_pr_auc_lift", "PR-AUC lift"),
                                  ("link_pr_auc_at_natural_prevalence", "PR-AUC @ natural prevalence")],
    "Shared-peer proxy (structural)": [("pair_roc_auc", "Pair ROC-AUC"), ("pair_pr_auc_lift", "Pair PR-AUC lift"),
                                       ("pair_pr_auc_at_judged_prevalence", "Pair PR-AUC @ judged prevalence (conditional)"),
                                       ("triplet_accuracy", "Triplet accuracy"), ("precision@3", "Precision@3"),
                                       ("recall@3", "Recall@3")],
    "External labels": [("probe_accuracy", "Probe accuracy"), ("probe_macro_f1", "Probe macro-F1")],
}
CHANCE = {"link_roc_auc": 0.5, "link_learned_roc_auc": 0.5, "link_new_roc_auc": 0.5, "pair_roc_auc": 0.5,
          "triplet_accuracy": 0.5, "link_pr_auc_lift": 1.0, "pair_pr_auc_lift": 1.0}


class BundleError(ValueError):
    """Папка не является прогоном."""


def discover(repo_root: Path) -> list[Path]:
    """Прогоны в репозитории: демо и любые runs/<имя>."""
    found = []
    demo = repo_root / "demo" / "synthetic_run"
    if (demo / "run" / "manifest.json").exists():
        found.append(demo)
    found += sorted(p for p in (repo_root / "runs").glob("*") if (p / "run" / "manifest.json").exists())
    return found


class Bundle:
    def __init__(self, root: Path):
        root = Path(root)
        run = root / "run" if (root / "run" / "manifest.json").exists() else root
        if not (run / "manifest.json").exists():
            raise BundleError(f"{root}: no run/manifest.json (run `python -m experiments.end_to_end` first)")
        self.root, self.run = root, run
        self.manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        self.kind = self.manifest.get("data_kind", "unverified")      # старые прогоны без пометки — unverified
        self.synthetic = self.kind == "synthetic_fixture"
        self.provenance_verified = False                              # пайплайн никогда не проверяет происхождение
        base = run.parent
        self.ref_dir = base / "reference"
        pm = [base / "processed_manifest.json", base / "processed" / "manifest.json"]
        self.processed_manifest = next((json.loads(p.read_text()) for p in pm if p.exists()), None)

    # ------------------------------------------------------------------ общие сведения
    @property
    def targets(self) -> list[str]:
        return [t for t in TARGET_ORDER if (self.run / "results" / t).exists()]

    @property
    def methods(self) -> list[str]:
        return list(self.manifest["methods"])

    @property
    def label(self) -> str:
        return {"synthetic_fixture": "SYNTHETIC FIXTURE", "lanl_declared": "LANL (declared)"}.get(self.kind, "UNVERIFIED")

    @property
    def badge(self) -> str:
        return BADGE.get(self.kind, BADGE["unverified"])

    @property
    def source_note(self) -> str | None:
        return (self.manifest.get("provenance") or {}).get("source_note")

    def cohort(self) -> dict | None:
        return self.manifest.get("cohort")

    def results(self, target: str) -> dict[str, dict]:
        d = self.run / "results" / target
        if not d.exists():
            raise BundleError(f"no results for target {target!r}")
        return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))}

    def identity(self, target: str) -> dict[str, dict]:
        d = self.run / "identity" / target
        return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))} if d.exists() else {}

    def audit(self) -> dict | None:
        p = self.run / "audit.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def reference_manifest(self, target: str) -> dict | None:
        p = self.ref_dir / target / "manifest.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    # ------------------------------------------------------------------ таблицы
    def board(self, target: str, keys: list[str] | None = None) -> pd.DataFrame:
        """Методы x метрики; пропущенные метрики — NaN, ничем не заполняются."""
        res = self.results(target)
        keys = keys or [k for g in METRIC_GROUPS.values() for k, _ in g]
        rows = {m: {k: r["metrics"].get(k, np.nan) for k in keys} for m, r in res.items()}
        return pd.DataFrame(rows).T[keys].rename_axis("method")

    def ci(self, target: str, method: str, key: str) -> tuple[float, float] | None:
        c = self.results(target)[method].get("metrics_ci", {}).get(key)
        return (float(c[0]), float(c[1])) if c else None

    def unavailable(self, target: str = "test") -> list[str]:
        res = self.results(target)
        return sorted({u for r in res.values() for u in r.get("unavailable_metrics", [])})

    def link_context(self, target: str = "test") -> dict:
        d = next(iter(self.results(target).values())).get("metric_details", {}).get("future_link_prediction", {})
        return {"persistence_roc_auc": (d.get("persistence_baseline") or {}).get("roc_auc"),
                "natural_prevalence": d.get("natural_prevalence"), "sampled_positive_rate": d.get("positive_rate"),
                "share_positives_seen_in_source": d.get("share_positives_seen_in_source"),
                "n_examples": d.get("n_examples")}

    # ------------------------------------------------------------------ эмбеддинги и соседи
    def embedding(self, method: str) -> pd.DataFrame:
        p = self.run / "embeddings" / method / "train.csv"
        if not p.exists():
            raise BundleError(f"embeddings of {method!r} are not in the bundle")
        return load_embeddings(p)

    def eval_hosts(self) -> list[str]:
        """Хосты, которые покрывает сохранённый test-эталон."""
        p = self.ref_dir / "test" / "eval_hosts.csv"
        if p.exists():
            return pd.read_csv(p, dtype=str)["host_id"].tolist()
        return sorted(self.reference_neighbors()["host_id"].unique())

    def scored_embedding(self, method: str) -> pd.DataFrame:
        """Эмбеддинг только по реально оцененным хостам."""
        emb = self.embedding(method)
        return emb.loc[self.eval_hosts()] if self.has_neighbors() else emb

    def has_neighbors(self) -> bool:
        return (self.ref_dir / "test" / "neighbors.csv").exists()

    def reference_neighbors(self) -> pd.DataFrame:
        p = self.ref_dir / "test" / "neighbors.csv"
        if not p.exists():
            raise BundleError("reference/test/neighbors.csv is not in the bundle")
        return pd.read_csv(p, dtype={"host_id": str, "neighbor_id": str})

    def per_host_agreement(self, method: str, k: int = 10) -> pd.Series:
        """Consistency@k по каждому хосту между соседями в эмбеддинге и замороженными соседями."""
        emb = self.scored_embedding(method)
        ref = read_frozen_neighbors(self.ref_dir / "test" / "neighbors.csv", list(emb.index), k_max=k, split="test")
        ph = neighborhood_metrics(ref[:, :k], top_k_neighbors(emb.to_numpy(float), k))
        ph.index = emb.index
        return ph["consistency"]

    def projection(self, method: str, k: int = 10) -> pd.DataFrame:
        """2D PCA оцениваемого эмбеддинга; совпадение по хостам, если есть эталон."""
        emb = self.scored_embedding(method)
        if emb.shape[1] < 2:
            raise BundleError("embedding has a single dimension; no 2D projection")
        xy = PCA(n_components=2, random_state=0).fit_transform(emb.to_numpy(float))
        df = pd.DataFrame({"host_id": emb.index, "PC1": xy[:, 0], "PC2": xy[:, 1]}).set_index("host_id")
        if self.has_neighbors():
            df["agreement"] = self.per_host_agreement(method, k)
        return df

    def neighborhood(self, method: str, host: str, k: int = 10) -> pd.DataFrame:
        """Соседи host в эмбеддинге с косинусом и пометкой, есть ли они в эталоне."""
        emb = self.scored_embedding(method)
        if host not in emb.index:
            raise BundleError(f"unknown host {host!r}")
        x = emb.to_numpy(float)
        unit = x / np.linalg.norm(x, axis=1, keepdims=True)
        sim = unit @ unit[emb.index.get_loc(host)]
        order = [i for i in np.argsort(-sim, kind="stable") if emb.index[i] != host][:k]
        out = pd.DataFrame({"rank": range(1, len(order) + 1), "host_id": emb.index[order], "cosine": sim[order]})
        if self.has_neighbors():
            ref = self.reference_neighbors()
            r = ref[ref.host_id == host].set_index("neighbor_id")["rank"]
            out["reference_rank"] = out.host_id.map(r)
            out["in_reference_top_k"] = out.reference_rank.le(k)
        return out

    def reference_list(self, host: str, k: int = 10) -> pd.DataFrame:
        ref = self.reference_neighbors()
        return ref[(ref.host_id == host) & (ref["rank"] <= k)][["rank", "neighbor_id"]].rename(
            columns={"neighbor_id": "host_id"}).reset_index(drop=True)
