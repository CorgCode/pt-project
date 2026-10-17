"""Единая оценка эмбеддинга: одна команда считает все метрики, для которых есть эталон.

    python -m evaluation.evaluate --embeddings E.csv --reference-dir reference/test \
        --method NAME --output results/NAME.json

Все метрики перечислены в METRICS ниже. Какая метрика считается, решают файлы в папке эталона
(их один раз строит `python -m pipeline.reference`). Нет файла — метрика помечается
«недоступна» с причиной; файл есть, но битый — ошибка.
Узнаваемость во времени (evaluation.identity) требует двух окон и считается в experiments.
Описание метрик — evaluation/README.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from evaluation.frozen_neighbors import evaluate_frozen_neighbors
from evaluation.future_links import (load_candidate_edges, persistence_baseline, score_embeddings,
                                      score_learned)
from evaluation.io import load_embeddings
from evaluation.knn_consistency import evaluate_embedding
from evaluation.linear_probe import linear_probe, load_labels
from evaluation.partners import load_partners, partner_ranking
from evaluation.retrieval import load_pairs, pair_auc, precision_recall_at_k
from evaluation.triplet_accuracy import load_triplets, triplet_accuracy
from evaluation.uncertainty import bootstrap_ci

KS = (5, 10, 20)
RETRIEVAL_KS = (1, 3, 5)
REFERENCE_FILES = ("features.csv", "neighbors.csv", "edges.csv", "partners.csv", "pairs.csv", "triplets.csv",
                   "labels.csv", "eval_hosts.csv")


@dataclass
class Context:
    emb: pd.DataFrame            # только оцениваемые хосты
    emb_full: pd.DataFrame       # все хосты из файла
    ref: Path
    protocol: dict
    split: str
    n_boot: int
    seed: int
    fit_ref: Path | None
    result: dict

    def ci(self, name: str, ci, method: str = "item bootstrap (hosts / anchors resampled)") -> None:
        if ci is not None:
            self.result["metrics_ci"][name] = [float(ci[0]), float(ci[1])]
            self.result.setdefault("ci_methods", {})[name] = method


@dataclass
class Metric:
    key: str                     # статус в metric_status
    group: str                   # на какой вопрос отвечает
    title: str
    needs: tuple[str, ...]       # нужен хотя бы один из файлов
    compute: Callable[[Context], None]
    headline: str                # главное число для таблицы результатов
    chance: Callable[[dict], float | None]
    also: tuple[str, ...] = ()   # дополнительные статусы, которые ставит та же функция
    show: bool = True            # показывать ли главное число в таблице результатов


# ---- метрики ------------------------------------------------------------------------------

def _knn(c: Context) -> None:
    neighbors, features = c.ref / "neighbors.csv", c.ref / "features.csv"
    if neighbors.exists():
        summary, per_host = evaluate_frozen_neighbors(c.emb, neighbors, KS, split=c.split, return_per_host=True)
    else:
        summary, per_host = evaluate_embedding(c.emb, load_embeddings(features, "host_id", "ref_"),
                                               ks=KS, return_per_host=True)
    for (k, metric), row in summary.iterrows():
        for stat in ("mean", "median", "std"):
            c.result["metrics"][f"{metric}@{k}_{stat}"] = float(row[stat])
    for k, ph in per_host.items():
        for metric in ph.columns:
            v = ph[metric].to_numpy()
            c.ci(f"{metric}@{k}_mean", bootstrap_ci(lambda i, v=v: float(v[i].mean()), len(v),
                                                    n_boot=c.n_boot, seed=c.seed))


def _links(c: Context) -> None:
    edges_file = c.ref / "edges.csv"
    edges = load_candidate_edges(edges_file)
    r = score_embeddings(c.emb, edges, split=c.split, n_boot=c.n_boot, seed=c.seed)
    m = c.result["metrics"]
    m["link_roc_auc"], m["link_pr_auc"], m["link_pr_auc_lift"] = r["roc_auc"], r["pr_auc"], r["pr_auc_lift"]
    if r.get("pr_auc_at_natural_prevalence") is not None:
        m["link_pr_auc_at_natural_prevalence"] = r["pr_auc_at_natural_prevalence"]
    if "new_links" in r:
        m["link_new_roc_auc"], m["link_new_pr_auc"] = r["new_links"]["roc_auc"], r["new_links"]["pr_auc"]
    c.ci("link_roc_auc", r["roc_auc_ci"], r["ci_method"])
    c.ci("link_pr_auc", r["pr_auc_ci"], r["ci_method"])
    details = {k: r[k] for k in ("n_examples", "n_positive", "positive_rate", "natural_prevalence",
                                 "share_positives_seen_in_source", "source_window", "target_window", "split",
                                 "interpretation", "new_links") if k in r}
    pers = persistence_baseline(edges, c.split)
    if pers is not None:
        details["persistence_baseline"] = pers
    c.result["metric_details"]["future_link_prediction"] = details
    fit_edges = c.fit_ref / "edges.csv" if c.fit_ref else None
    if fit_edges is not None and fit_edges.exists():
        fit_split = json.loads((c.fit_ref / "manifest.json").read_text()).get("split", "validation")
        lr = score_learned(c.emb_full, load_candidate_edges(fit_edges), edges,
                           fit_split=fit_split, eval_split=c.split, seed=c.seed, n_boot=c.n_boot)
        m["link_learned_roc_auc"], m["link_learned_pr_auc"] = lr["roc_auc"], lr["pr_auc"]
        m["link_learned_random_roc_auc"] = lr["random_roc_auc"]
        m["link_learned_pr_auc_lift"] = lr["pr_auc_lift"]
        c.ci("link_learned_roc_auc", lr["roc_auc_ci"], lr["ci_method"])
        c.result["metric_details"]["link_learned"] = {k: v for k, v in lr.items() if not k.endswith("_ci")}
        c.result["reference_info"]["fit_reference_files_sha256"] = {
            f: _sha(c.fit_ref / f) for f in REFERENCE_FILES if (c.fit_ref / f).exists()}


def _links_learned(c: Context) -> None:
    if "link_learned_roc_auc" not in c.result["metrics"]:
        raise _Unavailable("нет эталона validation или его хостов в эмбеддинге — классификатору не на чем учиться")


def _partners(c: Context) -> None:
    r = partner_ranking(c.emb, load_partners(c.ref / "partners.csv"), n_boot=c.n_boot, seed=c.seed)
    m = c.result["metrics"]
    for view, prefix in (("all", "partner"), ("new", "partner_new")):
        m[f"{prefix}_mrr"] = r[view]["embedding"]["mrr"]
        m[f"{prefix}_recall@{r['top_n']}"] = r[view]["embedding"]["recall_at_n"]
        m[f"{prefix}_history_mrr"] = r[view]["history"]["mrr"]
        m[f"{prefix}_popularity_mrr"] = r[view]["popularity"]["mrr"]
        m[f"{prefix}_lift"] = r[view]["lift_vs_best_baseline"]
    c.ci("partner_mrr", r["all"]["embedding"].get("mrr_ci"))
    c.ci("partner_new_mrr", r["new"]["embedding"].get("mrr_ci"))
    c.result["metric_details"]["partner_ranking"] = r


def _pairs(c: Context) -> None:
    pairs = load_pairs(c.ref / "pairs.csv")
    j = c.protocol.get("judgments") or {}
    n_pos, n_neg = j.get("n_positive_pairs", 0), j.get("n_negative_pairs", 0)
    judged_prev = n_pos / (n_pos + n_neg) if n_pos and n_neg else None
    a = pair_auc(c.emb, pairs, c.split, judged_prevalence=judged_prev, n_boot=c.n_boot, seed=c.seed)
    m = c.result["metrics"]
    m["pair_roc_auc"], m["pair_pr_auc"], m["pair_pr_auc_lift"] = a["roc_auc"], a["pr_auc"], a["pr_auc_lift"]
    if "pr_auc_at_judged_prevalence" in a:
        m["pair_pr_auc_at_judged_prevalence"] = a["pr_auc_at_judged_prevalence"]
    c.ci("pair_roc_auc", a["roc_auc_ci"], a["ci_method"])
    c.ci("pair_pr_auc", a["pr_auc_ci"], a["ci_method"])
    n_possible = j.get("n_possible_pairs")
    c.result["metric_details"]["pairwise_auc"] = {
        **{k: a[k] for k in ("n_pairs", "n_positive", "positive_rate", "split")},
        "judged_population_prevalence": judged_prev,
        "population_positive_rate": (n_pos / n_possible) if n_possible else None,
        "unjudged_share": (j.get("n_unjudged_pairs", 0) / n_possible) if n_possible else None,
        "interpretation": ("Размечены только явные пары: Jaccard >= tau (похожи) или 0 (не похожи). "
                           "Доля положительных — среди размеченных, а не среди всех пар хостов.")}
    pr = precision_recall_at_k(c.emb, pairs, RETRIEVAL_KS, c.split, n_boot=c.n_boot, seed=c.seed)
    for k, e in pr["by_k"].items():
        if e["eligible_anchors"]:
            m[f"precision@{k}"], m[f"recall@{k}"] = e["precision"], e["recall"]
            m[f"precision_lift@{k}"] = e["precision_lift"]
            c.ci(f"precision@{k}", e["precision_ci"])
            c.ci(f"recall@{k}", e["recall_ci"])
    c.result["metric_details"]["retrieval"] = pr


def _triplets(c: Context) -> None:
    t = triplet_accuracy(c.emb, load_triplets(c.ref / "triplets.csv"), c.split, n_boot=c.n_boot, seed=c.seed)
    c.result["metrics"]["triplet_accuracy"] = t["accuracy"]
    c.ci("triplet_accuracy", t["accuracy_ci"], t["ci_method"])
    c.result["metric_details"]["triplet_accuracy"] = {k: t[k] for k in ("n_triplets", "n_ties", "chance", "split")}


def _probe(c: Context) -> None:
    source = c.protocol.get("labels_source")
    if not source:
        raise _Unavailable("у labels.csv не указан источник (labels_source в manifest.json)")
    pr = linear_probe(c.emb, load_labels(c.ref / "labels.csv"), labels_source=source,
                      eval_split="test" if c.split == "test" else c.split, n_boot=c.n_boot, seed=c.seed)
    for k in ("accuracy", "macro_f1", "balanced_accuracy"):
        c.result["metrics"][f"probe_{k}"] = pr[k]
    c.result["metrics"]["probe_majority_baseline_accuracy"] = pr["majority_baseline_accuracy"]
    c.ci("probe_accuracy", pr["accuracy_ci"])
    c.result["metric_details"]["linear_probe"] = {k: v for k, v in pr.items() if k != "accuracy_ci"}


class _Unavailable(Exception):
    pass


def _knn_chance(r: dict) -> float | None:
    n = r.get("n_hosts") or 0
    return 10 / (n - 1) if n > 1 else None


METRICS: list[Metric] = [
    Metric("knn_consistency", "Поведение", "Соседи@10", ("neighbors.csv", "features.csv"),
           _knn, "consistency@10_mean", _knn_chance),
    Metric("future_link_prediction", "Будущие связи", "Связь, косинус", ("edges.csv",),
           _links, "link_roc_auc", lambda r: 0.5, show=False),
    Metric("link_learned", "Будущие связи", "Связь, AUC", ("edges.csv",),
           _links_learned, "link_learned_roc_auc", lambda r: r["metrics"].get("link_learned_random_roc_auc")),
    Metric("partner_ranking", "Будущие связи", "Новые собеседники, MRR", ("partners.csv",),
           _partners, "partner_new_mrr", lambda r: max(filter(None, [r["metrics"].get("partner_new_history_mrr"),
                                                                     r["metrics"].get("partner_new_popularity_mrr")]),
                                                       default=None)),
    Metric("pairwise_roc_auc", "Общие собеседники", "Пары, AUC", ("pairs.csv",),
           _pairs, "pair_roc_auc", lambda r: 0.5, also=("pairwise_pr_auc", "precision_at_k", "recall_at_k")),
    Metric("triplet_accuracy", "Общие собеседники", "Тройки", ("triplets.csv",),
           _triplets, "triplet_accuracy", lambda r: 0.5, show=False),
    Metric("linear_probe", "Внешние метки", "Зонд, accuracy", ("labels.csv",),
           _probe, "probe_accuracy", lambda r: r["metrics"].get("probe_majority_baseline_accuracy")),
]
REQUIRED = tuple(k for m in METRICS for k in (m.key, *m.also))
UNAVAILABLE_REASON = {
    "knn_consistency": "нет neighbors.csv и features.csv",
    "linear_probe": "нет labels.csv (внешних меток хостов нет, придумывать их нельзя)",
}


def _load_matrix(path: Path, id_col: str, prefix: str) -> pd.DataFrame:
    return load_embeddings(path, id_col, prefix)


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def run(embeddings_path: Path, reference_dir: Path, method: str, split: str | None = None,
        n_boot: int = 0, seed: int = 0, fit_reference_dir: Path | None = None) -> dict:
    emb_full = load_embeddings(embeddings_path, "host_id", "emb_")
    emb = emb_full
    reference_dir = Path(reference_dir)
    eval_file = reference_dir / "eval_hosts.csv"
    if eval_file.exists():
        eval_hosts = pd.read_csv(eval_file, dtype={"host_id": str})["host_id"].tolist()
        absent = sorted(set(eval_hosts) - set(emb_full.index))
        if absent:
            raise ValueError(f"в эмбеддинге нет {len(absent)} оцениваемых хостов (например {absent[:5]}); "
                             "каждый метод должен покрыть всю когорту")
        emb = emb_full.loc[eval_hosts]
    manifest_file = reference_dir / "manifest.json"
    protocol = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists() else {}
    split = split or protocol.get("split", "test")
    result: dict = {
        "method": method, "submission": str(embeddings_path), "n_hosts": int(len(emb)), "dim": int(emb.shape[1]),
        "n_embedded_hosts": int(len(emb_full)),
        "split": split, "n_bootstrap": n_boot, "seed": seed,
        "metrics": {}, "metrics_ci": {}, "metric_details": {},
        "metric_status": {}, "unavailable_metrics": [],
    }
    result["reference_info"] = {
        "features_sha256": _sha(reference_dir / "features.csv"),
        "neighbors_sha256": _sha(reference_dir / "neighbors.csv"),
        "files_sha256": {f: _sha(reference_dir / f) for f in REFERENCE_FILES if (reference_dir / f).exists()},
        "kind": protocol.get("kind", "unspecified_reference"),
        "ground_truth": protocol.get("ground_truth"),
        "provenance_verified": protocol.get("provenance_verified", False),
        "train_end_utc": protocol.get("train_end_utc"), "test_start_utc": protocol.get("test_start_utc"),
        "data_kind": protocol.get("data_kind", "unspecified"),
        "synthetic": protocol.get("synthetic"),
        "real_lanl_declared": protocol.get("real_lanl_declared"),
        "provenance_verified_by_pipeline": protocol.get("provenance_verified", False),
        "cohort_policy": protocol.get("cohort_policy"),
        "n_cohort": protocol.get("n_cohort"), "n_evaluable": protocol.get("n_evaluable"),
        "evaluable_coverage": protocol.get("coverage"),
        "target_window": protocol.get("target_window"), "source_window": protocol.get("source_window"),
        "cohort_sha256": protocol.get("cohort_sha256"),
        "labels_source": protocol.get("labels_source"),
        "fit_reference_files_sha256": None,
    }
    ctx = Context(emb, emb_full, reference_dir, protocol, split, n_boot, seed,
                  Path(fit_reference_dir) if fit_reference_dir else None, result)
    summary = []
    for m in METRICS:
        keys = (m.key, *m.also)
        reason = None
        if not any((reference_dir / f).exists() for f in m.needs):
            reason = UNAVAILABLE_REASON.get(m.key, f"нет {' / '.join(m.needs)}")
        else:
            try:
                m.compute(ctx)
            except _Unavailable as exc:
                reason = str(exc)
        for k in keys:
            if reason:
                result["metric_status"][k] = f"unavailable: {reason}"
                result["unavailable_metrics"].append(f"{k}: {reason}")
            else:
                result["metric_status"][k] = "ok"
        if not reason and m.show:
            summary.append({"group": m.group, "metric": m.headline, "title": m.title,
                            "value": result["metrics"].get(m.headline),
                            "ci": result["metrics_ci"].get(m.headline), "chance": m.chance(result)})
    result["summary"] = summary
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценить эмбеддинг хостов по всем доступным метрикам")
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--split", default=None, help="по умолчанию — split из manifest.json эталона")
    parser.add_argument("--fit-reference-dir", type=Path, default=None,
                        help="эталон БОЛЕЕ РАННЕГО окна (reference/validation): на нём учится классификатор связей")
    parser.add_argument("--bootstrap", type=int, default=200, help="число бутстреп-повторов для 95%% интервалов (0 — выкл.)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run(args.embeddings, args.reference_dir, args.method, args.split, args.bootstrap, args.seed,
                 args.fit_reference_dir)
    text = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
