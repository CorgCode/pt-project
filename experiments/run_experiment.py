"""Все методы на ОДНОЙ когорте, окнах, эталонах и кандидатах.

    python -m experiments.run_experiment --processed-dir data/processed \
        --reference-dir data/reference --out-dir runs/lanl_smoke [--submissions-dir submissions]

Методы: бейзлайны (обучаются только на train), графовые методы из --methods, потолок reference_ceiling
(«утекает» намеренно) и любые <submissions-dir>/<метод>/embeddings.csv (+ <метод>/<окно>.csv для
узнаваемости во времени). Ожидавшиеся, но не полученные методы перечисляются в отчёте, а не симулируются.

Выход: results/<окно>/<метод>.json, identity/<окно>/<метод>.json, scoreboard_<окно>.md/.csv,
audit.json, report.md, figures/, manifest.json.
"""
from __future__ import annotations

import argparse
import json
import time
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
from evaluation.identity import identity_retrieval
from evaluation import scoreboard
from evaluation.io import load_embeddings
from experiments import audit as A
from experiments import report as R
from pipeline.data import ProcessedData
from pipeline.provenance import BADGE

TARGETS = ("validation", "test")
SCOPE = {
    "synthetic_fixture": ("Числа получены на сгенерированных данных в формате LANL NetFlow. Они показывают работу "
                          "конвейера и метрик и ничего не говорят о реальной сети."),
    "unverified": ("Происхождение входных файлов не установлено, поэтому ничего здесь не выдаётся за результат на LANL. "
                   "Чтобы пометить прогон как LANL, укажите источник: `--data-kind lanl --source-note ...`."),
    "lanl_declared": ("Выводы относятся к заявленным файлам LANL NetFlow при протоколе из этого отчёта "
                      "(окна, когорта, эталоны). Это не утверждения о других данных, других окнах или ролях хостов."),
}
TAGS = {"synthetic_fixture": "СИНТЕТИКА", "unverified": "НЕПРОВЕРЕННЫЕ ДАННЫЕ", "lanl_declared": "LANL (заявлено)"}
BADGE_RU = {
    "synthetic_fixture": "СИНТЕТИКА (synthetic_fixture) — не трафик LANL",
    "unverified": "НЕПРОВЕРЕННЫЕ ДАННЫЕ (unverified) — происхождение не установлено, за LANL не выдаётся",
    "lanl_declared": "LANL (заявлено оператором; происхождение пайплайном не проверено)",
}
EXPECTED_EXTERNAL = ["node2vec", "deepwalk", "graphsage", "struc2vec", "rolx"]   # #20 / план ресёрча


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def _read_training_windows(submission_dir: Path) -> list[str] | None:
    """Окна, на которых, по заявлению внешнего метода, он обучался (provenance.json); None — не заявлено."""
    f = Path(submission_dir) / "provenance.json"
    if not f.exists():
        return None
    try:
        w = json.loads(f.read_text(encoding="utf-8")).get("trained_on_windows")
    except json.JSONDecodeError as exc:
        raise ValueError(f"{f}: некорректный JSON") from exc
    if not isinstance(w, list) or not all(isinstance(x, str) for x in w):
        raise ValueError(f'{f}: needs {{"trained_on_windows": ["train", ...]}}')
    return w


def _log(msg: str, t0: float | None = None) -> None:
    """Строка прогресса в stdout (видна в логе и в status.sh)."""
    extra = f" ({time.perf_counter() - t0:.0f} с)" if t0 is not None else ""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}{extra}", flush=True)


def _write(path: Path, df: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return path


def run_experiment(processed_dir: Path, reference_dir: Path, out_dir: Path, *, dim: int = 16, seed: int = 0,
                   n_boot: int = 200, submissions_dir: Path | None = None, do_audit: bool = True,
                   audit_configs: int = 12, null_draws: int = 30, methods: list[str] | None = None,
                   method_config: dict | None = None, fast_methods: bool = False,
                   port_free_baselines: bool = False, reuse_embeddings: bool = False) -> dict:
    processed_dir, reference_dir, out_dir = Path(processed_dir), Path(reference_dir), Path(out_dir)
    data = ProcessedData(processed_dir)
    targets = [t for t in TARGETS if (reference_dir / t).exists() and t in data.windows]
    if "test" not in targets:
        raise ValueError(f"{reference_dir}/test не найден; сначала python -m pipeline.reference")
    ref_manifest = json.loads((reference_dir / "test" / "manifest.json").read_text())
    if ref_manifest.get("cohort_sha256") != data.manifest["cohort"]["sha256"]:
        raise ValueError("эталон построен по другой когорте, чем подготовленные данные")
    for t in targets:                       # сохранённый эталон должен покрывать ровно оцениваемых хостов
        saved = pd.read_csv(reference_dir / t / "eval_hosts.csv", dtype=str)["host_id"].tolist()
        if saved != data.evaluable(t):
            raise ValueError(f"reference/{t}/eval_hosts.csv не совпадает с оцениваемыми хостами датасета")
    hosts_by_target = {t: data.evaluable(t) for t in targets}

    # ---- собрать эмбеддинги: метод -> {окно: DataFrame}; 'train' — оцениваемый эмбеддинг
    emb: dict[str, dict[str, pd.DataFrame]] = {}
    meta: dict[str, dict] = {}
    graph_methods_run = graph_methods.create(methods or [], dim, seed, method_config, fast_methods)
    todo = [*default_baselines(dim, seed, port_free_baselines), *graph_methods_run]
    for i, b in enumerate(todo, 1):
        t0 = time.perf_counter()
        saved = out_dir / "embeddings" / b.name
        files = {w: saved / f"{w}.csv" for w in ("train", *targets)}
        if reuse_embeddings and all(f.exists() for f in files.values()) and (saved / "info.json").exists():
            emb[b.name] = {w: load_embeddings(f) for w, f in files.items()}
            meta[b.name] = json.loads((saved / "info.json").read_text(encoding="utf-8"))
            _log(f"взято с диска {i}/{len(todo)}: {b.name}")
            continue
        _log(f"обучение {i}/{len(todo)}: {b.name}")
        b.fit(data)
        emb[b.name] = {"train": b.embed(data, "train"),
                       **{t: b.embed(data, t, hosts_by_target[t]) for t in targets}}
        meta[b.name] = b.info()
        for w, f in files.items():                      # сразу на диск: падение на оценке не стоит переобучения
            _write(f, emb[b.name][w])
        (saved / "info.json").write_text(json.dumps(meta[b.name], indent=2, sort_keys=True, default=str) + "\n",
                                         encoding="utf-8")
        _log(f"готово: {b.name}", t0)
    external, external_provenance = [], {}
    if submissions_dir and Path(submissions_dir).exists():
        for d in sorted(p for p in Path(submissions_dir).iterdir() if not p.name.startswith(".") and (p / "embeddings.csv").exists()):
            declared = _read_training_windows(d)
            if declared is not None and set(declared) - {"train"}:
                raise ValueError(f"отправка {d.name!r} обучалась на {sorted(set(declared) - {'train'})}; "
                                 "учиться можно только на train-окне (утечка по времени)")
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

    # ---- оценка на каждом целевом эталоне
    results: dict[str, dict[str, dict]] = {t: {} for t in targets}
    identity: dict[str, dict[str, dict]] = {t: {} for t in targets}
    for t in targets:
        ref_t = reference_dir / t
        feats = load_embeddings(ref_t / "features.csv", "host_id", "ref_")
        per_target = dict(emb)
        per_target["reference_ceiling"] = {"train": feats.rename(columns=lambda c: c.replace("ref_", "emb_"))}
        for j, (name, windows) in enumerate(per_target.items(), 1):
            t0 = time.perf_counter()
            path = _write(out_dir / "embeddings" / name / "train.csv", windows["train"])
            # потолок встраивает только оцениваемых хостов целевого окна, поэтому в обученном классификаторе
            # связей (его кандидаты — из более раннего validation) участвовать не может; он и так «утекает»
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
            _log(f"оценка на {t} {j}/{len(per_target)}: {name}", t0)
        scoreboard.build(out_dir / "results" / t, out_dir / f"scoreboard_{t}")

    # ---- аудит
    audit = None
    if do_audit:
        t0 = time.perf_counter()
        _log(f"аудит: {audit_configs} вариантов весов эталона, {null_draws} случайных эмбеддингов")
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
        _log("аудит готов", t0)
        import shutil
        shutil.rmtree(out_dir / "_null", ignore_errors=True)

    # ---- манифест, графики, отчёт
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
    R.knn_by_k(results["test"], fig_dir / "knn_by_k.png", f"{tag} — совпадение соседей при разных k (test)")
    R.identity_plot(identity["test"], fig_dir / "identity_test.png", f"{tag} — узнаваемость хоста (train -> test)")
    if audit:
        R.weights_plot(audit["weight_sensitivity"], fig_dir / "weight_sensitivity.png")
    (out_dir / "report.md").write_text(build_report(manifest, results, identity, audit), encoding="utf-8")
    return manifest


def build_report(manifest: dict, results: dict, identity: dict, audit: dict | None) -> str:
    kind = manifest["data_kind"]
    L = ["# Отчёт о прогоне\n", f"> **{BADGE_RU[kind]}.**\n>\n> " + SCOPE[kind] + "\n"]
    prov = manifest.get("provenance") or {}
    if kind == "lanl_declared":
        L.append(f"* источник, заявленный оператором: *{prov.get('source_note')}*. Пайплайн проверить это не может "
                 "(`provenance_verified: false`).")
    c = manifest["config"]
    L.append(f"* данные: `{kind}` (основание: {prov.get('basis')}); хостов в когорте: **{manifest['n_hosts']}**; "
             f"dim={c['dim']}, seed={c['seed']}, bootstrap={c['bootstrap']}; окна оценки: {', '.join(c['targets'])}")
    L.append(f"* оценены методы: {', '.join(manifest['methods'])}")
    L.append(f"* **в этом прогоне нет** (эмбеддинги не переданы, ничего не симулируется): "
             f"{', '.join(manifest['missing_expected_methods']) or 'нет'}")
    coh = manifest.get("cohort")
    if coh:
        cov = ", ".join(f"{t}: {coh['n_evaluable'][t]}/{coh['n_cohort']} ({coh['coverage'][t]:.0%})" for t in coh["n_evaluable"])
        L.append(f"* политика когорты `{coh['policy']}`; оцениваемых хостов по окнам: {cov}")
        if coh.get("selection"):
            L.append(f"  * {coh['selection']['note']}")
    for name, pv in (manifest.get("external_provenance") or {}).items():
        L.append(f"* внешний метод `{name}`: обучен на {pv['trained_on_windows']} (заявлено, не проверено)"
                 if pv.get("trained_on_windows") is not None else
                 f"* внешний метод `{name}`: **окна обучения не заявлены** — утечку исключить нельзя")
    L.append("")
    cols = ["consistency@10_mean", "ndcg@10_mean", "link_roc_auc", "link_learned_roc_auc", "link_new_roc_auc",
            "partner_new_mrr", "pair_roc_auc", "triplet_accuracy", "precision@3", "recall@3"]
    for t in c["targets"]:
        rows = {m: {k: r["metrics"].get(k) for k in cols} for m, r in results[t].items()}
        df = pd.DataFrame(rows).T
        df.index.name = "method"
        L.append(f"\n## Результаты — эталон {t}\n")
        L.append(R.md_table(df))
        pers = next(iter(results[t].values()))["metric_details"].get("future_link_prediction", {}).get("persistence_baseline")
        if pers:
            L.append(f"\nТочка отсчёта для связей без эмбеддинга — *пара уже общалась в source-окне*: ROC-AUC "
                     f"**{pers['roc_auc']:.3f}**. `link_learned_*` — классификатор рёбер, обученный на кандидатах "
                     f"validation и оценённый на test; косинус показан для полноты.")
        L.append(f"\n![scorecard]({'figures/scorecard_' + t + '.png'})\n")
    unavail = sorted({u for r in results["test"].values() for u in r["unavailable_metrics"]})
    L.append("\n## Недоступные метрики\n")
    L.extend(f"* {u}" for u in unavail)
    ci_rows = []
    for m, r in results["test"].items():
        for k in ("consistency@10_mean", "link_roc_auc", "pair_roc_auc", "triplet_accuracy"):
            ci = r["metrics_ci"].get(k)
            if ci:
                ci_rows.append({"method": m, "metric": k, "estimate": r["metrics"][k], "ci_low": ci[0], "ci_high": ci[1]})
    if ci_rows:
        L.append("\n## 95% бутстреп-интервалы (test)\n")
        L.append(R.md_table(pd.DataFrame(ci_rows).set_index("method")))
    L.append("\n## Узнаваемость хоста во времени (train -> test)\n")
    rows = {}
    for m, v in identity["test"].items():
        rows[m] = {"общие координаты заявлены": str(v["shared_coordinate_system_declared"]),
                   "Top-1 сырые": v.get("none", {}).get("top1"), "Top-1 Procrustes": v.get("procrustes", {}).get("top1"),
                   "MRR Procrustes": v.get("procrustes", {}).get("mrr"),
                   "ROC-AUC пар Procrustes": v.get("procrustes", {}).get("pair_roc_auc"),
                   "случайный Top-1": v.get("procrustes", {}).get("chance_top1")}
    if rows:
        L.append(R.md_table(pd.DataFrame(rows).T))
        L.append("\n![identity](figures/identity_test.png)\n")
        L.append("Сырые координаты осмысленны только для методов, заявивших общую систему координат.")
    if audit:
        w, s = audit["weight_sensitivity"], audit["reference_stability"]
        L.append("\n## Аудит валидности\n")
        L.append(f"* **Чувствительность к весам** ({len(w['configs'])} изменённых поведенческих эталонов, k={w['k']}): "
                 f"совпадение соседей с основным эталоном в среднем {w['mean_reference_agreement']:.2f} "
                 f"(минимум {w['min_reference_agreement']:.2f}; случайно {w['chance_agreement']:.3f}); "
                 f"тау Кендалла порядка методов в среднем {w['mean_kendall_tau']:.2f}, минимум {w['min_kendall_tau']:.2f}.")
        for pair, v in s["pairs"].items():
            L.append(f"* **Стабильность эталона** {pair}: Consistency@{s['k']} = {v['consistency']:.2f} "
                     f"(случайно {s['chance_consistency']:.3f}) — поведенческое соседство дрейфует между окнами.")
        if audit["split_agreement"]:
            L.append("* **Согласие порядка validation и test** (тау Кендалла): " +
                     ", ".join(f"{k}: {v:.2f}" for k, v in audit["split_agreement"].items() if v is not None))
        nm = audit["random_null"]["metrics"]
        L.append("* **Случайный уровень (случайные эмбеддинги, test)**: " + ", ".join(
            f"{k} {nm[k]['mean']:.3f} ± {nm[k]['std']:.3f}" for k in
            ("consistency@10_mean", "link_roc_auc", "link_learned_roc_auc", "pair_roc_auc", "triplet_accuracy") if k in nm))
        pv = audit["prevalence"]["future_links"]
        L.append(f"* **Доля связей среди кандидатов**: в выборке {pv['sampled_positive_rate']:.2f}, на самом деле "
                 f"{pv['natural_prevalence']:.3f}; {pv['share_positives_seen_in_source']:.0%} будущих связей уже были "
                 f"в source-окне (см. ROC-AUC только новых связей).")
        L.append("\n![weights](figures/weight_sensitivity.png)\n")
        L.append("\n### Что метод делит с эталоном по построению\n")
        ov = pd.DataFrame(audit["information_overlap"]).T
        ov.index.name = "method"
        L.append(R.md_table(ov))
    L.append("\n## Как читать эти числа\n")
    L.append("* **Прогноз:** будущие связи (результат виден позже) и узнаваемость. "
             "**Согласие с выбранным нами приближением:** kNN (поведенческий профиль), пары/тройки/P@k (общие собеседники).")
    L.append("* `reference_ceiling` — признаки самого целевого окна, «утекает» намеренно; бейзлайны делят с эталонами "
             "признаки или структуру собеседников (таблица выше), поэтому это точки калибровки, а не независимое доказательство.")
    L.append("* Интервалы: для kNN и Precision@k пересэмплируются хосты; для связей, пар и троек — бутстреп по хостам "
             "(пары делят хостов). Они описывают только выбор оцениваемых элементов, а не seed модели, эталон или окна.")
    L.append("* " + {"synthetic_fixture": "Данные синтетические, поэтому о LANL ни одно число выше ничего не говорит.",
                    "unverified": "Происхождение не проверено — считайте это результатом на неразмеченном датасете.",
                    "lanl_declared": "Результаты верны только для заявленных файлов; какие хосты они описывают, "
                                     "решает политика когорты (см. выше)."}[manifest["data_kind"]])
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
    ap.add_argument("--reuse-embeddings", action="store_true",
                    help="не обучать заново методы, чьи эмбеддинги всех окон уже лежат в <out-dir>/embeddings")
    ap.add_argument("--no-audit", action="store_true")
    ap.add_argument("--audit-configs", type=int, default=12)
    ap.add_argument("--null-draws", type=int, default=30)
    a = ap.parse_args()
    m = run_experiment(a.processed_dir, a.reference_dir, a.out_dir, dim=a.dim, seed=a.seed, n_boot=a.bootstrap,
                       submissions_dir=a.submissions_dir, do_audit=not a.no_audit,
                       audit_configs=a.audit_configs, null_draws=a.null_draws, methods=a.methods,
                       method_config=_load_config(a.method_config), fast_methods=a.fast_methods, port_free_baselines=a.port_free_baselines,
                       reuse_embeddings=a.reuse_embeddings)
    print(json.dumps({k: m[k] for k in ("data_kind", "methods", "missing_expected_methods")}, indent=2))
    print(f"report: {a.out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
