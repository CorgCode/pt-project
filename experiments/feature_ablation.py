"""Абляция блоков признаков поведенческого бейзлайна; главное — вариант без портов (#43).

    python -m experiments.feature_ablation --processed-dir data/processed --reference-dir data/reference \
        --out-dir runs/ablation_ports

Варианты — поведенческий профиль train-окна (volume, peers, ratio, ports, proto, hourly) с полностью
убранными блоками: full, no_ports и «минус один блок» для остальных. Все на ОДНИХ хостах, окнах и эталонах,
оценка — общая (evaluation.evaluate.run), хеши файлов записываются. Для full против no_ports — парные
разности с интервалами (хосты пересэмплируются вместе для обоих вариантов).

Отдельная ДИАГНОСТИКА (не для таблицы результатов): эталон соседей, пересобранный без портов. Показывает,
сколько поведенческого эталона держится на портах. У него другой отпечаток — смешивать с таблицей нельзя.
Всё обучается только на train.
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
    """a минус b на стандартном эталоне, с парными бутстреп-интервалами."""
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
    """Совпадение соседей с эталоном целевого окна, пересобранным БЕЗ портов (только анализ)."""
    test_flows = load_flows(data.root / "flows" / "test.csv")
    hosts = pd.read_csv(reference_dir / "eval_hosts.csv", dtype=str)["host_id"].tolist() \
        if (reference_dir / "eval_hosts.csv").exists() else data.cohort
    feats_full = build_reference_features(test_flows, hosts=hosts)
    feats_free = build_reference_features(test_flows, hosts=hosts, exclude_blocks=("ports",))
    self_agree = evaluate_embedding(feats_free, feats_full, ks=KS)          # соседи эталона без портов против полного
    out: dict = {
        "note": ("эталон пересобран по тому же целевому окну БЕЗ блока портов; другой отпечаток, "
                 "с таблицей результатов НЕ сравнивается"),
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
        raise ValueError(f"{ref} не найден; сначала python -m pipeline.reference")
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
    if not identical:                                                       # невозможно, пока run() не изменится
        raise RuntimeError("варианты оценены на разных эталонах или хостах")
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


SCOPE = {"synthetic_fixture": "Синтетика: у ролей генератора свои порты ПО ПОСТРОЕНИЮ, поэтому порты здесь информативны; "
                              "для LANL отсюда ничего не следует.",
         "unverified": "Происхождение данных не установлено; ничего здесь не выдаётся за результат на LANL.",
         "lanl_declared": "Выводы относятся только к заявленным файлам LANL при этом протоколе."}
BADGE_RU = {"synthetic_fixture": "СИНТЕТИКА (synthetic_fixture) — не трафик LANL",
            "unverified": "НЕПРОВЕРЕННЫЕ ДАННЫЕ (unverified)",
            "lanl_declared": "LANL (заявлено оператором; происхождение не проверено)"}


def _f(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def _diff_cell(rec: dict | None) -> str:
    """'+0.012 [+0.003, +0.021] *'; без бутстрепа (n_bootstrap = 0) интервал не пишется."""
    if rec is None:
        return "нет пары"
    if rec["ci95"] is None:
        return f"{rec['difference']:+.3f} (без интервала: бутстреп выключен)"
    return f"{rec['difference']:+.3f} [{rec['ci95'][0]:+.3f}, {rec['ci95'][1]:+.3f}]" + (" *" if rec["ci_excludes_zero"] else "")


def render(rep: dict) -> str:
    L = ["# Поведенческий бейзлайн без портов: абляция признаков\n",
         f"> **{BADGE_RU.get(rep['data_kind'], rep['badge'])}.** {SCOPE[rep['data_kind']]}\n"]
    pr = rep["protocol"]
    L.append(f"Одинаково для всех вариантов (проверено по хешам файлов): split `{pr['split']}`, оцениваемых хостов "
             f"{pr['n_hosts_scored']}, одни и те же замороженные эталоны, обучение только на окне `{pr['fit_window']}`, "
             f"политика когорты `{pr['cohort_policy']}`.\n")
    L.append("## Стандартный эталон: полный профиль против профиля без портов\n")
    L.append("| метрика | полный | без портов | полный − без портов (парный 95% CI) |\n|---|---|---|---|")
    keymap = {"consistency@10_mean": "consistency@10", "link_roc_auc": "link_roc_auc", "pair_roc_auc": "pair_roc_auc",
              "triplet_accuracy": "triplet_accuracy"}
    for k, label in HEADLINE:
        a, b = rep["metrics"]["full"][k], rep["metrics"]["no_ports"][k]
        L.append(f"| {label} | {_f(a)} | {_f(b)} | {_diff_cell(rep['paired_full_minus_no_ports'].get(keymap.get(k, ''), None))} |")
    L.append("\n`*` — парный интервал не содержит 0. Хосты пересэмплируются одновременно для обоих вариантов (для пар, связей "
             "и троек — бутстреп по хостам); интервалы описывают только выбор оцениваемых элементов.\n")
    L.append("## Минус один блок (тот же протокол)\n")
    L.append("| вариант (убранные блоки) | " + " | ".join(l for _, l in HEADLINE) + " |\n|---|" + "---|" * len(HEADLINE))
    for n, ex in pr["variants"].items():
        L.append(f"| {n} ({', '.join(ex) or 'нет'}) | " + " | ".join(_f(rep["metrics"][n][k]) for k, _ in HEADLINE) + " |")
    dg = rep["port_free_reference_diagnostic"]
    L.append("\n## Диагностика: эталон, пересобранный БЕЗ портов (только анализ)\n")
    L.append(f"_{dg['note']}_\n")
    L.append("| kNN Consistency@k | полный | без портов | случайно |\n|---|---|---|---|")
    for k in KS:
        L.append(f"| @{k} | {_f(dg['full'][f'consistency@{k}'])} | {_f(dg['no_ports'][f'consistency@{k}'])} | {_f(dg['chance'][f'consistency@{k}'])} |")
    p = dg["paired_full_minus_no_ports@10"]
    L.append(f"\nПарная разность «полный − без портов» при k=10: {_diff_cell(p)}. Совпадение соседей эталона без портов "
             f"с соседями стандартного: Consistency@10 = {_f(dg['reference_self_agreement']['consistency@10'])} "
             f"(случайно {_f(dg['chance']['consistency@10'])}).\n")
    L.append("## Что это значит для методологии\n")
    std = rep["paired_full_minus_no_ports"].get("consistency@10")
    L.append(f"1. **Стандартный поведенческий эталон содержит блок портов** (около {rep['port_block_share_of_profile_weight']:.0%} "
             "квадрата веса профиля). Поэтому полный бейзлайн делит эту информацию с эталоном **по построению**; вариант без портов "
             "это пересечение убирает. Разница на kNN — это пересечение информации, а не качество представления"
             + (f" (здесь парная разность kNN@10 = {std['difference']:+.3f})." if std else "."))
    L.append("2. **Порты есть только в kNN-эталоне.** Эталоны пар и троек по общим собеседникам и кандидаты будущих связей строятся "
             "из собеседников, поэтому абляция портов на них показывает, сколько *идентичность сервиса* добавляет к структурному "
             "и прогнозному сигналу, а не пересечение с эталоном.")
    L.append("3. **Диагностика без портов показывает остаток.** Если разность «полный − без портов» исчезает, когда порты убраны "
             "и из эталона, то разность на стандартном эталоне была пересечением по портам; если остаётся — её дают другие блоки. "
             "Диагностика не является эталоном таблицы результатов, смешивать их нельзя.")
    L.append("4. **Что такое порты в LANL NetFlow.** Известные порты сохранены, остальные обезличены в стабильные `PortNNNN`, "
             "поэтому блок портов отчасти работает как идентификатор сервиса. Профиль без портов проверяет, опирается ли метод на "
             "сервис или на форму трафика (объём, направление, собеседники, протоколы, время суток).")
    L.append("5. **Таблица результатов не меняется.** Стандартные эталоны, хосты и окна те же; варианты без портов — дополнительные "
             "строки (`--port-free-baselines`). Каждый блок нормируется отдельно, поэтому оставшиеся колонки совпадают с полным профилем.")
    L.append("6. **Ограничения.** Одна пара окон, одна когорта, нет меток ролей; абляция говорит, какую информацию поощряет "
             "*приближение*, а не какая информация определяет роли.")
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
