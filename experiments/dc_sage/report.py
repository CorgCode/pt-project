"""Сводка пяти сидов, сравнение на общем day02 и экспорт компактных результатов."""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
import scipy
import sklearn
import torch

from evaluation import scoreboard
from experiments.dc_sage.data import sha
from experiments.dc_sage.run import ROOT, dump, variants
from methods.graph._dc_sage import Config

KEYS = ["consistency@10_mean", "link_learned_roc_auc", "partner_new_mrr", "pair_roc_auc", "identity_top1", "triplet_accuracy"]


def interval(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return [float(values[0]), float(values[0])]
    half = t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))
    return [float(values.mean()-half), float(values.mean()+half)]


def score(result):
    cols = {s["metric"]: s["title"] for s in result["summary"]}
    return float(scoreboard.overall_score([result], cols).iloc[0])


def aggregate(name, rows):
    result = copy.deepcopy(rows[0])
    result["method"] = name
    keys = set.intersection(*(set(r["metrics"]) for r in rows))
    result["metrics"] = {key: float(np.mean([r["metrics"][key] for r in rows])) for key in keys
                         if all(isinstance(r["metrics"][key], (float, int)) for r in rows)}
    result["metrics_ci"] = {key: interval([r["metrics"][key] for r in rows]) for key in result["metrics"]}
    result["metrics_std_over_seeds"] = {key: float(np.std([r["metrics"][key] for r in rows], ddof=1)) for key in result["metrics"]}
    for s in result["summary"]:
        k = s["metric"]
        s["value"], s["ci"] = result["metrics"][k], result["metrics_ci"][k]
        chances = [next(x["chance"] for x in r["summary"] if x["metric"] == k) for r in rows]
        s["chance"] = float(np.mean(chances)) if all(v is not None for v in chances) else None
    result["score_by_seed"] = [score(r) for r in rows]
    result["score_mean"] = float(np.mean(result["score_by_seed"]))
    result["score_std"] = float(np.std(result["score_by_seed"], ddof=1))
    result["ci_kind"] = "95% Student-t interval across 5 seeds; host-bootstrap CIs remain in individual JSON files"
    result["n_bootstrap"] = None
    return result


def existing_day02():
    output = []
    for path in sorted((ROOT / "scoreboard/results").glob("*.json")):
        r = json.loads(path.read_text())
        if "by_window" in r:
            w = r["by_window"]["test"]
            r["metrics"], r["summary"], r["metrics_ci"] = w["metrics"], w["summary"], w["metrics_ci"]
            r.pop("by_window")
            r.pop("test_windows", None)
        output.append(r)
    return output


def criteria(groups, baseline):
    def tasks(rows):
        return [scoreboard.task_scores(r, {s["metric"]: s["title"] for s in r["summary"]}) for r in rows]
    dc, skip = tasks(groups["dc_sage"]), tasks(groups["skip_only"])
    names = list(dc[0])
    out = {}
    for task in names:
        a, b = [r[task] for r in dc], [r[task] for r in skip]
        ia, ib = interval(a), interval(b)
        out[task] = {"dc_mean": float(np.mean(a)), "skip_mean": float(np.mean(b)), "dc_ci": ia, "skip_ci": ib,
                     "paired_difference_ci": interval(np.asarray(a)-np.asarray(b)),
                     "dc_detectably_worse": ia[1] < ib[0], "dc_detectably_better": ia[0] > ib[1]}
    neighbor_delta = np.mean([r["metrics"]["consistency@10_mean"] for r in groups["dc_sage"]])-np.mean([r["metrics"]["consistency@10_mean"] for r in groups["skip_only"]])
    checks = {
        "1_no_task_with_disjoint_adverse_seed_CI": not any(v["dc_detectably_worse"] for v in out.values()),
        "2_future_or_shared_task_has_disjoint_favorable_seed_CI": any(v["dc_detectably_better"] for k,v in out.items() if k in ("Будущие связи", "Общие собеседники")),
        "3_neighbors_drop_at_most_0.02": bool(neighbor_delta >= -.02),
        "4_score_at_least_linear": bool(np.mean([score(r) for r in groups["dc_sage"]]) >= score(baseline)),
    }
    return {"tasks": out, "checks": checks, "all_passed": all(checks.values()), "neighbors_delta": float(neighbor_delta),
            "interpretation": "suggested instruction criteria; task CIs here measure 5-seed variation, not host sampling"}


def table(groups, baselines):
    header = "| метод | score | Neighbors@10 | Links AUC | New partners MRR | Pairs AUC | Identity Top-1 | Triplet accuracy |"
    lines = [header, "|---|---|---|---|---|---|---|---|"]
    for name, rows in groups.items():
        vals = [[score(r) for r in rows], *[[r["metrics"][k] for r in rows] for k in KEYS]]
        cells = [f"{np.mean(a):.3f} ± {np.std(a, ddof=1):.3f}" for a in vals]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    for r in baselines:
        lines.append(f"| {r['method']} | {score(r):.3f} | " + " | ".join(f"{r['metrics'][k]:.3f}" for k in KEYS) + " |")
    return "\n".join(lines) + "\n"


def figures(work, out, groups, baselines):
    os.environ.setdefault("MPLCONFIGDIR", str(work / "matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    order = list(groups)
    values = [[score(r) for r in groups[name]] for name in order]
    ax = axes[0, 0]
    ax.barh(order, [np.mean(v) for v in values], xerr=[np.std(v, ddof=1) for v in values], color="#4682a9")
    ax.invert_yaxis()
    ax.set(xlabel="Official score (mean ± SD, 5 seeds)", title="Test: graph and ablations")
    ax = axes[0, 1]
    for name in ("dc_sage", "skip_only", "no_contrastive"):
        histories = [json.loads((work / "models" / name / f"seed{s}/history.json").read_text()) for s in range(5)]
        cos = np.array([[r["mean_cosine"] for r in h] for h in histories])
        ax.plot(np.arange(1, cos.shape[1]+1), cos.mean(0), label=name)
    ax.set(xlabel="Epoch", ylabel="Mean pairwise cosine", title="Collapse diagnostic on training slices", ylim=(0, 1))
    ax.legend()
    task_names = ["Поведение", "Будущие связи", "Общие собеседники", "Узнаваемость"]
    ax = axes[1, 0]
    for i, name in enumerate(("dc_sage", "skip_only")):
        scores = [scoreboard.task_scores(r, {s["metric"]:s["title"] for s in r["summary"]}) for r in groups[name]]
        y = [np.mean([r[k] for r in scores]) for k in task_names]
        ax.bar(np.arange(4)+(i-.5)*.3, y, width=.3, label=name)
    ax.set(xticks=np.arange(4), xticklabels=["Behavior", "Future links", "Shared peers", "Identity"], ylabel="Task score", title="Where the graph helps")
    ax.legend()
    ax = axes[1, 1]
    rows = json.loads((out / "anomalies.json").read_text())
    scenarios = ["role_change", "scanner", "benign_x2"]
    y = [[r["embedding_drift"]["roc_auc"] for r in rows if r["scenario"] == key] for key in scenarios]
    ax.bar(scenarios, [np.mean(v) for v in y], yerr=[np.std(v, ddof=1) for v in y], color=["#4682a9", "#4682a9", "#c98b50"])
    ax.axhline(.5, linestyle="--", color="#444", linewidth=1)
    ax.set(ylabel="Drift ROC-AUC", title="Synthetic changes; benign should be near 0.5", ylim=(0, 1))
    fig.suptitle("DC-SAGE • LANL day02 • fixed train / validation / test", fontsize=15)
    fig.savefig(out / "summary.png", dpi=160)
    plt.close(fig)


def readme(work, out, groups, baselines, published, checks):
    def ms(values, digits=3):
        return f"{np.mean(values):.{digits}f} ± {np.std(values, ddof=1):.{digits}f}"
    primary = groups["dc_sage"]
    dc_score = ms([score(r) for r in primary], 2)
    skip_score = ms([score(r) for r in groups["skip_only"]], 2)
    linear = next(r for r in baselines if r["method"] == "linear")
    leader = max((r for r in published if r["method"] not in scoreboard.SERVICE), key=score)
    inductive = json.loads((out / "new_hosts.json").read_text())
    baseline_new = json.loads((work / "inductive/profile_baseline.json").read_text())
    anomalies = json.loads((out / "anomalies.json").read_text())
    freeze = json.loads((work / "evaluation_freeze.json").read_text())
    final = "Все четыре предложенных критерия выполнены." if checks["all_passed"] else "Все четыре предложенных критерия одновременно НЕ выполнены. Существенное преимущество полного графового рецепта этим экспериментом не установлено."
    lines = [
        "# DC-SAGE: результат обучения на LANL day02",
        "",
        f"Полный метод: **{dc_score} балла** по пяти сидам. MLP без графа: **{skip_score}**. "
        f"Простой линейный агрегат: **{score(linear):.2f}**. "
        f"Лучший опубликованный результат на этом же day02 — `{leader['method']}`: **{score(leader):.2f}**.",
        "",
        final,
        "",
        "Это day02 score, не среднее за три дня. Дни 03 и 04 отсутствуют локально; пользователь подтвердил это. "
        "Полная многодневная оценка остаётся невыполненной до появления потоков. Общая таблица репозитория не изменена.",
        "",
        "## Что реализовано",
        "",
        "Собственная реализация спецификации `experiments/dc_sage/Instruction.md`: directed GraphSAGE, "
        "два слоя по 128 скрытых координат, отдельные входящие/исходящие сообщения, IDF соседей, top64, "
        "остаточные связи, skip-MLP и нулевая инициализация графового выхода. "
        "95 входных признаков = 64 TF-IDF/SVD координаты сервисов с ролями + 31 поведенческий признак; выход 64.",
        "",
        "Сеть обучена InfoNCE по двум половинам train и асимметричным декодером связей S1→S2. "
        "Новые позитивные рёбра составляют половину обучающего набора. Негативы не содержат self-связей и настоящих рёбер S2. "
        "Есть dropout, AdamW/cosine schedule, gradient clip, контроль коллапса и величины графовой ветви.",
        "",
        "Вопреки утверждению спецификации, skip-связь не гарантирует нижнюю границу качества: "
        "нулевая графовая ветвь гарантирует только начальное равенство выхода MLP. "
        "После совместного обучения качество может стать хуже, что и проверяют сравнения.",
        "",
        "## Данные и выбор параметров",
        "",
        "Из локального широкого train восстановлены текущие окна общей оценки: "
        "train `[126000,129600)` (9 368 480 потоков), validation `[129600,131400)` (4 526 585), "
        "test `[131400,135000)` (10 297 984). Test сейчас длится час, а не 30 минут из исторического примера инструкции. "
        "Когорта — 6453; оценка test — 5445. Порог и списки хостов взяты из текущего `scoreboard/data`, не из текста примера.",
        "",
        "Числа потоков и события каждого хоста по каждому окну точно совпали с общим processed-манифестом. "
        "Потоки, пересекающие границы, исключены. S1 содержит 3 998 806 потоков, S2 — 5 144 117. "
        "6264 хоста достаточно активны в обоих срезах для InfoNCE; граф содержит 26 437 узлов. "
        "Профиль хранит 49 057 токенов, включая обезличенные категории; min_df=2.",
        "",
        "Все обучаемые преобразования подогнаны только на полном train. Локальные признаки S1/S2 считают только свои потоки. "
        "Рассмотрены три веса link loss (0.1, 0.3, 1.0), выбран 1.0 по официальному validation score на seed0. "
        "Остальные параметры — стартовые из инструкции. По validation выбиралась эпоха 25 или 50. "
        "Затем обучены 9 вариантов × 5 сидов. Все checkpoints зафиксированы до первого расчёта метрик test. "
        "Ни одна настройка не исправлялась ради улучшения увиденного test.",
        "",
        "Одна эпоха здесь — один полнографный шаг с усреднением всех контрастных батчей по 512. "
        "Всего 50 обновлений AdamW; это не эквивалент 50 эпохам mini-batch SGD с отдельным обновлением после каждого батча. "
        "Пять сидов меняют инициализацию сети, dropout, батчи и выборку рёбер; SVD фиксирован seed0.",
        "",
        "## Результаты и абляции",
        "",
        "Числа ниже — среднее ± выборочное стандартное отклонение пяти сидов. "
        "У простых бейзлайнов одно детерминированное обучение. Все новые варианты имеют 64 координаты "
        "(поведенческий профиль из 31 признака дополнен нулями); `dc_sage32` — отдельная проверка 32 координат.",
        "",
        table(groups, baselines),
        "Оценка использует неизменённые `evaluation.evaluate.run` и `evaluation.submit.add_identity`; "
        "включает Triplet accuracy и дополнительные метрики Тихона. Классификатор связей обучается на validation, "
        "а подаваемые ему векторы в обоих окнах — из train. Identity посчитана штатным Procrustes на половине хостов "
        "(2722 кандидата/запроса на test). Разметки ролей нет, linear probe недоступен.",
        "",
        "Полная [таблица с опубликованными методами](comparison_day02.md) использует только их сохранённые day02-результаты. "
        "Бейзлайны проекта заново не обучались. Совместимость подтверждена по хешам всех эталонов и validation-эталона классификатора, "
        "числу хостов, типу данных и split. В общей таблице опубликованные средние за 02–04 здесь не используются.",
        "",
        "[Графики](summary.png), [все численные значения](comparison_day02.csv), "
        "[сводка пяти сидов](five_seeds.md). Подробности сохраняются локально в runs/dc_sage/report_details/. "
        "В исходных JSON — 200 bootstrap-повторов по хостам; в `means/` интервалы Student-t по пяти сидам. "
        "Это разные источники неопределённости, они не объединены в один интервал.",
        "",
        "## Критерии успеха из инструкции",
        "",
    ]
    for key, value in checks["checks"].items():
        lines.append(f"- `{key}`: **{value}**.")
    lines += [
        "",
        "Критерий 1: нет задачи с непересекающимся неблагоприятным 95% интервалом сидов. "
        "Критерий 2: у Future links или Shared partners благоприятные интервалы не пересекаются. "
        "Критерий 3: падение Neighbors относительно skip-only не более 0.02. "
        "Критерий 4: средний общий балл не ниже линейного агрегата. "
        "Это предложенные автором эвристики, не доказательство эквивалентности при пересечении интервалов. "
        "Подробно: [success_criteria.json](success_criteria.json).",
        "",
        "## Новые хосты",
        "",
        "1936 хостов (30%, seed2026) исключены до обучения: удалены все их потоки как src или dst, "
        "их присутствие в графе, вклад в SVD/IDF и нормировки. Оставлены 4517 обучающих хостов. "
        "Переобучены пять сетей с замороженными параметрами основного опыта и 25 эпохами, выбранными основным пилотом; "
        "метрики отложенных хостов не участвовали в остановке. Новые хосты добавляются только при выводе по их потокам train-окна. "
        "Галерея соседей содержит все оцениваемые хосты; усреднение метрик — только по отложенным запросам.",
        "",
        f"Отложенных запросов в test: {inductive[0]['heldout_test_hosts']}. "
        f"Neighbors@10: **{ms([r['neighbors@10'] for r in inductive])}**; "
        f"новые собеседники MRR: **{ms([r['new_partners']['embedding']['mrr'] for r in inductive])}**. "
        f"Профиль без нейросети, тоже обученный без этих хостов: **{baseline_new['neighbors@10']:.3f}** и "
        f"**{baseline_new['new_partners']['embedding']['mrr']:.3f}** соответственно. "
        "[Результаты](new_hosts.json), протокол исключения сохраняется локально в runs/dc_sage/report_details/.",
        "",
        "## Искусственные изменения",
        "",
        "Для одних и тех же 272 хостов (5% test, seed2026) созданы независимые сценарии. "
        "Замена поведения удаляет их исходные потоки и копирует потоки случайного донора из другого KMeans-кластера "
        "замороженных train-векторов. Сканер добавляет каждому 200 попыток на порт 445 к случайным узлам. "
        "Безобидный сценарий дублирует все их потоки один раз. Параметры по результатам инъекций не подбирались.",
        "",
        "| сценарий | ROC-AUC дрейфа DC-SAGE, 5 сидов | дрейф исходных признаков |",
        "|---|---|---|",
    ]
    for key in ("role_change", "scanner", "benign_x2"):
        rows = [r for r in anomalies if r["scenario"] == key]
        lines.append(f"| {key} | {ms([r['embedding_drift']['roc_auc'] for r in rows])} | {rows[0]['profile_drift']['roc_auc']:.3f} |")
    lines += [
        "",
        "Дрейф — `1 − cos(train, изменённый test)`. Для двух первых сценариев желателен большой AUC; "
        "для безобидного изменения — около 0.5. Это диагностика реакции на синтетику, не оценка реальных атак. "
        "[Полные результаты](anomalies.json).",
        "",
        "## Веса и повторение",
        "",
        f"Готовый экспорт: `runs/dc_sage/export/`, выбран **seed{freeze['selected_export_seed_by_validation']} по validation до test**. "
        "В нём `best.pt`, `preprocessor.pkl`, `embeddings.csv` (train), `test.csv`, `*_parts.npz`, "
        "`SHARED_COORDINATES` и provenance. Это один конкретный checkpoint, а не ансамбль пяти сидов. "
        "Векторы можно проверить `python -m evaluation.validate`. В scoreboard отправка пока не добавлена: "
        "без test_day03/test_day04 она не проходит текущий полный многодневный протокол.",
        "",
        "Команды всех стадий и применения к новому CSV — в [инструкции запуска](../../experiments/dc_sage/README.md). "
        "Версии, устройство CPU и хеши кода — [environment.json](environment.json). "
        "SVD/IDF/нормировки и веса не меняются при применении к новым окнам.",
        "",
        "Ограничения: один день, старый test, косвенные эталоны и лишь пять сидов. "
        "Следующая независимая проверка требует потоков дней 03–04. "
        "Подгонять новые параметры по уже просмотренному test дня 02 нельзя.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")


def build(work, out):
    details = work / "report_details"
    details.mkdir(parents=True, exist_ok=True)
    groups = {name: [json.loads((work / "results/test" / f"{name}_seed{s}.json").read_text()) for s in range(5)] for name in variants(Config())}
    baselines = [json.loads((work / "results/test" / f"{name}.json").read_text()) for name in ("profile", "behavioral", "linear", "random")]
    for r in baselines:
        if r["method"] == "behavioral":
            r["method"] = "dc_behavioral64"
    mean = [aggregate(name, rows) for name, rows in groups.items()]
    published = [r for r in existing_day02() if r["method"] != "random"]
    scoreboard.check_compatible([*mean, *baselines, *published])
    out.mkdir(parents=True, exist_ok=True)
    (out / "five_seeds.md").write_text(table(groups, baselines))
    (out / "comparison_day02.md").write_text(scoreboard.headline_table([*mean, *baselines, *published]))
    scoreboard.full_table([*mean, *baselines, *published]).to_csv(out / "comparison_day02.csv", index=False)
    for r in mean:
        dump(details / "means" / f"{r['method']}.json", r)
    checks = criteria(groups, next(r for r in baselines if r["method"] == "linear"))
    dump(out / "success_criteria.json", checks)
    shutil.copytree(work / "results", details / "metrics", dirs_exist_ok=True)
    for filename in ("protocol.json", "frozen_config.json", "evaluation_freeze.json", "feature_manifest.json"):
        shutil.copy2(work / filename, out / filename)
    shutil.copy2(work / "cache/manifest.json", out / "data_manifest.json")
    for path, dest in (("inductive/results.json", "new_hosts.json"), ("inductive/holdout.json", "new_hosts_protocol.json"),
                       ("anomalies/results.json", "anomalies.json"), ("anomalies/manifest.json", "anomalies_protocol.json")):
        shutil.copy2(work / path, (details if dest.endswith("_protocol.json") else out) / dest)
    shutil.copy2(work / "inductive/profile_baseline.json", out / "new_hosts_profile_baseline.json")
    for variant in groups:
        for s in range(5):
            folder = work / "models" / variant / f"seed{s}"
            for filename in ("selection.json", "history.json"):
                dest = details / "training" / variant / f"seed{s}"
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(folder / filename, dest / filename)
    source_files = [ROOT / "methods/graph/_dc_sage.py", *sorted((ROOT / "experiments/dc_sage").glob("*.py")), ROOT / "experiments/dc_sage/Instruction.md"]
    dump(out / "environment.json", {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                                    "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                                    "device": "CPU", "threads": 4,
                                    "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in source_files}})
    if (work / "replay_verification.json").exists():
        shutil.copy2(work / "replay_verification.json", out / "replay_verification.json")
    # Конкретный экспортный seed выбран по validation ДО теста, не по лучшему test.
    seed = json.loads((work / "evaluation_freeze.json").read_text())["selected_export_seed_by_validation"]
    export = work / "export"
    export.mkdir(exist_ok=True)
    shutil.copy2(work / "models/dc_sage" / f"seed{seed}/best.pt", export / "best.pt")
    shutil.copy2(work / "preprocessor.pkl", export / "preprocessor.pkl")
    for filename in ("embeddings.csv", "test.csv", "train_parts.npz", "test_parts.npz"):
        shutil.copy2(work / "submissions" / f"dc_sage_seed{seed}" / filename, export / filename)
    (export / "SHARED_COORDINATES").touch()
    dump(export / "provenance.json", {"trained_on_windows": ["train"], "seed": seed, "seed_selection": "validation",
                                       "test_windows": ["test"], "missing_windows": ["test_day03", "test_day04"]})
    shutil.copy2(ROOT / "experiments/dc_sage/README.md", export / "README.md")
    shutil.copy2(ROOT / "experiments/dc_sage/Instruction.md", export / "Instruction.md")
    readme(work, out, groups, baselines, published, checks)
    figures(work, out, groups, baselines)
    print(table(groups, baselines))
    print("Критерии:", checks["checks"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", type=Path, default=ROOT / "runs/dc_sage")
    ap.add_argument("--output", type=Path, default=ROOT / "results/dc_sage_day02")
    args = ap.parse_args()
    build(args.workdir, args.output)


if __name__ == "__main__":
    main()
