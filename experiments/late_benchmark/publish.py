"""Promote only a completed day-90 benchmark; retain the entire historical table."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np

from evaluation import scoreboard, submit
from pipeline.prepare import _sha256

from .evaluate import REQUIRED
from .models import ROOT, WORK
from .prepare import load_protocol


def validate(results, processed, reference, statuses, spec):
    if not results:
        raise ValueError("no late-window results; historical scoreboard remains untouched")
    if processed.get("data_kind") != "lanl_declared" or processed.get("synthetic") is not False:
        raise ValueError("only the real-data benchmark may replace the LANL scoreboard")
    windows = set(spec["test_days"])
    if 90 not in spec["test_days"].values() or set(processed["windows"]) != {"train", "validation"} | windows:
        raise ValueError("day 90 or another predeclared window is missing")
    for name, bounds in spec["windows"].items():
        actual = processed["windows"][name]
        if [actual.get("start"), actual.get("end")] != bounds:
            raise ValueError("processed time windows do not match the predeclared protocol")
    complete_methods = {r["method"] for r in results}
    if {name for name, item in statuses.items() if item["status"] == "trained"} - complete_methods:
        raise ValueError("not all reproducible models have completed evaluation")
    if any(item["status"] == "failed" for item in statuses.values()):
        raise ValueError("a reproducible model failed; resolve it before replacing the scoreboard")
    if any(item["status"] not in ("trained", "unavailable") for item in statuses.values()):
        raise ValueError("model recovery/preparation is still incomplete")
    if not {"behavioral", "random"} <= complete_methods:
        raise ValueError("required calibration/random baselines missing")
    for result in results:
        if result.get("benchmark_id") != spec["benchmark_id"] or set(result.get("by_window", {})) != windows:
            raise ValueError("a method contains old or missing test windows")
        for name, row in result["by_window"].items():
            expected = json.loads((Path(reference) / name / "manifest.json").read_text())
            for filename, digest in expected["files_sha256"].items():
                if digest is not None and _sha256(Path(reference) / name / filename) != digest:
                    raise ValueError("a frozen reference changed before publication")
            if row["reference_info"]["files_sha256"] != expected["files_sha256"]:
                raise ValueError("a method was scored against a different reference")
            if row.get("score_calibration") != expected["score_calibration"]:
                raise ValueError("score calibration differs between methods")
            if any(row["metrics"].get(key) is None or not np.isfinite(row["metrics"][key]) for key in REQUIRED):
                raise ValueError("a model is missing a mandatory late-window metric")
        if result["training_provenance"].get("new_test_used_in_fit_or_selection") is not False:
            raise ValueError("model preparation did not declare an untouched test")
    scoreboard.check_compatible(results)
    cols, _, _ = scoreboard.ranking(results)
    if not np.isfinite(scoreboard.overall_score(results, cols)).all():
        raise ValueError("some scores are undefined")


def publish(data_root=ROOT / "data/late_benchmark", work=WORK, board=ROOT / "scoreboard"):
    data_root, work, board = Path(data_root), Path(work), Path(board)
    spec = load_protocol()
    processed = json.loads((data_root / "processed/manifest.json").read_text())
    statuses = json.loads((work / "model_status.json").read_text())
    results = scoreboard.load_results(work / "results")
    validate(results, processed, data_root / "reference", statuses, spec)
    stage = work / "publication"
    if stage.exists():
        shutil.rmtree(stage)  # this directory contains only regenerated publication output
    (stage / "data").mkdir(parents=True)
    shutil.copytree(data_root / "reference", stage / "data/reference")
    (stage / "data/processed").mkdir()
    for filename in ("cohort.csv", "manifest.json"):
        shutil.copy2(data_root / "processed" / filename, stage / "data/processed" / filename)
    shutil.copytree(work / "results", stage / "results")
    submit.BOARD, submit.DATA, submit.RESULTS, submit.SUBMISSIONS = stage, stage / "data", stage / "results", board / "submissions"
    submit.render()
    content = (stage / "README.md").read_text()
    content = content.replace(
        "Для Neighbors идеал — не 1.0, а уровень `behavioral` (поведение хоста в train): поведение в test-окне\n"
        "во многом новое, и даже признаки за 30 минут до test дают лишь ~0.21. Выше этого уровня — 100.",
        "Для Neighbors идеал — уровень сохранения соседей `behavioral` между исходным train и конкретным поздним окном. "
        "Калибровка записана в манифесте каждого окна; значение выше этого уровня получает 100.")
    details = [f"**Бенчмарк `{spec['benchmark_id']}`.** Обучение и параметры зафиксированы до поздних test-дней. "
               "Сохранённые checkpoints применяются без обучения; остальные методы переобучены по опубликованным конфигурациям на исходных обучающих окнах. "
               "Для многодневных методов использованы только исторические дни до test; окна обучения и происхождение кода записаны в benchmark.json и результатах.",
               "", "Когорта источника фиксирована на дне 02. Покрытие поздних дней не используется для отбора обучающих хостов.",
               "", "| Окно | Активных хостов | Покрытие исходной когорты |", "|---|---:|---:|"]
    for window, day in spec["test_days"].items():
        n = processed["cohort"]["evaluable"][window]
        details.append(f"| День {day} | {n} | {n / processed['cohort']['n_hosts']:.1%} |")
    if any(item.get("recovered_from_pr") in (151, 166) for item in statuses.values()):
        details += ["", "Восстановленные из PR модели обучены заново, без нового подбора. Для GNN параметры beta и lam "
                    "взяты из опубликованных округлённых карточек: исходный файл точных tuned-параметров не сохранён. "
                    "Это повторная реализация опубликованной конфигурации, а не заявление о побитовом совпадении исторических весов."]
    details += ["", "Нормировка Neighbors заново зафиксирована по behavioral persistence для каждого позднего окна. "
                "В таблице показаны и исходные метрики; scores нового протокола нельзя напрямую сравнивать с историческими 70,0.",
                "", "[Историческая таблица дней 14–16](history/days14-16/README.md).", "", "## Методы без переоценки", "",
                "Старые test-векторы не позволяют вычислить узнаваемость на новом дне. Исторические scores в новый рейтинг не перенесены.",
                "", "| Метод | Причина |", "|---|---|"]
    for name, item in sorted(statuses.items()):
        if item["status"] == "unavailable":
            reason = item.get("reason_ru") or ("Нет оригинальной Plus-головы и сборки новых векторов; основа оценена отдельно."
                      if name == "multiday_codebook_plus" else "Нет полного воспроизводимого адаптера inference / checkpoint для нового окна.")
            details.append(f"| `{name}` | {reason} |")
    content = content.replace("# Таблица результатов\n", "# Таблица результатов\n\n" + "\n".join(details) + "\n", 1)
    (stage / "README.md").write_text(content)
    metadata = {"benchmark_id": spec["benchmark_id"], "protocol": spec, "model_status": statuses,
                "processed_manifest_sha256": _sha256(data_root / "processed/manifest.json"),
                "reference_manifests_sha256": {w: _sha256(data_root / "reference" / w / "manifest.json") for w in spec["test_days"]}}
    audit_path = work / "recovery/audit.json"
    if audit_path.exists():
        metadata["recovery_audit"] = json.loads(audit_path.read_text())
    (stage / "benchmark.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    archive = board / "history/days14-16"
    current = board / "benchmark.json"
    if not archive.exists() and not current.exists():
        archive.mkdir(parents=True)
        for filename in ("README.md", "scoreboard.csv"):
            shutil.copy2(board / filename, archive / filename)
        archived_readme = archive / "README.md"
        archived_readme.write_text(archived_readme.read_text().replace(
            "](../evaluation/", "](../../../evaluation/"))
        shutil.copytree(board / "results", archive / "results")
        shutil.copytree(board / "data/reference", archive / "reference")
        shutil.copytree(board / "data/processed", archive / "processed")
    # All data is ready and validated before the visible table is changed.
    # A complete local backup permits rollback if a filesystem operation fails.
    backup = work / "scoreboard_before_publish"
    if not backup.exists():
        backup.mkdir()
        for filename in ("README.md", "scoreboard.csv"):
            shutil.copy2(board / filename, backup / filename)
        shutil.copytree(board / "results", backup / "results")
        shutil.copytree(board / "data", backup / "data")
    for relative in ("data/reference", "data/processed", "results"):
        target = board / relative
        temporary = target.with_name(target.name + ".late-new")
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(stage / relative, temporary)
        previous = target.with_name(target.name + ".late-old")
        if previous.exists():
            raise ValueError("an unfinished previous promotion requires inspection")
        target.rename(previous)
        try:
            temporary.rename(target)
        except BaseException:
            previous.rename(target)
            raise
        shutil.rmtree(previous)
    for filename in ("scoreboard.csv", "benchmark.json", "README.md"):
        temporary = board / (filename + ".late-new")
        shutil.copy2(stage / filename, temporary)
        os.replace(temporary, board / filename)
    marker = {"benchmark_id": spec["benchmark_id"], "published_methods": [r["method"] for r in results],
              "table_sha256": _sha256(board / "README.md"), "publication": "local_only_no_push"}
    (work / "published.json").write_text(json.dumps(marker, indent=2) + "\n")
    if board.resolve() == (ROOT / "scoreboard").resolve():
        readme = ROOT / "README.md"
        text = readme.read_text()
        start = text.index("## Актуальные результаты")
        end = text.index("\n## ", start + 3)
        scores = scoreboard.overall_score(results, scoreboard.ranking(results)[0]).drop(labels=list(scoreboard.SERVICE), errors="ignore")
        winner = scores.idxmax()
        replacement = ("## Актуальные результаты\n\n"
            f"[Активная таблица](scoreboard/README.md) использует протокол `{spec['benchmark_id']}`: "
            f"test — дни 88–90, source — неизменённый час дня 02. Лучший из переоценённых методов — "
            f"`{winner}`, средний score **{scores[winner]:.1f}**.\n\n"
            "Результат `multiday_codebook_plus` 70,0 относится к [историческому бенчмарку дней 14–16](scoreboard/history/days14-16/README.md). "
            "Он не является оценкой на дне 90. Состав переоценённых методов, покрытие и ограничения приведены в активной таблице.\n")
        readme.write_text(text[:start] + replacement + text[end:])
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/late_benchmark")
    parser.add_argument("--work", type=Path, default=WORK)
    args = parser.parse_args()
    print(json.dumps(publish(args.data, args.work), indent=2), flush=True)


if __name__ == "__main__":
    main()
