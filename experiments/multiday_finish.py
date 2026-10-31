"""Зафиксировать победителя по validation и один раз проверить на новом дне."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import cloudpickle

from evaluation import scoreboard
from experiments.multiday_gain import evaluate
from experiments.overnight.model import T
from pipeline.data import ProcessedData


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_selection(candidates: list[Path], out: Path):
    out.mkdir(parents=True, exist_ok=True)
    frozen = out / "frozen_selection.json"
    if frozen.exists():
        return json.loads(frozen.read_text())
    rows = [{**json.loads((d / "best.json").read_text()), "directory": str(d)} for d in candidates]
    rows.sort(key=lambda r: (-r["score"], r["directory"]))
    best = rows[0]
    shutil.copy2(Path(best["directory"]) / "best.pkl", out / "model.pkl")
    record = {"selected_at": datetime.now().astimezone().isoformat(), "selected_on": "validation only",
              "winner": best, "all_validation_winners": rows,
              "checkpoint_sha256": file_hash(out / "model.pkl"), "target_score": 80, "target_gain": 10}
    root = Path(__file__).resolve().parents[1]
    sources = [*Path(__file__).parent.glob("multiday*.py"), root / "scoreboard/temporal_encoder.py",
               *[p for p in (root / "evaluation").glob("*.py") if not p.name.startswith("test_")]]
    record["source_hashes"] = {str(p): file_hash(p) for p in sources}
    frozen.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", nargs="+", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if (args.out / "REPORT.md").exists():
        print("Финальный отчёт уже существует; test повторно не оценивается", flush=True)
        return
    frozen = freeze_selection(args.candidates, args.out)
    if file_hash(args.out / "model.pkl") != frozen["checkpoint_sha256"]:
        raise ValueError("Замороженный checkpoint изменился")
    if any(file_hash(p) != expected for p, expected in frozen["source_hashes"].items()):
        raise ValueError("Код изменился после фиксации модели; автоматическая оценка остановлена")
    data = ProcessedData(args.data)
    scores, results = {}, {}
    for name in ["baseline", "winner"]:
        saved = args.out / name / "scored.json"
        if saved.exists():
            row = json.loads(saved.read_text())
            scores[name], results[name] = row["score"], row["result"]
            continue
        if name == "baseline":
            model = T.TemporalEncoder().fit(data)
        else:
            with (args.out / "model.pkl").open("rb") as source:
                model = cloudpickle.load(source)
        score, result = evaluate(model, data, "test", args.reference, args.out / name, name, n_boot=200)
        scores[name], results[name] = score, result
        saved.write_text(json.dumps({"score": score, "result": result}, ensure_ascii=False, indent=2) + "\n")
        del model
    scoreboard.check_compatible(list(results.values()))
    gain = scores["winner"] - scores["baseline"]
    summary = {"scores": scores, "gain": gain, "score_80_reached": scores["winner"] >= 80,
               "gain_10_reached": gain >= 10, "selected_on": "validation only",
               "test": "day09, 12:30–13:30", "original_scoreboard_comparable": False}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    lines = ["# Многодневное обучение: финальная проверка", "",
             "Train: дни 02–07, validation: 08, test: 09. Когорта задана по дню 02.",
             "Это отдельный протокол: его числа не заменяют баллы исходной таблицы за дни 02–04.", "",
             f"Исходная модель: **{scores['baseline']:.3f}**. Выбранная: **{scores['winner']:.3f}**.",
             f"Прирост на общем test: **{gain:+.3f}**.",
             f"80+ достигнуто: {summary['score_80_reached']}; +10 достигнуто: {summary['gain_10_reached']}.", "",
             f"Выбор по validation: {frozen['winner']['name']} ({frozen['winner']['score']:.3f}).", "",
             "Нормировка Neighbors задана persistence из замороженного эталона и одинакова у всех методов.",
             "Формула геометрического среднего задач сохранена. Для метрик посчитано 200 bootstrap-повторов.", "",
             "| метод | Neighbors@10 | Links AUC | New partners MRR | Pairs AUC | Identity Top-1 |",
             "|---|---|---|---|---|---|"]
    for name, result in results.items():
        metrics = result["metrics"]
        values = [metrics[k] for k in ["consistency@10_mean", "link_learned_roc_auc", "partner_new_mrr",
                                       "pair_roc_auc", "identity_top1"]]
        lines.append(f"| {name} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
