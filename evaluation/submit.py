"""Общая таблица результатов: положил эмбеддинг — получил место в таблице.

    # оценить одну отправку (папка scoreboard/submissions/<метод>/)
    python -m evaluation.submit scoreboard/submissions/my_method

    # оценить все отправки, у которых нет свежего результата (так делает CI)
    python -m evaluation.submit --all

    # пересчитать бейзлайны и пять графовых методов на общих данных
    python -m evaluation.submit --builtin all

Папка отправки:
    embeddings.csv        эмбеддинг train-окна: host_id, emb_0, emb_1, ...   (обязательно)
    test.csv              эмбеддинг test-окна тех же хостов — для «узнаваемости во времени» (по желанию)
    SHARED_COORDINATES    пустой файл, если оба файла в одной системе координат (по желанию)

Всё сохраняется в scoreboard/: results/<метод>.json — полный результат, README.md — таблица,
scoreboard.csv — все метрики. Данные для обучения и эталоны — scoreboard/data/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from evaluation import scoreboard
from evaluation.evaluate import run
from evaluation.identity import identity_retrieval
from evaluation.io import load_embeddings
from evaluation.validate import validate_submission

ROOT = Path(__file__).resolve().parents[1]
BOARD = ROOT / "scoreboard"
DATA = BOARD / "data"
SUBMISSIONS = BOARD / "submissions"
RESULTS = BOARD / "results"
N_BOOT = 200
SEED = 0


def _sha(*paths: Path) -> str:
    h = hashlib.sha256()
    for p in paths:
        if p.exists():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _inputs(d: Path) -> list[Path]:
    return [d / "embeddings.csv", d / "test.csv", d / "SHARED_COORDINATES", DATA / "reference" / "test" / "manifest.json"]


def score(sub: Path) -> dict:
    """Проверить и оценить одну отправку, сохранить results/<метод>.json."""
    sub = Path(sub)
    method = sub.name
    emb_file = sub / "embeddings.csv"
    validate_submission(str(emb_file))
    ref_test, ref_val = DATA / "reference" / "test", DATA / "reference" / "validation"
    r = run(emb_file, ref_test, method, n_boot=N_BOOT, seed=SEED, fit_reference_dir=ref_val)
    test_file = sub / "test.csv"
    if test_file.exists():
        validate_submission(str(test_file))
        hosts = pd.read_csv(ref_test / "eval_hosts.csv", dtype=str)["host_id"].tolist()
        a, b = load_embeddings(emb_file), load_embeddings(test_file)
        missing = sorted(set(hosts) - set(b.index))
        if missing:
            raise ValueError(f"{test_file}: нет {len(missing)} оцениваемых хостов (например {missing[:5]})")
        ident = identity_retrieval(a.loc[hosts], b.loc[hosts], alignment="procrustes", seed=SEED, n_boot=N_BOOT)
        r["metrics"].update({"identity_top1": ident["top1"], "identity_mrr": ident["mrr"],
                             "identity_pair_roc_auc": ident["pair_roc_auc"]})
        if ident.get("top1_ci"):
            r["metrics_ci"]["identity_top1"] = list(ident["top1_ci"])
        r["metric_details"]["identity"] = {**ident, "shared_coordinate_system_declared": (sub / "SHARED_COORDINATES").exists()}
        r["metric_status"]["identity"] = "ok"
        r["summary"].append({"group": "Узнаваемость", "metric": "identity_top1", "title": "Тот же хост через время, Top-1",
                             "value": ident["top1"], "ci": r["metrics_ci"].get("identity_top1"),
                             "chance": ident["chance_top1"]})
    else:
        r["metric_status"]["identity"] = "unavailable: нет test.csv (эмбеддинга test-окна)"
        r["unavailable_metrics"].append("identity: нет test.csv (эмбеддинга test-окна)")
    r["submission"] = str(emb_file.relative_to(ROOT)) if emb_file.is_relative_to(ROOT) else str(emb_file)
    r["inputs_sha256"] = _sha(*_inputs(sub))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{method}.json").write_text(json.dumps(r, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                                            encoding="utf-8")
    return r


def is_fresh(sub: Path) -> bool:
    f = RESULTS / f"{sub.name}.json"
    return f.exists() and json.loads(f.read_text(encoding="utf-8")).get("inputs_sha256") == _sha(*_inputs(sub))


def render() -> None:
    """Пересобрать scoreboard/README.md и scoreboard.csv из results/."""
    results = scoreboard.load_results(RESULTS)
    known = {d.name for d in SUBMISSIONS.iterdir() if d.is_dir()} if SUBMISSIONS.exists() else set()
    for r in results:                                   # результат без отправки — устарел
        if r["method"] not in known:
            (RESULTS / f"{r['method']}.json").unlink()
    results = [r for r in results if r["method"] in known]
    if not results:
        return
    scoreboard.check_compatible(results)
    scoreboard.full_table(results).to_csv(BOARD / "scoreboard.csv", index=False)
    _, _, order = scoreboard.ranking(results)
    info = results[0].get("reference_info") or {}
    leader = f"**Лидер: `{order.index[0]}`** (средний ранг {order.iloc[0]:.1f} по главным метрикам)." if len(order) else ""
    kind = {"synthetic_fixture": "синтетика в формате LANL — проверка кода, а не результат на реальной сети",
            "lanl_declared": "заявленные данные LANL"}.get(info.get("data_kind"), "непроверенные данные")
    text = f"""# Таблица результатов

Данные: **{kind}**; оцениваемых хостов: {results[0].get('n_hosts')}; эталон: test-окно.
{leader}

{scoreboard.headline_table(results)}
**Жирным** — лучшее значение в колонке. В скобках — 95% бутстреп-интервал. *Курсивом* — ориентиры, а не методы:
`random` — случайные векторы, `reference_ceiling` — копия эталона (потолок, «утекает» намеренно).
«Случайный уровень» — чего ждать от бесполезного эмбеддинга (для собеседников — лучший прогноз без эмбеддинга).
Средний ранг считается по главным метрикам, которые есть у метода. Все метрики — [`scoreboard.csv`](scoreboard.csv),
полные результаты — `results/<метод>.json`, что значит каждая метрика — [`evaluation/README.md`](../evaluation/README.md).

## Как попасть в таблицу

1. Обучите метод **только на train-окне** общих данных:

   ```python
   from pipeline.data import ProcessedData
   d = ProcessedData("scoreboard/data/processed")
   flows, graph, hosts = d.training_flows(), d.training_graph(), d.cohort   # учиться можно только на этом
   test_flows = d.flows("test")                                             # только чтобы встроить test-окно
   ```

2. Положите файлы в `scoreboard/submissions/<имя_метода>/`:
   - `embeddings.csv` — эмбеддинг train-окна для всех хостов `d.cohort`: `host_id, emb_0, emb_1, …`;
   - `test.csv` — по желанию: эмбеддинг test-окна (тем же методом) для узнаваемости во времени;
   - `SHARED_COORDINATES` — пустой файл, если у окон одна система координат.
3. Откройте PR — CI сам оценит отправку и допишет результат и таблицу в ваш PR.
   Или локально: `python -m evaluation.submit scoreboard/submissions/<имя_метода>`.

Подбирать параметры по этой таблице нельзя — это test. Для подбора: `python -m experiments.tune_methods` (validation).

## Когда будут настоящие данные LANL

Пересобрать `scoreboard/data/` из настоящих файлов и пересчитать всех (старые отправки придётся обучить заново):

```bash
python -m pipeline.prepare run --netflow /path/netflow_day-*.bz2 --config cfg.json --output-dir scoreboard/data/processed --data-kind lanl --source-note "форма LANL, <дата>, <кто>"
python -m pipeline.reference --processed-dir scoreboard/data/processed --output-dir scoreboard/data/reference --pairs-per-anchor 10
python -m evaluation.submit --builtin all --force
```
"""
    (BOARD / "README.md").write_text(text, encoding="utf-8")


def builtin(names: list[str], dim: int = 16) -> None:
    """Обучить бейзлайны и графовые методы на общих данных и положить их как обычные отправки."""
    from benchmark import methods as graph_methods
    from benchmark.baselines.temporal import default_baselines
    from pipeline.data import ProcessedData

    data = ProcessedData(DATA / "processed")
    hosts_test = data.evaluable("test")
    base = {b.name: b for b in default_baselines(dim, SEED)}
    wanted = list(base) + sorted(graph_methods.available()) if names == ["all"] else names
    for name in wanted:
        m = base[name] if name in base else graph_methods.create([name], dim, SEED)[0]
        m.fit(data)
        d = SUBMISSIONS / name
        d.mkdir(parents=True, exist_ok=True)
        m.embed(data, "train").to_csv(d / "embeddings.csv", index_label="host_id")
        m.embed(data, "test", hosts_test).to_csv(d / "test.csv", index_label="host_id")
        shared = d / "SHARED_COORDINATES"
        if m.info().get("shared_coordinate_system"):
            shared.touch()
        elif shared.exists():
            shared.unlink()
        print("готово:", name)
    if names == ["all"]:                                # потолок: признаки эталона как эмбеддинг
        d = SUBMISSIONS / "reference_ceiling"
        d.mkdir(parents=True, exist_ok=True)
        f = load_embeddings(DATA / "reference" / "test" / "features.csv", "host_id", "ref_")
        f.columns = [c.replace("ref_", "emb_") for c in f.columns]
        f.to_csv(d / "embeddings.csv", index_label="host_id")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submission", nargs="*", type=Path, help="папки scoreboard/submissions/<метод>")
    ap.add_argument("--all", action="store_true", help="оценить все отправки без свежего результата")
    ap.add_argument("--force", action="store_true", help="пересчитать даже свежие (например, после смены данных)")
    ap.add_argument("--builtin", nargs="+", metavar="METHOD", help="'all' или имена бейзлайнов / графовых методов")
    a = ap.parse_args()
    if a.builtin:
        builtin(a.builtin)
    todo = list(a.submission)
    if a.all or a.builtin:
        todo += sorted(d for d in SUBMISSIONS.iterdir() if d.is_dir() and (d / "embeddings.csv").exists())
    for d in dict.fromkeys(todo):
        if not a.force and not a.submission.count(d) and is_fresh(d):
            continue
        r = score(d)
        print(f"оценено: {d.name}  " + ", ".join(f"{s['title']}: {s['value']:.3f}" for s in r["summary"]
                                                 if s["value"] is not None))
    render()
    if (BOARD / "README.md").exists():
        print((BOARD / "README.md").read_text(encoding="utf-8").split("**Жирным**")[0])


if __name__ == "__main__":
    main()
