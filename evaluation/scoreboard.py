"""Таблица результатов (scoreboard) по JSON-файлам из evaluation.evaluate.

    python -m evaluation.scoreboard --results-dir runs/x/results/test --output runs/x/scoreboard_test

Пишет два файла:
  <output>.csv — все метрики, строка на метод;
  <output>.md  — главные числа по группам с 95% интервалами и строка «случайный уровень».
Смешивать результаты нельзя: если методы оценены на разных эталонах, когортах, split
или типах данных (синтетика / реальные), скрипт падает с ошибкой.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def load_results(results_dir: Path) -> list[dict]:
    out = []
    for path in sorted(p for p in Path(results_dir).glob("*.json") if not p.name.startswith(".")):   # ._* — служебные файлы macOS
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("method", path.stem)
        out.append(data)
    return out


def check_compatible(results: list[dict]) -> None:
    refs, fits, hosts, kinds, splits = set(), set(), set(), set(), set()
    for r in results:
        info = r.get("reference_info") or {}
        files = info.get("files_sha256")
        fp = tuple(sorted(files.items())) if files else (info.get("features_sha256"), info.get("neighbors_sha256"))
        if not any(fp):
            raise ValueError(f"{r['method']}: нет отпечатка эталона — переоцените метод текущим evaluation.evaluate")
        refs.add(fp)
        fit = info.get("fit_reference_files_sha256")
        if fit and any(k.startswith("link_learned") for k in r.get("metrics", {})):
            fits.add(tuple(sorted(fit.items())))
        hosts.add(r.get("n_hosts"))
        kinds.add((info.get("data_kind", "unspecified"), info.get("synthetic")))
        splits.add(r.get("split"))
    if len(refs) > 1:
        raise ValueError("методы оценены на разных эталонах — сравнивать нельзя")
    if len(fits) > 1:
        raise ValueError("классификатор связей обучен на разных эталонах")
    if len(hosts) > 1:
        raise ValueError("методы оценены на разных когортах хостов")
    if len(kinds) > 1:
        raise ValueError(f"смешаны типы данных (например синтетика и реальные): {sorted(map(str, kinds))}")
    if len(splits) > 1:
        raise ValueError(f"смешаны split: {sorted(map(str, splits))}")


def full_table(results: list[dict]) -> pd.DataFrame:
    rows = [{"method": r["method"], "n_hosts": r.get("n_hosts"), **r.get("metrics", {})} for r in results]
    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)


def _fmt(v, ci=None) -> str:
    if v is None or pd.isna(v):
        return "—"
    s = f"{v:.3f}"
    return f"{s} [{ci[0]:.2f}–{ci[1]:.2f}]" if ci else s


COL_EN = {"Соседи@10": "Neighbors@10", "Связь, AUC": "Links, AUC", "Новые собеседники, MRR": "New partners, MRR",
          "Пары, AUC": "Pairs, AUC", "Узнаваемость, Top-1": "Identity, Top-1"}
TASK_EN = {"Поведение": "Behavior", "Будущие связи": "Future links", "Общие собеседники": "Shared partners",
           "Узнаваемость": "Identity"}
# метрики Тихона (#40, evaluation/tikhon.py): в score не входят; уровень случайности — его бейзлайн random
EXTRA = {"tikhon_lp_direct_mrr": "Tikhon LP, MRR", "tikhon_transfer_direct_mrr": "Tikhon transfer, MRR",
         "tikhon_transfer_soft_mrr": "Tikhon soft transfer, MRR"}
EXTRA_CHANCE = "tikhon_random_mrr"
SERVICE = {"random", "reference_ceiling"}      # ориентиры, а не методы: в выборе лучшего не участвуют


def ranking(results: list[dict]) -> tuple[dict, pd.DataFrame, pd.Series]:
    """Колонки главных метрик, значения и общий балл методов (по убыванию)."""
    cols = {}
    for r in results:
        for s in r.get("summary", []):
            cols.setdefault(s["metric"], s["title"])
    values = pd.DataFrame({r["method"]: {k: r["metrics"].get(k) for k in cols} for r in results}).T.astype(float)
    score = overall_score(results, cols)
    order = score.drop(index=[m for m in score.index if m in SERVICE]).sort_values(ascending=False)
    return cols, values, order


def column_scores(r: dict, cols) -> dict[str, float]:
    """Балл метода в каждой колонке: 0 — случайный уровень, 100 — идеал (1.0)."""
    chance = {s["metric"]: s.get("chance") for s in r.get("summary", [])}
    out = {}
    for k in cols:
        v, c = r["metrics"].get(k), chance.get(k)
        if v is None or c is None or c >= 1:
            continue
        out[k] = 100 * min(max((v - c) / (1 - c), 0.0), 1.0)
    return out


FLOOR = 1.0     # колонка ниже 1 балла считается за 1: иначе один ноль обнуляет весь балл


def _geo(xs) -> float:
    xs = [max(x, FLOOR) for x in xs]
    return float(math.exp(sum(math.log(x) for x in xs) / len(xs)))


def task_scores(r: dict, cols) -> dict[str, float]:
    """Балл по задачам (группам колонок): среднее геометрическое баллов колонок задачи."""
    group = {s["metric"]: s.get("group") or s["metric"] for s in r.get("summary", [])}
    by: dict[str, list[float]] = {}
    for k, v in column_scores(r, cols).items():
        by.setdefault(group.get(k, k), []).append(v)
    return {g: _geo(v) for g, v in by.items()}


def complete(r: dict, cols) -> bool:
    """Есть ли все обязательные колонки: без любой из них общего балла и места нет."""
    return all(r["metrics"].get(k) is not None for k in cols)


def _one_score(r: dict, cols, required: bool = True) -> float:
    """required=False — только для ориентиров (SERVICE): у reference_ceiling части задач нет по устройству."""
    if required and not complete(r, cols):
        return float("nan")             # пропуск сложной задачи не должен поднимать балл
    ts = task_scores(r, cols)
    return _geo(ts.values()) if ts else float("nan")


def overall_score(results: list[dict], cols) -> pd.Series:
    """Общий балл 0–100: среднее геометрическое по задачам. Каждая задача весит одинаково,
    провал в одной тянет вниз весь балл, рост в k раз в любой колонке весит одинаково.
    Если test-окон несколько (by_window) — балл считается на каждом и усредняется."""
    out = {}
    for r in results:
        req = r["method"] not in SERVICE
        if r.get("by_window"):
            days = [_one_score({"summary": w["summary"], "metrics": w["metrics"]}, cols, req)
                    for w in r["by_window"].values()]
            out[r["method"]] = float(np.mean(days)) if not any(np.isnan(days)) else float("nan")
        else:
            out[r["method"]] = _one_score(r, cols, req)
    return pd.Series(out)


def headline_table(results: list[dict], card_url=None) -> str:
    """card_url(метод) -> ссылка на карточку метода (для таблицы scoreboard); None — без ссылок."""
    cols, values, order = ranking(results)
    if not cols:
        return "Нет главных метрик (результаты без поля summary).\n"
    chance: dict[str, list[float]] = {}
    for r in results:
        if r["method"] in SERVICE:
            continue
        for s in r.get("summary", []):
            if s.get("chance") is not None:
                chance.setdefault(s["metric"], []).append(s["chance"])
    score = overall_score(results, cols)
    extra = {k: t for k, t in EXTRA.items() if any(r["metrics"].get(k) is not None for r in results)}
    shown = list(cols) + list(extra)
    real = pd.DataFrame({m: {k: r["metrics"].get(k) for k in shown} for m, r in ((r["method"], r) for r in results)
                         if m in order.index}).T.astype(float)
    best = {k: real[k].max() for k in shown if k in real and real[k].notna().any()}
    lines = ["| # | method | submitted by | **score** | dim | " + " | ".join(COL_EN.get(t, t) for t in cols.values())
             + "".join(f" | {t}*" for t in extra.values()) + " |",
             "|---|---|---|---|---|" + "---|" * len(shown)]
    by_name = {r["method"]: r for r in results}

    def row(place: str, m: str) -> str:
        r = by_name[m]
        cells = []
        for k in shown:
            c = _fmt(r["metrics"].get(k), r.get("metrics_ci", {}).get(k))
            v = r["metrics"].get(k)
            if m not in SERVICE and v is not None and k in best and abs(v - best[k]) < 1e-12:
                c = f"**{c}**"
            cells.append(c)
        name = f"[`{m}`]({card_url(m)})" if card_url else f"`{m}`"
        name = f"*{name}*" if m in SERVICE else name
        sc = ("—" if m in SERVICE else "incomplete") if pd.isna(score.get(m)) else f"{score[m]:.1f}"
        if len(order) and m == order.index[0]:
            sc = f"**{sc}**"
        dim = r.get("dim") or "—"
        who = f"@{r['author']}" if r.get("author") else "—"
        return f"| {place} | {name} | {who} | {sc} | {dim} | " + " | ".join(cells) + " |"

    ranked = order.dropna()                        # без всех задач — без места (неполная оценка)
    for i, m in enumerate(ranked.index, 1):
        lines.append(row(str(i), m))
    for m in order.index.difference(ranked.index):
        lines.append(row("—", m))
    for m in sorted(SERVICE & set(by_name)):
        lines.append(row("—", m))
    def chance_cell(vals):                         # у некоторых метрик уровень зависит от метода (размерности)
        if not vals:
            return "—"
        lo, hi = min(vals), max(vals)
        return _fmt(lo) if hi - lo < 1e-9 else f"{lo:.2f}–{hi:.2f}"

    lines.append("| | *chance level* | | 0 | | " + " | ".join([chance_cell(chance.get(k)) for k in cols]
                                                           + [chance_cell([r["metrics"][EXTRA_CHANCE] for r in results
                                                                           if r["metrics"].get(EXTRA_CHANCE) is not None])]
                                                              * len(extra)) + " |")
    return "\n".join(lines) + "\n"


def window_scores_table(results: list[dict], labels: dict[str, str]) -> str:
    """Score каждого метода на каждом test-окне (если окон несколько): видно, устойчив ли метод по дням."""
    multi = [r for r in results if r.get("by_window")]
    if not multi:
        return ""
    ws = multi[0]["test_windows"]
    cols, _, order = ranking(results)
    per = {w: pd.Series({r["method"]: _one_score(r["by_window"][w], cols, r["method"] not in SERVICE)
                           for r in multi}) for w in ws}
    lines = ["| method | " + " | ".join(labels.get(w, w) for w in ws) + " | **mean** |",
             "|---|" + "---|" * (len(ws) + 1)]
    mean = overall_score(results, cols)
    for m in list(order.index) + sorted(SERVICE & {r["method"] for r in multi}):
        cells = ["—" if pd.isna(per[w].get(m)) else f"{per[w][m]:.1f}" for w in ws]
        lines.append(f"| `{m}` | " + " | ".join(cells) + f" | {'—' if pd.isna(mean.get(m)) else f'{mean[m]:.1f}'} |")
    return "\n".join(lines) + "\n"


def build(results_dir: Path, output: Path) -> pd.DataFrame | None:
    results = load_results(results_dir)
    if not results:
        print("Нет JSON-файлов с результатами.")
        return None
    check_compatible(results)
    df = full_table(results)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output.with_suffix(".csv"), index=False)
    info = results[0].get("reference_info") or {}
    head = (f"# Результаты: split `{results[0].get('split')}`, хостов {results[0].get('n_hosts')}, "
            f"данные `{info.get('data_kind')}`\n\n"
            "В скобках — 95% бутстреп-интервал. «Случайный уровень» — чего ждать от бесполезного эмбеддинга "
            "(для собеседников — лучший бейзлайн без эмбеддинга).\n\n")
    output.with_suffix(".md").write_text(head + headline_table(results), encoding="utf-8")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="путь без расширения (или с .csv)")
    a = ap.parse_args()
    df = build(a.results_dir, a.output)
    if df is not None:
        print((a.output.with_suffix(".md")).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
