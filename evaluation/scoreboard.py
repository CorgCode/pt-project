"""Таблица результатов (scoreboard) по JSON-файлам из evaluation.evaluate.

    python -m evaluation.scoreboard --results-dir runs/x/results/test --output runs/x/scoreboard_test

Пишет два файла:
  <output>.csv — все метрики, строка на метод;
  <output>.md  — главные числа по группам с 95% интервалами, строка «случайный уровень»
                 и средний ранг метода по этим числам.
Смешивать результаты нельзя: если методы оценены на разных эталонах, когортах, split
или типах данных (синтетика / реальные), скрипт падает с ошибкой.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_results(results_dir: Path) -> list[dict]:
    out = []
    for path in sorted(Path(results_dir).glob("*.json")):
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


def headline_table(results: list[dict]) -> str:
    cols, chance = {}, {}
    for r in results:
        for s in r.get("summary", []):
            cols.setdefault(s["metric"], f"{s['group']}: {s['title']}")
            if s.get("chance") is not None:
                chance.setdefault(s["metric"], s["chance"])
    if not cols:
        return "Нет главных метрик (результаты без поля summary).\n"
    values = pd.DataFrame({r["method"]: {k: r["metrics"].get(k) for k in cols} for r in results}).T.astype(float)
    order = values.rank(ascending=False).mean(axis=1).sort_values()
    lines = ["| метод | " + " | ".join(cols.values()) + " | средний ранг |",
             "|---|" + "---|" * len(cols) + "---|"]
    for m in order.index:
        r = next(x for x in results if x["method"] == m)
        cells = [_fmt(r["metrics"].get(k), r.get("metrics_ci", {}).get(k)) for k in cols]
        lines.append(f"| `{m}` | " + " | ".join(cells) + f" | {order[m]:.1f} |")
    lines.append("| *случайный уровень* | " + " | ".join(_fmt(chance.get(k)) for k in cols) + " | |")
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
