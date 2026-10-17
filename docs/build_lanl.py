"""Вставить на сайт таблицу результатов прогона на LANL.

    python docs/build_lanl.py results/lanl_day02

Берёт главные числа из results/test/*.json (те же, что в таблице результатов) и пишет HTML-таблицу
в docs/index.html между <!-- LANL:BEGIN --> и <!-- LANL:END -->.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.scoreboard import SERVICE, load_results, ranking  # noqa: E402

PAGE = ROOT / "docs" / "index.html"
BEGIN, END = "<!-- LANL:BEGIN -->", "<!-- LANL:END -->"


def _cell(v, ci=None, best=False) -> str:
    if v is None:
        return "<td class=\"num\">—</td>"
    s = f"{v:.3f}".replace(".", ",")
    if ci:
        s += f" <small>[{ci[0]:.2f}–{ci[1]:.2f}]</small>".replace(".", ",")
    return f"<td class=\"num\">{'<b>' + s + '</b>' if best else s}</td>"


def render(run_dir: Path) -> str:
    results = load_results(Path(run_dir) / "results" / "test")
    cols, values, order = ranking(results)
    by = {r["method"]: r for r in results}
    chance = {}
    for r in results:
        for s in r.get("summary", []):
            if s.get("chance") is not None:
                chance.setdefault(s["metric"], s["chance"])
    real = values.loc[order.index]
    best = {k: real[k].max() for k in cols if real[k].notna().any()}
    manifest = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))
    coh = manifest.get("cohort", {})
    head = "".join(f"<th>{html.escape(t)}</th>" for t in cols.values())
    rows = []
    for i, m in enumerate(list(order.index) + sorted(SERVICE & set(by)), 1):
        r = by[m]
        place = str(i) if m not in SERVICE else "—"
        cells = "".join(_cell(r["metrics"].get(k), r.get("metrics_ci", {}).get(k),
                              m not in SERVICE and r["metrics"].get(k) is not None and k in best
                              and abs(r["metrics"][k] - best[k]) < 1e-12) for k in cols)
        name = f"<i>{html.escape(m)}</i>" if m in SERVICE else html.escape(m)
        rank = f"{order[m]:.1f}".replace(".", ",") if m in order.index else "—"
        rows.append(f"<tr><td>{place}</td><td>{name}</td>{cells}<td class=\"num\">{rank}</td></tr>")
    rows.append("<tr><td></td><td><i>случайный уровень</i></td>" + "".join(_cell(chance.get(k)) for k in cols) + "<td></td></tr>")
    n_eval = coh.get("n_evaluable", {}).get("test")
    return f"""{BEGIN}
    <p class="intro">Один день (netflow_day-02), окна: train — 1 час, test — 30 минут. Когорта — {coh.get('n_cohort')} активных хостов,
    оценено на test — {n_eval}. Жирным — лучшее в колонке, в скобках — 95% интервал. Место — по среднему рангу.</p>
    <div class="tablewrap"><table>
      <thead><tr><th>место</th><th>метод</th>{head}<th>средний ранг</th></tr></thead>
      <tbody>
        {chr(10).join('        ' + r for r in rows).strip()}
      </tbody>
    </table></div>
    <p>Полные результаты, отчёт и аудит — <a href="https://github.com/CorgCode/pt-project/tree/main/results/lanl_day02">results/lanl_day02</a>.
    Происхождение данных заявлено (форма LANL), пайплайном не проверяется.</p>
    {END}"""


def main() -> None:
    run_dir = Path(sys.argv[1])
    page = PAGE.read_text(encoding="utf-8")
    a, b = page.index(BEGIN), page.index(END) + len(END)
    PAGE.write_text(page[:a] + render(run_dir) + page[b:], encoding="utf-8")
    print("таблица LANL вставлена в docs/index.html")


if __name__ == "__main__":
    main()
