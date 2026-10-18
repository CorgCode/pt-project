"""Одностраничная сводка для ментора: метрики, как считается score, таблица результатов, кто что сделал.

    python docs/build_report.py            # читает docs/data/results.json, пишет docs/report.html

Числа берутся только из results.json (его собирает docs/build_data.py), текст — отсюда.
"""
from __future__ import annotations

import json
from datetime import date
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = "https://github.com/CorgCode/pt-project"

TASK_EN = {"Поведение": "Behavior", "Будущие связи": "Future links",
           "Общие собеседники": "Shared partners", "Узнаваемость": "Identity"}

# задача → [(метрика, что проверяет, случайный уровень)]
METRICS = {
    "Поведение": [("consistency@10", "Из 10 ближайших соседей хоста — сколько тех же, что у эталона поведения в test.", "≈ 10 / число хостов")],
    "Будущие связи": [("link ROC-AUC", "Будут ли два хоста общаться в test (классификатор на парах векторов).", "0,64–0,77"),
                      ("new-partner MRR", "Угадать новых собеседников хоста, которых не было в train.", "0,065")],
    "Общие собеседники": [("pair ROC-AUC", "Хосты с общими собеседниками ближе, чем без них?", "0,5")],
    "Узнаваемость": [("identity top-1", "Ближайший вектор в test — тот же хост?", "0,001")],
}

MODELS = {
    "behavioral": "профиль трафика хоста",
    "behavioral_pca": "behavioral → PCA 16",
    "behavioral_svd": "behavioral → SVD 16",
    "graph_spectral": "спектр графа host–host",
    "deepwalk": "блуждания, граф host–host",
    "node2vec": "блуждания p, q, граф host–host",
    "deepwalk_services": "блуждания, граф host–service",
    "node2vec_services": "блуждания p, q, граф host–service",
    "graphsage": "агрегация соседей",
    "rolx": "структурные роли",
    "struc2vec": "структура окрестности",
    "hybrid_services": "behavioral ×4 + node2vec_services + spectral",
    "hybrid_weighted": "behavioral ×w + node2vec + spectral",
    "hybrid_behavioral_node2vec_spectral": "behavioral + node2vec + spectral",
    "hybrid_behavioral_node2vec": "behavioral + node2vec",
    "hybrid_behavioral_deepwalk": "behavioral + deepwalk",
    "random": "случайные векторы",
    "reference_ceiling": "сам эталон поведения",
}
GRAPH = {"graph_spectral", "deepwalk", "node2vec", "deepwalk_services", "node2vec_services", "graphsage", "rolx", "struc2vec"}
AUTHOR = "physcorgi"   # все методы в таблице прислал один участник; вклад остальных — в «Кто что сделал»

TEAM = [
    ("physcorgi", "конвейер LANL → окна → граф, все методы в таблице, оценка и scoreboard, сайт",
     [("#75", "pull/75"), ("#76", "pull/76"), ("#79", "pull/79"), ("#80", "pull/80")]),
    ("Tikhon2783", "link prediction — основа задачи «Будущие связи»; triplet accuracy", [("#65", "pull/65"), ("#36", "pull/36")]),
    ("kalivixZ", "устойчивость эмбеддинга во времени — основа задачи «Узнаваемость»; подготовка данных и графа",
     [("#64", "pull/64"), ("#37", "pull/37")]),
    ("ElliOrNora", "линейный зонд; кластеризация и поиск аномалий (на ревью)", [("#39", "pull/39"), ("#67", "pull/67")]),
]

PAPERS = [
    ("DeepWalk", "Perozzi, Al-Rfou, Skiena. DeepWalk: Online Learning of Social Representations. KDD 2014."),
    ("node2vec", "Grover, Leskovec. node2vec: Scalable Feature Learning for Networks. KDD 2016."),
    ("RolX", "Henderson et al. RolX: Structural Role Extraction & Mining in Large Graphs. KDD 2012."),
    ("struc2vec", "Ribeiro, Saverese, Figueiredo. struc2vec: Learning Node Representations from Structural Identity. KDD 2017."),
    ("GraphSAGE", "Hamilton, Ying, Leskovec. Inductive Representation Learning on Large Graphs. NeurIPS 2017."),
    ("LANL", "Turcotte, Kent, Hash. Unified Host and Network Data Set. LANL, 2017 (csr.lanl.gov/data/2017)."),
]


def f3(v) -> str:
    return "—" if v is None else f"{v:.3f}".replace(".", ",")


def f1(v) -> str:
    return "—" if v is None else f"{v:.1f}".replace(".", ",")


def num(v) -> str:
    return f"{v:,}".replace(",", " ")


def build(info: dict) -> str:
    team, ds = info["team"], info["dataset"]
    tasks = list(team["tasks"])
    real = [r for r in team["rows"] if not r.get("service")]
    service = [r for r in team["rows"] if r.get("service")]
    by = {r["method"]: r for r in team["rows"]}

    def cell(r, t):
        v = r["tasks"].get(t)
        if v is None or (r["method"] == "reference_ceiling" and t != "Поведение"):
            return '<td class="h">—</td>'
        raw = " · ".join(f3(r["values"].get(c)) for c in team["tasks"][t])
        p = round(min(100, v) * 0.8)
        dark = " d" if p > 45 else ""
        return f'<td class="h{dark}" style="background:color-mix(in srgb,var(--heat) {p}%,transparent)">{round(v)}<small>{raw}</small></td>'

    def row(r, place=None):
        m, svc = r["method"], r.get("service")
        sc = "—" if m == "reference_ceiling" else f1(r["score"])
        bar = "" if svc else f'<i style="width:{max(2, r["score"]) * 1.4:.0f}px"></i>'
        cls = ' class="svc"' if svc else (' class="lead"' if place == 1 else "")
        return (f'<tr{cls}><td class="n">{place or ""}</td><td><code>{escape(m)}</code><small>{escape(MODELS.get(m, ""))}</small></td>'
                f'<td class="n">{r["dim"] or "—"}</td><td class="sc"><b>{sc}</b>{bar}</td>'
                + "".join(cell(r, t) for t in tasks) + "</tr>")

    rows = "".join(row(r, i) for i, r in enumerate(real, 1)) + "".join(row(r) for r in service)
    metric_rows = "".join(
        "".join(f'<tr>{f"<td rowspan={len(ms)}><b>{TASK_EN[t]}</b></td>" if j == 0 else ""}<td><code>{escape(m)}</code></td>'
                f'<td>{escape(q)}</td><td class="n">{escape(c)}</td></tr>' for j, (m, q, c) in enumerate(ms))
        for t, ms in METRICS.items())
    team_rows = "".join(f'<li><b>@{u}</b> — {escape(what)} ' + " ".join(f'<a href="{REPO}/{p}">{n}</a>' for n, p in links) + "</li>"
                        for u, what, links in TEAM)
    papers = "".join(f"<li><b>{n}</b> — {escape(t)}</li>" for n, t in PAPERS)
    lead = real[0]
    single = next(r for r in real if not r["method"].startswith("hybrid"))
    graph = [r["score"] for r in real if r["method"] in GRAPH]
    ex = " · ".join(f"{TASK_EN[t]} {round(lead['tasks'][t])}" for t in tasks)
    n2v, n2vs = by.get("node2vec"), by.get("node2vec_services")
    services = (f"<li>Граф host–service вместо host–host: <code>node2vec</code> {f1(n2v['score'])} → <b>{f1(n2vs['score'])}</b>.</li>"
                if n2v and n2vs else "")

    return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PT-05 — сводка</title>
<meta name="description" content="PT-05: метрики, как считается score, результаты на LANL, кто что сделал.">
<style>
:root{{--bg:#fff;--ink:#17201f;--muted:#5b6664;--rule:#e1e4e0;--soft:#f3f5f3;--accent:#1f6f6b;--heat:#1f6f6b}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151b1a;--ink:#e7ecea;--muted:#9eaaa7;--rule:#2b3432;--soft:#1d2524;--accent:#6cc2bb;--heat:#4fb3ab}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif}}
main{{max-width:1000px;margin:0 auto;padding:32px 16px 48px}}
h1{{font-size:1.6rem;margin:0}}h2{{font-size:1.1rem;margin:1.8rem 0 .5rem;padding-top:.8rem;border-top:1px solid var(--rule)}}
p,li{{max-width:80ch}}p{{margin:.4rem 0}}ul,ol{{margin:.3rem 0;padding-left:1.2rem}}
.sub{{color:var(--muted);margin:.3rem 0 0}}
a{{color:var(--accent)}}code{{font:.86em ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}}
.tw{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:.86rem;margin:.4rem 0}}
th,td{{text-align:left;padding:.35rem .5rem;border-bottom:1px solid var(--rule);vertical-align:top}}
th{{font-size:.72rem;text-transform:uppercase;letter-spacing:.03em;color:var(--muted);font-weight:600}}
td.n,th.n{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}td.b{{font-weight:700}}
td small{{display:block;color:var(--muted);font-size:.78rem}}
tr.lead td{{background:var(--soft)}}tr.svc td{{color:var(--muted);font-style:italic}}
.formula{{background:var(--soft);border-radius:8px;padding:10px 14px;font:.9rem ui-monospace,Menlo,Consolas,monospace;overflow-x:auto}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:0 32px}}
.facts{{display:flex;flex-wrap:wrap;gap:6px 24px;margin:.8rem 0 0;color:var(--muted)}}.facts b{{color:var(--ink)}}
table.res td{{vertical-align:middle}}
td.h,th.h{{text-align:center;font-variant-numeric:tabular-nums;border-left:2px solid var(--bg);font-weight:600}}
td.h small{{font-weight:400;font-size:.72rem;white-space:nowrap}}td.h.d{{color:#fff}}td.h.d small{{color:rgba(255,255,255,.85)}}
td.sc{{white-space:nowrap}}td.sc b{{display:inline-block;min-width:2.4em;font-variant-numeric:tabular-nums}}
td.sc i{{display:inline-block;height:7px;border-radius:4px;background:var(--accent);vertical-align:middle;margin-left:4px}}
.key{{background:var(--soft);border-left:3px solid var(--accent);border-radius:6px;padding:8px 14px;margin:1rem 0 0}}.key p{{margin:0}}
@media (max-width:700px){{.grid{{grid-template-columns:1fr}}}}
@page{{size:A4;margin:8mm}}
@media print{{:root{{--bg:#fff;--ink:#000;--muted:#444;--soft:#f2f2f2}}main{{padding:0;max-width:none}}body{{font-size:9px;line-height:1.35}}h2{{margin:.8rem 0 .3rem;padding-top:.4rem}}th,td{{padding:.15rem .4rem}}table{{font-size:8.5px}}td small{{display:inline;margin-left:.5em;font-size:.95em}}td.h small{{display:block;margin:0;font-size:.85em;white-space:nowrap}}td.sc i{{height:5px}}.key{{padding:4px 10px}}main{{zoom:.8}}h1{{font-size:1.9em}}h2{{font-size:1.25em;margin:.5rem 0 .2rem;padding-top:.3rem}}p{{margin:.2rem 0}}.facts{{margin:.3rem 0 0;gap:2px 16px}}.grid p{{margin:0}}ul{{margin:.1rem 0}}table{{margin:.2rem 0}}th,td{{padding:.1rem .35rem}}*{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}th{{font-size:.85em}}code{{font-size:.95em}}.formula{{font-size:1em;padding:6px 10px}}.tw{{overflow:visible}}h2{{break-after:avoid}}tr{{break-inside:avoid}}a{{color:inherit;text-decoration:none}}.noprint{{display:none}}}}
</style>
</head>
<body>
<main>
<h1>PT-05 · Эмбеддинги хостов по NetFlow — сводка</h1>
<p class="sub">МФТИ × Positive Technologies · {date.today().strftime("%d.%m.%Y")} · данные из коммита <code>{escape(info["meta"]["source_commit"][:8])}</code>
<span class="noprint">· <a href="./">сайт</a> · <a href="{REPO}">репозиторий</a></span></p>

<div class="key">
<p><b>Главное</b></p>
<ul>
<li>Лучший метод — <code>{escape(lead["method"])}</code>, score <b>{f1(lead["score"])}</b> из 100. Лучший одиночный — <code>{escape(single["method"])}</code>, {f1(single["score"])}.</li>
<li>Графовые методы в одиночку — {f1(min(graph))}–{f1(max(graph))}, но в гибриде с <code>behavioral</code> дают прирост.</li>
{services}
<li><b>Нужно решение:</b> веса задач в score (сейчас равные; предложение — Behavior и Identity × 2).</li>
</ul>
</div>

<h2>Задача и данные</h2>
<p>Для каждого хоста получить вектор по сетевому трафику: хосты с похожим поведением — рядом, тот же хост узнаётся через время.
Учимся только на train-окне, всё оцениваем на более позднем test-окне.</p>
<div class="facts">
<span>Данные: <b>LANL Unified Host and Network 2017</b>, NetFlow, день 02</span>
<span>Записей: <b>{num(ds["valid_rows"])}</b></span>
<span>Окна: train <b>{ds["windows"]["train"]["minutes"]} мин</b> → validation → test <b>{ds["windows"]["test"]["minutes"]} мин</b></span>
<span>Хостов в оценке: <b>{num(ds["evaluated"])}</b></span>
</div>
<p>Эталоны для метрик строятся из логов test-окна, без эмбеддингов (<code>pipeline/reference.py</code>). Подбор параметров — только на validation.</p>

<h2>Метрики</h2>
<div class="tw"><table>
<tr><th>Задача</th><th>Метрика</th><th>Что проверяет</th><th class="n">Случайно</th></tr>
{metric_rows}
</table></div>
<p>88% связей test уже были в train (правило «уже общались» — AUC {f3(team["persistence_auc"])}), поэтому есть new-partner MRR.</p>

<h2>Как считается score</h2>
<div class="formula">1. каждая метрика:  s = 100 · (v − случайный) / (1 − случайный), обрезка в [0, 100]<br>
2. задача:         среднее геометрическое её метрик<br>
3. score:          среднее геометрическое 4 задач, значения ниже 1 считаются за 1</div>
<p>Пример, <code>{escape(lead["method"])}</code>: {ex} → score <b>{f1(lead["score"])}</b>.</p>
<div class="grid">
<div><p><b>Почему так</b></p><ul>
<li>Нормировка к случайному уровню: у метрик разная шкала.</li>
<li>Среднее по задачам, а не по метрикам: две метрики про связи не весят вдвое больше.</li>
<li>Геометрическое: провал в одной задаче не скрыть хорошими AUC.</li>
</ul></div>
<div><p><b>Проверки</b></p><ul>
<li>Порог 0,1–2 вместо 1 — порядок тот же.</li>
<li>Шум в пределах 95% интервалов — лидер первый в 96% случаев (простое среднее — 89%).</li>
<li>5 вариантов весов — лидер везде тот же.</li>
</ul></div>
</div>

<h2>Результаты (test)</h2>
<p class="sub">Все методы прислал @{AUTHOR}; вклад команды — ниже. В ячейке — балл задачи 0–100, мелко — исходные метрики.</p>
<div class="tw"><table class="res">
<tr><th class="n">#</th><th>Method</th><th class="n">dim</th><th>Score</th>{"".join(f'<th class="h">{TASK_EN[t]}</th>' for t in tasks)}</tr>
{rows}
</table></div>
<p class="sub">95% интервалы — в <a href="{REPO}/blob/main/scoreboard/README.md">scoreboard/README.md</a>.</p>

<div class="grid">
<div><h2>Кто что сделал</h2>
<ul>{team_rows}</ul></div>
<div><h2>Что прочитали</h2>
<ul>{papers}</ul></div>
</div>

</main>
</body>
</html>
"""


def main() -> None:
    info = json.loads((ROOT / "data" / "results.json").read_text(encoding="utf-8"))
    out = ROOT / "report.html"
    out.write_text(build(info), encoding="utf-8")
    print(f"{out}: методов {len(info['team']['rows'])}")


if __name__ == "__main__":
    main()
