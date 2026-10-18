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

# метрика → (задача, вопрос, случайный уровень)
METRICS = [
    ("consistency@10", "Поведение", "Из 10 ближайших соседей хоста — сколько тех же, что у эталона поведения в test.", "≈ 10 / число хостов"),
    ("link ROC-AUC", "Будущие связи", "По паре векторов предсказать, будут ли хосты общаться в test. Классификатор учится на validation.", "тот же классификатор на случайных векторах, 0,64–0,77"),
    ("new-partner MRR", "Будущие связи", "Для хоста ранжировать кандидатов в новые собеседники (которых не было в train).", "лучший прогноз без эмбеддинга, 0,065"),
    ("pair ROC-AUC", "Общие собеседники", "Хосты с общими собеседниками ближе друг к другу, чем без них?", "0,5"),
    ("identity top-1", "Узнаваемость", "Ближайший вектор в test-окне — тот же хост? Нужен test.csv.", "1 / число хостов, 0,001"),
]

# метод → (модель, кто прислал)
MODELS = {
    "behavioral": "Профиль поведения хоста из NetFlow: объём, порты, протоколы, собеседники.",
    "behavioral_pca": "behavioral, сжатый PCA до 16.",
    "behavioral_svd": "behavioral, сжатый TruncatedSVD до 16.",
    "graph_spectral": "Спектральное вложение графа host–host.",
    "deepwalk": "DeepWalk: random walks + skip-gram по графу host–host.",
    "node2vec": "node2vec: блуждания с p, q + skip-gram по графу host–host.",
    "deepwalk_services": "DeepWalk по графу host–service, ослабление хабов, dim 32.",
    "node2vec_services": "node2vec по графу host–service, p = 0,5, q = 2 (подбор на validation), dim 32.",
    "graphsage": "GraphSAGE: агрегация признаков соседей.",
    "rolx": "RolX: структурные роли из признаков графа.",
    "struc2vec": "struc2vec: похожесть по структуре окрестности.",
    "hybrid_services": "behavioral × 4 + node2vec_services + graph_spectral (вес подобран на validation).",
    "hybrid_weighted": "behavioral × w + node2vec + graph_spectral (вес подобран на validation).",
    "hybrid_behavioral_node2vec_spectral": "behavioral + node2vec + graph_spectral.",
    "hybrid_behavioral_node2vec": "behavioral + node2vec.",
    "hybrid_behavioral_deepwalk": "behavioral + deepwalk.",
    "random": "Случайные векторы — нижняя граница.",
    "reference_ceiling": "Сам эталон поведения — верхняя граница для consistency.",
}
AUTHOR = "physcorgi"   # все методы в таблице прислал один участник; вклад остальных — в разделе «Кто что сделал»

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
    cols = list(team["columns"])
    real = [r for r in team["rows"] if not r.get("service")]
    service = [r for r in team["rows"] if r.get("service")]

    def task_cells(r):
        return "".join(f'<td class="n">{"—" if r["tasks"].get(t) is None else round(r["tasks"][t])}</td>' for t in tasks)

    def raw_cells(r):
        return "".join(f'<td class="n">{f3(r["values"].get(c))}</td>' for c in cols)

    rows = []
    for i, r in enumerate(real, 1):
        m = r["method"]
        rows.append(f'<tr{" class=lead" if i == 1 else ""}><td class="n">{i}</td><td><code>{escape(m)}</code>'
                    f'<small>{escape(MODELS.get(m, ""))}</small></td><td class="n">{r["dim"] or "—"}</td>'
                    f'<td>@{AUTHOR}</td><td class="n b">{f1(r["score"])}</td>{task_cells(r)}</tr>')
    for r in service:
        m = r["method"]
        rows.append(f'<tr class="svc"><td></td><td><code>{escape(m)}</code><small>{escape(MODELS.get(m, ""))}</small></td>'
                    f'<td class="n">{r["dim"] or "—"}</td><td>ориентир</td><td class="n">{f1(r["score"])}</td>{task_cells(r)}</tr>')

    raw = "".join(f'<tr><td><code>{escape(r["method"])}</code></td>{raw_cells(r)}</tr>' for r in real + service)
    chance = "".join(f'<td class="n">{f3(team["chance"][c][0]) if abs(team["chance"][c][1] - team["chance"][c][0]) < 1e-9 else f3(team["chance"][c][0]) + "–" + f3(team["chance"][c][1])}</td>'
                     for c in cols)
    metric_en = {"consistency@10_mean": "consistency@10", "link_learned_roc_auc": "link ROC-AUC",
                 "partner_new_mrr": "new-partner MRR", "pair_roc_auc": "pair ROC-AUC", "identity_top1": "identity top-1"}

    metric_rows = "".join(f"<tr><td><b>{TASK_EN[t]}</b></td><td><code>{escape(m)}</code></td><td>{escape(q)}</td><td>{escape(c)}</td></tr>"
                          for m, t, q, c in METRICS)
    team_rows = "".join(f'<li><b>@{u}</b> — {escape(what)} ' + " ".join(f'<a href="{REPO}/{p}">{n}</a>' for n, p in links) + "</li>"
                        for u, what, links in TEAM)
    papers = "".join(f"<li><b>{n}</b> — {escape(t)}</li>" for n, t in PAPERS)
    lead = real[0]
    ex = " · ".join(f"{TASK_EN[t]} {round(lead['tasks'][t])}" for t in tasks)

    return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PT-05 — сводка</title>
<meta name="description" content="PT-05: метрики, как считается score, результаты на LANL, кто что сделал.">
<style>
:root{{--bg:#fff;--ink:#17201f;--muted:#5b6664;--rule:#e1e4e0;--soft:#f3f5f3;--accent:#1f6f6b}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151b1a;--ink:#e7ecea;--muted:#9eaaa7;--rule:#2b3432;--soft:#1d2524;--accent:#6cc2bb}}}}
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
@media (max-width:700px){{.grid{{grid-template-columns:1fr}}}}
@page{{size:A4;margin:8mm}}
@media print{{:root{{--bg:#fff;--ink:#000;--muted:#444;--soft:#f2f2f2}}main{{padding:0;max-width:none}}body{{font-size:9px;line-height:1.35}}h2{{margin:.8rem 0 .3rem;padding-top:.4rem}}th,td{{padding:.15rem .4rem}}table{{font-size:8.5px}}td small{{display:inline;margin-left:.5em;font-size:.95em}}th{{font-size:.85em}}code{{font-size:.95em}}.formula{{font-size:1em;padding:6px 10px}}.tw{{overflow:visible}}h2{{break-after:avoid}}tr{{break-inside:avoid}}a{{color:inherit;text-decoration:none}}.noprint{{display:none}}}}
</style>
</head>
<body>
<main>
<h1>PT-05 · Эмбеддинги хостов по NetFlow — сводка</h1>
<p class="sub">МФТИ × Positive Technologies · {date.today().strftime("%d.%m.%Y")} · данные из коммита <code>{escape(info["meta"]["source_commit"][:8])}</code>
<span class="noprint">· <a href="./">сайт</a> · <a href="{REPO}">репозиторий</a></span></p>

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
<tr><th>Задача</th><th>Метрика</th><th>Что проверяет</th><th>Случайный уровень</th></tr>
{metric_rows}
</table></div>
<p>Без эмбеддинга правило «пара уже общалась» даёт link ROC-AUC {f3(team["persistence_auc"])}: 88% связей test уже были в train.
Поэтому в задаче Future links есть и new-partner MRR — она про новых собеседников.</p>

<h2>Как считается score</h2>
<div class="formula">1. каждая метрика:  s = 100 · (v − случайный) / (1 − случайный), обрезка в [0, 100]<br>
2. задача:         среднее геометрическое её метрик<br>
3. score:          среднее геометрическое 4 задач, значения ниже 1 считаются за 1</div>
<p>Пример, <code>{escape(lead["method"])}</code>: {ex} → score <b>{f1(lead["score"])}</b>.</p>
<div class="grid">
<div><p><b>Почему так</b></p><ul>
<li><b>Нормировка к случайному уровню.</b> У метрик разная шкала: AUC 0,75 — это слабо, MRR 0,23 — хорошо. После нормировки 0 — случайно, 100 — идеал.</li>
<li><b>Задачи, а не метрики.</b> Две метрики про связи не должны весить вдвое больше одной про узнаваемость.</li>
<li><b>Среднее геометрическое.</b> Считается рост во сколько раз: 8 → 16 весит как 45 → 90. Провал в одной задаче не скрыть хорошими AUC.</li>
<li><b>Score не зависит от других методов.</b> Новый метод не меняет баллы остальных.</li>
</ul></div>
<div><p><b>Проверки устойчивости</b></p><ul>
<li>Порог «ниже 1 считать за 1» от 0,1 до 2 — порядок методов не меняется.</li>
<li>Шум в метриках в пределах их 95% интервалов: лидер остаётся первым в 96% случаев (у простого среднего — 89%).</li>
<li>5 вариантов весов задач: лидер везде <code>{escape(lead["method"])}</code>, лучший одиночный метод — <code>behavioral</code>.</li>
<li>Сейчас веса задач равные. Предложение: Behavior и Identity × 2 — <b>нужно ваше решение</b>.</li>
</ul></div>
</div>

<h2>Результаты (test)</h2>
<div class="tw"><table>
<tr><th class="n">#</th><th>Method · модель</th><th class="n">dim</th><th>Прислал</th><th class="n">Score</th>{"".join(f'<th class="n">{TASK_EN[t]}</th>' for t in tasks)}</tr>
{"".join(rows)}
</table></div>
<p>Баллы задач — 0–100. Исходные значения метрик:</p>
<div class="tw"><table>
<tr><th>Method</th>{"".join(f'<th class="n">{metric_en.get(c, c)}</th>' for c in cols)}</tr>
{raw}
<tr class="svc"><td>случайный уровень</td>{chance}</tr>
</table></div>
<p>95% интервалы — в <a href="{REPO}/blob/main/scoreboard/README.md">scoreboard/README.md</a>.</p>

<h2>Кто что сделал</h2>
<ul>{team_rows}</ul>

<h2>Что прочитали</h2>
<ul>{papers}</ul>

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
