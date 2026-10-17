"""Вкладки дашборда: каждая функция рисует один вид по прогону (app/results.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.results import CHANCE, LEAKY, METRIC_GROUPS, PALETTE, Bundle

LABELS = {k: v for g in METRIC_GROUPS.values() for k, v in g}


def _color(m: str) -> str:
    return PALETTE.get(m, "#5f6368")


def _style(fig: go.Figure, h: int = 380, legend_y: float = -0.3) -> go.Figure:
    """Общий стиль: легенда ПОД графиком, не наезжает на подписи осей."""
    fig.update_layout(template="plotly_white", height=h, margin=dict(t=30, b=10, l=10, r=10),
                      legend=dict(orientation="h", yanchor="top", y=legend_y, xanchor="left", x=0, title_text=""),
                      font=dict(size=13))
    fig.update_xaxes(automargin=True)
    return fig


def _fmt(v, spec: str = "{:.3f}") -> str:
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else spec.format(v)


def _text_table(df: pd.DataFrame, spec: str = "{:.3f}", highlight: pd.DataFrame | None = None):
    """Пропуски как '-' (Streamlit показывает NaN как 'None'); необязательная зелёная подсветка."""
    txt = df.apply(lambda col: col.map(lambda v: _fmt(v, spec)))
    if highlight is None:
        return txt
    return txt.style.apply(lambda col: ["background-color: #d9f2e1" if highlight.loc[i, col.name] else "" for i in col.index])


def _rows(df: pd.DataFrame, extra: int = 1) -> int:
    return 35 * (len(df) + extra) + 3          # таблица видна целиком, строки не обрезаются


def _methods_for_chart(b: Bundle, names, key: str):
    """Методы для графика; «утекающий» потолок по желанию, чтобы не сплющивал остальные столбцы."""
    if any(m in LEAKY for m in names) and not st.toggle("Показать потолок эталона (утекает намеренно)",
                                                          value=False, key=key):
        return [m for m in names if m not in LEAKY]
    return list(names)


def banner(b: Bundle) -> None:
    """Всегда пишем вид происхождения данных; ни один не бывает «проверенным»."""
    if b.kind == "synthetic_fixture":
        st.warning(f"**{b.badge}.** Все числа ниже получены на сгенерированных данных в формате LANL NetFlow. "
                   "Они показывают протокол и метрики; о LANL отсюда ничего не следует.",
                   icon="⚠️")
    elif b.kind == "lanl_declared":
        st.info(f"**{b.badge}.** Заявленный источник: *{b.source_note or '—'}*. Пайплайн не может проверить, откуда "
                "файлы; выводы относятся только к этим файлам при указанном протоколе.", icon="ℹ️")
    else:
        st.error(f"**{b.badge}.** Ничего здесь не выдаётся за результат на LANL. Чтобы пометить прогон как LANL, "
                 "укажите источник (`--data-kind lanl --source-note ...`).", icon="🚫")


# --------------------------------------------------------------------------------------- обзор
def overview(b: Bundle) -> None:
    m, pm = b.manifest, b.processed_manifest or {}
    c = st.columns(4)
    c[0].metric("Хостов в когорте", m["n_hosts"])
    c[1].metric("Оценено методов", len(m["methods"]))
    c[2].metric("Размерность", m["config"]["dim"])
    c[3].metric("Повторов бутстрепа", m["config"]["bootstrap"])
    coh = b.cohort()
    if coh:
        st.markdown("##### Когорта")
        order = [w for w in ("validation", "test") if w in coh["n_evaluable"]]
        cov = pd.DataFrame({"оцениваемых хостов": coh["n_evaluable"], "покрытие": coh["coverage"]}).loc[order]
        cov.index.name = "окно"
        c2 = st.columns([1, 2])
        c2[0].metric("Политика когорты", coh["policy"])
        c2[1].dataframe(cov.style.format({"покрытие": "{:.0%}"}), width="stretch")
        if coh.get("selection"):
            with st.expander("Смещение отбора: от чего зависит когорта"):
                st.write(coh["selection"]["note"])
                st.caption(f"Хостов, активных во всех окнах: {coh['selection']['n_hosts_active_in_every_window']} из "
                           f"{coh['n_cohort']}. " + coh["selection"].get("interpretation", ""))
    st.markdown("##### Что оценивается")
    st.markdown(
        "Эмбеддинги строятся только по **train**-окну. Они сравниваются с эталонами из **более позднего** окна: "
        "поведенческие соседи, будущие связи, общие собеседники и (если есть) внешние метки. Каждый эталон отвечает "
        "на свой вопрос — см. docs/METHODOLOGY.md.")
    miss = m.get("missing_expected_methods") or []
    if miss:
        st.info("**В этом прогоне не оценены** (эмбеддинги не переданы, ничего не симулируется): " + ", ".join(miss), icon="ℹ️")
    if pm:
        st.markdown("##### Качество данных")
        w = pd.DataFrame(pm["windows"]).T[["start", "end", "n_flows", "graph_edges"]]
        w.index.name = "окно"
        st.dataframe(w.loc[[x for x in ("train", "validation", "test") if x in w.index]], width="stretch")
        inv = pm.get("invalid_rows", {})
        cnt = pm.get("counts", {})
        q = st.columns(3)
        q[0].metric("Выброшено битых строк", int(sum(inv.values())) if inv else 0)
        q[1].metric("Выброшено потоков на границе окон", int(sum(v for k, v in cnt.items() if k.startswith("straddling"))))
        q[2].metric("Размер когорты", pm["cohort"]["n_hosts"])
        if inv:
            st.caption("Битые строки по причинам: " + ", ".join(f"{k}={v}" for k, v in inv.items()))


# --------------------------------------------------------------------------------------- таблица результатов
def leaderboard(b: Bundle) -> None:
    target = st.radio("Окно эталона", b.targets, horizontal=True, key="lb_target",
                      help="Подбор — на validation, итоговые числа — на test.")
    group = st.selectbox("Группа метрик", list(METRIC_GROUPS), key="lb_group")
    keys = [k for k, _ in METRIC_GROUPS[group]]
    board = b.board(target, keys)
    show_ci = st.toggle("Показать 95% бутстреп-интервалы", value=False)
    if board.isna().all().all():
        st.warning("Ни одна из этих метрик недоступна в этом прогоне.")
        for u in b.unavailable(target):
            st.write(f"- {u}")
        return
    if show_ci:
        cells = board.copy().astype(object)
        for m in board.index:
            for k in keys:
                v, ci = board.loc[m, k], b.ci(target, m, k)
                cells.loc[m, k] = "-" if pd.isna(v) else (f"{v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else f"{v:.3f}")
        st.dataframe(cells.rename(columns=LABELS), width="stretch")
        methods_used = sorted({v for r in b.results(target).values() for k, v in r.get("ci_methods", {}).items() if k in keys})
        if methods_used:
            st.caption("Метод интервалов: " + "; ".join(methods_used) + ". Интервалы описывают только выбор оцениваемых элементов.")
    else:
        honest = board.drop(index=[m for m in board.index if m in LEAKY])
        mask = pd.DataFrame(False, index=board.index, columns=board.columns)
        for k in keys:
            if honest[k].notna().any():
                mask.loc[honest[k].idxmax(), k] = True
        st.dataframe(_text_table(board.rename(columns=LABELS), highlight=mask.rename(columns=LABELS)), width="stretch")
        best = [f"{LABELS[k]}: **{honest[k].idxmax()}** {honest[k].max():.3f}" for k in keys if honest[k].notna().any()]
        if best:
            st.caption("Лучшие среди методов без утечки — " + " · ".join(best))
    st.caption("Потолок эталона — признаки самого целевого окна, в выборе лучшего не участвует. "
               "Случайный уровень: AUC и точность 0.5, lift 1.0, kNN ≈ k/(n-1).")
    shown = _methods_for_chart(b, list(board.index), "lb_ceiling")
    fig = go.Figure()
    for m in shown:
        err = [[(b.ci(target, m, k) or (np.nan, np.nan))[i] for k in keys] for i in (0, 1)]
        vals = board.loc[m, keys].to_numpy(float)
        fig.add_bar(name=m, x=[LABELS[k] for k in keys], y=vals, marker_color=_color(m),
                    error_y=dict(type="data", symmetric=False, array=np.array(err[1]) - vals,
                                 arrayminus=vals - np.array(err[0]), thickness=1))
    fig.update_layout(barmode="group", yaxis_title="значение", xaxis_tickangle=-30)
    st.plotly_chart(_style(fig, 460, legend_y=-0.42), width="stretch")
    un = b.unavailable(target)
    if un:
        with st.expander(f"Недоступные метрики ({len(un)})"):
            for u in un:
                st.write(f"- {u}")


# --------------------------------------------------------------------------------------- прогноз связей
def links(b: Bundle) -> None:
    ctx, audit = b.link_context(), b.audit()
    board = b.board("test", ["link_roc_auc", "link_learned_roc_auc", "link_new_roc_auc", "link_pr_auc_lift"])
    if board["link_roc_auc"].isna().all():
        st.warning("Кандидатов будущих связей (edges.csv) в этом прогоне нет.")
        return
    c = st.columns(4)
    c[0].metric("Доля связей среди кандидатов", f"{ctx['sampled_positive_rate']:.2f}")
    c[1].metric("Настоящая доля связей", f"{ctx['natural_prevalence']:.3f}")
    c[2].metric("Связи, уже бывшие в source", f"{ctx['share_positives_seen_in_source']:.0%}")
    c[3].metric("ROC-AUC «уже общались»", f"{ctx['persistence_roc_auc']:.3f}")
    shown = _methods_for_chart(b, list(board.index), "link_ceiling")
    fig = go.Figure()
    for key, name, col in (("link_roc_auc", "косинусная близость", "#9aa0a6"), ("link_learned_roc_auc", "обученный классификатор рёбер", "#1a73e8")):
        fig.add_bar(name=name, x=shown, y=board.loc[shown, key], marker_color=col)
    ref_lines = [("случайно (0.5)", 0.5, "#d93025", "dot")]
    if ctx["persistence_roc_auc"]:
        ref_lines.append(("«пара уже общалась в source»", ctx["persistence_roc_auc"], "#188038", "dash"))
    for name, y, col, dash in ref_lines:
        fig.add_hline(y=y, line_dash=dash, line_color=col)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=name, line=dict(color=col, dash=dash)))
    if audit and "link_learned_roc_auc" in audit["random_null"]["metrics"]:
        n = audit["random_null"]["metrics"]["link_learned_roc_auc"]
        fig.add_hrect(y0=n["mean"] - 2 * n["std"], y1=n["mean"] + 2 * n["std"], fillcolor="#1a73e8", opacity=0.12, line_width=0)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="случайные эмбеддинги, классификатор (±2 sd)",
                                 marker=dict(symbol="square", size=12, color="rgba(26,115,232,0.25)")))
    fig.update_layout(barmode="group", yaxis=dict(range=[0.35, 0.85], title="ROC-AUC"))
    st.plotly_chart(_style(fig, 480, legend_y=-0.28), width="stretch")
    st.markdown(
        "**Как читать.** Кандидаты — пары, которые общались (1) или не общались (0) в более позднем окне. "
        "*Косинус* для трафика «клиент–сервер» бывает **ниже случайного**: связанные хосты непохожи, а похожие "
        "клиенты никогда не связаны. *Классификатор* учится на `[|a-b|, a*b]` по кандидатам validation и "
        "оценивается на test. У случайных эмбеддингов он выше 0.5 (вектор работает как ID хоста, и классификатор "
        "запоминает хабы), поэтому всегда сравнивайте с полосой случайных и линией «уже общались». PR-AUC зависит "
        "от доли связей: смотрите lift или значение при настоящей доле.")
    st.dataframe(_text_table(board.rename(columns=LABELS)), width="stretch")


# --------------------------------------------------------------------------------------- соседство
@st.cache_data(show_spinner=False)
def _projection(_b: Bundle, key: str, method: str, k: int) -> pd.DataFrame:
    return _b.projection(method, k)


def neighborhoods(b: Bundle) -> None:
    emb_methods = [m for m in b.methods if (b.run / "embeddings" / m / "train.csv").exists()]
    if not emb_methods or not b.has_neighbors():
        st.warning("В прогоне нет сохранённых эмбеддингов или эталонных соседей — виды по хостам недоступны.")
        return
    c = st.columns([2, 1, 3])
    method = c[0].selectbox("Метод", emb_methods, index=emb_methods.index("behavioral") if "behavioral" in emb_methods else 0)
    k = c[1].select_slider("k", [5, 10, 20], value=10)
    proj = _projection(b, str(b.root), method, k)
    hosts = sorted(proj.index)
    best = proj["agreement"].idxmax()
    host = c[2].selectbox("Хост", hosts, index=hosts.index(best),
                          help="Сравнить соседей хоста в эмбеддинге с замороженным эталоном. "
                               "По умолчанию — хост с наибольшим совпадением у этого метода.")
    nb, ref = b.neighborhood(method, host, k), b.reference_list(host, k)
    overlap = int(nb["in_reference_top_k"].sum())
    st.caption(f"Среднее совпадение соседей у **{method}** при k={k}: {proj['agreement'].mean():.3f}. "
               f"У **{host}**: {overlap} из {k} соседей в эмбеддинге есть и в эталоне "
               f"(совпадение этого хоста: {proj.loc[host, 'agreement']:.2f}).")
    d = proj.reset_index()
    fig = px.scatter(d, x="PC1", y="PC2", color="agreement", hover_name="host_id", range_color=[0, 1],
                     color_continuous_scale="Viridis", opacity=0.75)
    for ids, name, sym, col in ((list(nb.host_id), "соседи в эмбеддинге", "circle-open", "#e8710a"),
                                (list(ref.host_id), "эталонные соседи", "diamond-open", "#d93025")):
        s = proj.loc[ids]
        fig.add_trace(go.Scatter(x=s.PC1, y=s.PC2, mode="markers", name=name, text=ids, hoverinfo="text",
                                 marker=dict(symbol=sym, size=14, line=dict(width=2, color=col), color=col)))
    s = proj.loc[[host]]
    fig.add_trace(go.Scatter(x=s.PC1, y=s.PC2, mode="markers", name=host, hoverinfo="name",
                             marker=dict(symbol="star", size=18, color="black")))
    fig.update_coloraxes(colorbar=dict(title="совпадение", thickness=12, len=0.8))
    st.plotly_chart(_style(fig, 520), width="stretch")
    t1, t2 = st.columns(2)
    nbt = nb.copy()
    nbt["in_reference_top_k"] = nbt["in_reference_top_k"].map({True: "да", False: "нет"})
    nbt["reference_rank"] = nbt["reference_rank"].map(lambda v: "-" if pd.isna(v) else str(int(v)))
    nbt["cosine"] = nbt["cosine"].map(lambda v: f"{v:.3f}")
    t1.markdown("**Соседи в эмбеддинге** (косинус)")
    t1.dataframe(nbt, width="stretch", hide_index=True)
    t2.markdown("**Эталонные соседи** (поведение в позднем окне)")
    t2.dataframe(ref, width="stretch", hide_index=True)
    st.caption("2D-проекция PCA искажает эмбеддинг: расстояния на плоскости — не те, что в метриках. "
               "Цвет — совпадение соседей хоста с замороженным эталоном (Consistency@k).")


# --------------------------------------------------------------------------------------- время
def temporal(b: Bundle) -> None:
    st.markdown("##### Validation против test")
    keys = [k for g in METRIC_GROUPS.values() for k, _ in g]
    avail = [k for k in keys if not b.board("test", [k]).isna().all().all()]
    metric = st.selectbox("Метрика", avail, format_func=lambda k: LABELS[k], index=avail.index("consistency@10_mean"))
    if "validation" in b.targets:
        d = pd.concat([b.board(t, [metric])[metric].rename(t) for t in ("validation", "test")], axis=1)
        d = d.loc[_methods_for_chart(b, list(d.index), "temp_ceiling")]
        fig = go.Figure()
        for t, col in (("validation", "#9aa0a6"), ("test", "#1a73e8")):
            fig.add_bar(name=t, x=list(d.index), y=d[t], marker_color=col)
        if metric in CHANCE:
            fig.add_hline(y=CHANCE[metric], line_dash="dot", line_color="#d93025", annotation_text="случайно")
        fig.update_layout(barmode="group")
        st.plotly_chart(_style(fig, 340), width="stretch")
        audit = b.audit()
        if audit and audit["split_agreement"].get(metric) is not None:
            st.caption(f"Согласие порядка методов на validation и test (тау Кендалла): {audit['split_agreement'][metric]:.2f}. "
                       "Если все методы около случайного уровня, порядок — шум.")
    st.markdown("##### Узнаётся ли хост через время (train → test)")
    ident = b.identity("test")
    if ident:
        fig = go.Figure()
        methods = list(ident)
        raw = [ident[m]["none"].get("top1") if ident[m].get("shared_coordinate_system_declared") else None for m in methods]
        pro = [ident[m]["procrustes"].get("top1") for m in methods]
        fig.add_bar(name="сырые координаты (только методы с общими координатами)", x=methods, y=raw, marker_color="#1a73e8")
        fig.add_bar(name="после выравнивания (Procrustes)", x=methods, y=pro, marker_color="#e8710a")
        chance = ident[methods[0]]["procrustes"]["chance_top1"]
        fig.add_hline(y=chance, line_dash="dot", line_color="#d93025", annotation_text="случайно")
        fig.update_layout(barmode="group", yaxis_title="Top-1: тот же хост найден первым")
        st.plotly_chart(_style(fig, 340), width="stretch")
        st.caption("Правильный ответ — тот же host_id в другом окне: это проверка стабильности *и* различимости, "
                   "а не поведенческой похожести. Эмбеддинги, обученные на каждом окне заново, поворачиваются, поэтому "
                   "сырой косинус для них бессмыслен; поворот подбирается на половине хостов, оценка — на другой.")
    audit = b.audit()
    if audit:
        st.markdown("##### Насколько дрейфует само поведение?")
        s = audit["reference_stability"]
        d = pd.DataFrame({p: v for p, v in s["pairs"].items()}).T
        chron = {"train": 0, "validation": 1, "test": 2}
        d = d.loc[sorted(d.index, key=lambda p: tuple(chron[w] for w in p.split("->")))]
        d.index.name = "окна"
        st.dataframe(_text_table(d), width="stretch")
        st.caption(f"Совпадение поведенческих соседей, построенных по двум окнам (случайно Consistency@{s['k']} = "
                   f"{s['chance_consistency']:.3f}). Это ориентир, чего может достичь эмбеддинг раннего окна на kNN.")


# --------------------------------------------------------------------------------------- аудит
def audit_view(b: Bundle) -> None:
    a = b.audit()
    if not a:
        st.warning("В прогоне нет audit.json (запустите без --no-audit).")
        return
    st.markdown("##### Случайный уровень (случайные эмбеддинги, test)")
    rows = {LABELS.get(k, k): v for k, v in a["random_null"]["metrics"].items() if k in LABELS}
    nulls = pd.DataFrame(rows).T[["mean", "std", "p5", "p95"]]
    st.dataframe(_text_table(nulls), width="stretch", height=_rows(nulls))
    st.caption(f"{a['random_null']['n_draws']} случайных эмбеддингов. Значимость судите по этим полосам, а не по 0.5.")
    w = a["weight_sensitivity"]
    st.markdown("##### Чувствительность поведенческого эталона к весам признаков")
    c = st.columns(3)
    c[0].metric("Среднее совпадение соседей", f"{w['mean_reference_agreement']:.2f}", help=f"случайно {w['chance_agreement']:.3f}")
    c[1].metric("Средний тау Кендалла порядка", f"{w['mean_kendall_tau']:.2f}")
    c[2].metric("Худший тау", f"{w['min_kendall_tau']:.2f}")
    cfg = pd.DataFrame(w["configs"])
    fig = px.bar(cfg, x="config", y="kendall_tau", color=cfg["dropped_block"].fillna("все блоки"),
                 labels={"kendall_tau": "тау Кендалла порядка методов", "config": "вариант весов", "color": "убранный блок"})
    fig.update_layout(yaxis_range=[-1, 1])
    st.plotly_chart(_style(fig, 300), width="stretch")
    pv = a.get("prevalence")
    if pv:
        st.markdown("##### Доля положительных среди кандидатов")
        fl, pr = pv["future_links"], pv["pairs"]
        rows = {"будущие связи": {"в выборке": fl["sampled_positive_rate"], "на самом деле (все пары)": fl["natural_prevalence"]},
                "пары по общим собеседникам": {"в выборке": pr["sampled_positive_rate"],
                                               "среди размеченных": pr.get("judged_population_prevalence"),
                                               "среди всех пар": pr.get("population_positive_rate")}}
        st.dataframe(_text_table(pd.DataFrame(rows).T), width="stretch")
        st.caption("Для связей каждая пара либо общалась, либо нет, поэтому настоящая доля определена; "
                   "для пар по общим собеседникам доля среди размеченных — условная (неразмеченные пары исключены).")
    st.markdown("##### Что метод делит с эталоном по построению")
    ov = pd.DataFrame(a["information_overlap"]).T
    ov.index.name = "method"
    st.table(ov)                              # длинный текст переносится, а не обрезается
    st.caption("Метод, чей вход пересекается с источником эталона, хорошо на нём выступает по тривиальной причине. "
               "Такие строки — точки калибровки, а не независимое доказательство.")


# --------------------------------------------------------------------------------------- происхождение
def provenance(b: Bundle) -> None:
    m = b.manifest
    st.markdown("##### Прогон")
    st.json({k: m[k] for k in ("data_kind", "synthetic", "n_hosts", "config", "methods", "external_methods",
                               "missing_expected_methods", "environment")}, expanded=False)
    st.markdown("##### Эталоны")
    for t in b.targets:
        ref = b.reference_manifest(t)
        if not ref:
            continue
        with st.expander(f"reference/{t}"):
            st.write(f"вид: `{ref.get('kind')}` · правда о ролях: `{ref.get('ground_truth')}` · "
                     f"происхождение проверено: `{ref.get('provenance_verified')}`")
            st.write(f"source-окно: `{ref.get('source_window')}` → целевое окно: `{ref.get('target_window')}`")
            st.json(ref.get("files_sha256", {}), expanded=False)
            if ref.get("judgments"):
                st.json(ref["judgments"], expanded=False)
            if ref.get("reference_views"):
                st.json(ref["reference_views"], expanded=False)
    st.caption("У каждого результата есть SHA-256 всех файлов эталона; таблица результатов отказывается смешивать "
               "разные эталоны, когорты, split, синтетику и реальные данные.")
