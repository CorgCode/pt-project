"""Дашборд бенчмарка эмбеддингов хостов.

    streamlit run app/streamlit_app.py

Показывает прогон (по умолчанию — синтетическое демо; или любой runs/<имя> от experiments.end_to_end)
и вкладку для оценки своего файла эмбеддингов.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # корень репозитория для benchmark/evaluation/pipeline

import plotly.express as px
import streamlit as st

from app import core, views
from app.results import Bundle, BundleError, discover

st.set_page_config(page_title="Эмбеддинги хостов — дашборд", layout="wide")


@st.cache_resource(show_spinner=False)
def load_bundle(path: str) -> Bundle:
    return Bundle(Path(path))


@st.cache_resource
def demo_reference() -> Path:
    return core.make_demo_reference(Path(tempfile.mkdtemp(prefix="demo_reference_")))


@st.cache_data
def sample_csv() -> bytes:
    with tempfile.TemporaryDirectory() as wd:
        return core.sample_embedding(demo_reference(), Path(wd))


def try_it() -> None:
    """Оценить загруженный embeddings.csv на маленьком синтетическом эталоне демо."""
    st.caption("Загрузите `embeddings.csv` (`host_id, emb_0, emb_1, ...`) для синтетической демо-когорты (`h0`..`h299`) "
               "и сравните с простыми бейзлайнами. У этой вкладки свой маленький синтетический эталон, не зависящий "
               "от прогона в боковой панели.")
    c = st.columns([3, 1])
    upload = c[0].file_uploader("embeddings.csv", type=["csv"])
    c[1].download_button("Пример embeddings.csv", sample_csv(), "embeddings.csv", "text/csv")
    if upload is None:
        return
    if not st.button("Оценить", type="primary"):
        st.stop()
    try:
        with st.spinner("Считаю метрики..."):
            out = core.evaluate_upload(upload.getvalue(), demo_reference())
    except core.UploadError as exc:
        st.error(str(exc))
        return
    board, up = out["board"], out["board"].loc[core.UPLOADED]
    cols = st.columns(3)
    for col, (m, label) in zip(cols, core.METRICS.items()):
        col.markdown(f"**{label}**")
        sub = col.columns(len(core.KS))
        for s, k in zip(sub, core.KS):
            s.metric(f"@{k}", f"{up[f'{m}@{k}']:.3f}")
    st.dataframe(board.style.format("{:.3f}"), width="stretch")
    if out["projection"] is not None:
        st.plotly_chart(px.scatter(out["projection"], x="PC1", y="PC2", hover_name="host_id", opacity=0.75)
                        .update_layout(template="plotly_white", height=420), width="stretch")
    st.warning("Пока недоступно: " + "; ".join(out["unavailable"]))


# ------------------------------------------------------------------------------------------ раскладка
with st.sidebar:
    st.header("Результаты")
    options = {str(p.relative_to(REPO)): p for p in discover(REPO)}
    env = os.environ.get("PT_RESULTS_DIR")
    if env:
        options[env] = Path(env)
    keys = list(options) or ["(ничего не найдено)"]
    choice = st.selectbox("Прогон", keys, index=keys.index(env) if env in options else 0,
                          help="demo/synthetic_run — синтетический пример из репозитория; настоящие прогоны — в runs/<имя>. "
                               "Если задан PT_RESULTS_DIR, он выбран по умолчанию.")
    custom = st.text_input("...или путь", placeholder="runs/lanl")
    st.caption("Прогон создаётся командой `python -m experiments.end_to_end` (см. experiments/README.md).")

st.title("Эмбеддинги хостов: результаты")

path = Path(custom) if custom else options.get(choice)
bundle = None
if path is not None:
    try:
        bundle = load_bundle(str(path))
    except BundleError as exc:
        st.error(str(exc))

if bundle is None:
    st.info("Прогон не выбран. Вкладка *Свой эмбеддинг* всё равно работает.")
    (tab,) = st.tabs(["Свой эмбеддинг"])
    with tab:
        try_it()
    st.stop()

views.banner(bundle)
tabs = st.tabs(["Обзор", "Таблица", "Прогноз связей", "Соседи", "Во времени",
                "Аудит", "Происхождение", "Свой эмбеддинг"])
for tab, fn in zip(tabs[:-1], (views.overview, views.leaderboard, views.links, views.neighborhoods,
                               views.temporal, views.audit_view, views.provenance)):
    with tab:
        fn(bundle)
with tabs[-1]:
    try_it()
