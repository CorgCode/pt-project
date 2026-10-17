"""Дашборд бенчмарка эмбеддингов хостов.

    streamlit run app/streamlit_app.py

Показывает прогон: опубликованный results/<имя> (по умолчанию — LANL) или локальный runs/<имя>.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # корень репозитория для benchmark/evaluation/pipeline

import streamlit as st

from app import views
from app.results import Bundle, BundleError, discover

st.set_page_config(page_title="Эмбеддинги хостов — дашборд", layout="wide")


@st.cache_resource(show_spinner=False)
def load_bundle(path: str) -> Bundle:
    return Bundle(Path(path))


# ------------------------------------------------------------------------------------------ раскладка
with st.sidebar:
    st.header("Результаты")
    options = {str(p.relative_to(REPO)): p for p in discover(REPO)}
    env = os.environ.get("PT_RESULTS_DIR")
    if env:
        options[env] = Path(env)
    keys = list(options) or ["(ничего не найдено)"]
    choice = st.selectbox("Прогон", keys, index=keys.index(env) if env in options else 0,
                          help="Опубликованные прогоны — results/<имя>, локальные — runs/<имя>. "
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
    st.info("Прогон не выбран. Свой метод проверяется через общую таблицу: scoreboard/README.md.")
    st.stop()

views.banner(bundle)
tabs = st.tabs(["Обзор", "Таблица", "Прогноз связей", "Соседи", "Во времени", "Аудит", "Происхождение"])
for tab, fn in zip(tabs, (views.overview, views.leaderboard, views.links, views.neighborhoods,
                          views.temporal, views.audit_view, views.provenance)):
    with tab:
        fn(bundle)
