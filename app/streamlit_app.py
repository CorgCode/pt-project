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

st.set_page_config(page_title="Host embedding research dashboard", layout="wide")


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
    st.caption("Upload an `embeddings.csv` (`host_id, emb_0, emb_1, ...`) for the synthetic demo cohort (`h0`..`h299`) "
               "and compare it with simple baselines. This tab uses its own small synthetic reference, independent of "
               "the bundle selected in the sidebar.")
    c = st.columns([3, 1])
    upload = c[0].file_uploader("embeddings.csv", type=["csv"])
    c[1].download_button("Sample embeddings.csv", sample_csv(), "embeddings.csv", "text/csv")
    if upload is None:
        return
    if not st.button("Run evaluation", type="primary"):
        st.stop()
    try:
        with st.spinner("Evaluating..."):
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
    st.warning("Not available yet: " + "; ".join(out["unavailable"]))


# ------------------------------------------------------------------------------------------ раскладка
with st.sidebar:
    st.header("Results")
    options = {str(p.relative_to(REPO)): p for p in discover(REPO)}
    env = os.environ.get("PT_RESULTS_DIR")
    if env:
        options[env] = Path(env)
    keys = list(options) or ["(none found)"]
    choice = st.selectbox("Run bundle", keys, index=keys.index(env) if env in options else 0,
                          help="demo/synthetic_run is a committed synthetic example; real runs live in runs/<name>. "
                               "PT_RESULTS_DIR, when set, is selected by default.")
    custom = st.text_input("...or a path", placeholder="runs/lanl")
    st.caption("Create a bundle with `python -m experiments.end_to_end` (see experiments/README.md).")

st.title("Host embedding research dashboard")

path = Path(custom) if custom else options.get(choice)
bundle = None
if path is not None:
    try:
        bundle = load_bundle(str(path))
    except BundleError as exc:
        st.error(str(exc))

if bundle is None:
    st.info("No run bundle selected. The *Try your embedding* tab still works.")
    (tab,) = st.tabs(["Try your embedding"])
    with tab:
        try_it()
    st.stop()

views.banner(bundle)
tabs = st.tabs(["Overview", "Leaderboard", "Link prediction", "Neighbourhoods", "Temporal",
                "Validity audit", "Provenance", "Try your embedding"])
for tab, fn in zip(tabs[:-1], (views.overview, views.leaderboard, views.links, views.neighborhoods,
                               views.temporal, views.audit_view, views.provenance)):
    with tab:
        fn(bundle)
with tabs[-1]:
    try_it()
