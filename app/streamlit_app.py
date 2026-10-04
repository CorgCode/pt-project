"""Interactive benchmark demo.  Run:  streamlit run app/streamlit_app.py"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root for benchmark/evaluation

import plotly.express as px
import streamlit as st

from app import core

st.set_page_config(page_title="Host embedding benchmark", layout="wide")


@st.cache_resource
def demo_reference() -> Path:
    return core.make_demo_reference(Path(tempfile.mkdtemp(prefix="demo_reference_")))


@st.cache_data
def sample_csv() -> bytes:
    with tempfile.TemporaryDirectory() as wd:
        return core.sample_embedding(demo_reference(), Path(wd))


st.title("Host embedding benchmark")
st.caption("Upload an `embeddings.csv` (`host_id, emb_0, emb_1, ...`) and compare it with simple baselines.")
st.info("Reference data is **synthetic demo data** (300 hosts, 5 roles), not real traffic. "
        "Scores are for exploring the benchmark only.", icon="ℹ️")

with st.sidebar:
    st.header("Input")
    upload = st.file_uploader("embeddings.csv", type=["csv"])
    st.download_button("Download sample embeddings.csv", sample_csv(), "embeddings.csv", "text/csv",
                       help="A valid file for the demo reference; also shows the expected format.")
    run_clicked = st.button("Run evaluation", type="primary", disabled=upload is None)

if upload is None:
    st.write("Upload a file in the sidebar to begin.")
    st.stop()

if run_clicked:
    try:
        with st.spinner("Evaluating..."):
            st.session_state["outcome"] = core.evaluate_upload(upload.getvalue(), demo_reference())
            st.session_state["name"] = upload.name
    except core.UploadError as exc:
        st.session_state.pop("outcome", None)
        st.error(str(exc))
        st.stop()

outcome = st.session_state.get("outcome")
if outcome is None:
    st.write("Press **Run evaluation**.")
    st.stop()
if st.session_state.get("name") != upload.name:
    st.warning("Showing results for a previous file; press **Run evaluation** to update.")

board = outcome["board"]
up = board.loc[core.UPLOADED]

st.subheader("Uploaded embedding")
cols = st.columns(3)
for col, (m, label) in zip(cols, core.METRICS.items()):
    with col:
        st.markdown(f"**{label}**")
        sub = st.columns(len(core.KS))
        for c, k in zip(sub, core.KS):
            c.metric(f"@{k}", f"{up[f'{m}@{k}']:.3f}")

st.subheader("Comparison")
tabs = st.tabs([*core.METRICS.values(), "Leaderboard"])
for tab, m in zip(tabs[:-1], core.METRICS):
    with tab:
        long = core.metric_long(board, m)
        fig = px.bar(long, x="k", y=core.METRICS[m], color="method", barmode="group",
                     category_orders={"method": list(board.index)}, range_y=[0, 1])
        fig.update_layout(margin=dict(t=10, b=10), legend_title=None)
        st.plotly_chart(fig, use_container_width=True)
with tabs[-1]:
    st.dataframe(board.style.format("{:.3f}").highlight_max(axis=0, color="#d4edda"),
                 use_container_width=True)
    st.caption("Sorted by Consistency@10. Baselines use the same dimension as the upload "
               "(capped by the reference size). The reference ceiling leaks by design.")

st.subheader("2D PCA projection")
proj = outcome["projection"]
if proj is None:
    st.info("The embedding has a single dimension; a 2D projection needs at least 2.")
else:
    fig = px.scatter(proj, x="PC1", y="PC2", hover_name="host_id", opacity=0.75)
    fig.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Metrics not available yet")
st.warning("These benchmark metrics are not computed yet; they are shown as unavailable "
           "rather than estimated (issues #22-#25):")
st.markdown("\n".join(f"- {u}" for u in outcome["unavailable"]))
