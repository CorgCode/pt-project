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
    if any(m in LEAKY for m in names) and not st.toggle("Include the reference ceiling (leaks by design)",
                                                          value=False, key=key):
        return [m for m in names if m not in LEAKY]
    return list(names)


def banner(b: Bundle) -> None:
    """Всегда пишем вид происхождения данных; ни один не бывает «проверенным»."""
    if b.kind == "synthetic_fixture":
        st.warning(f"**{b.badge}.** Every number below comes from generated data written in the LANL netflow "
                   "layout. It demonstrates the protocol and the metrics only; no statement about LANL follows.",
                   icon="⚠️")
    elif b.kind == "lanl_declared":
        st.info(f"**{b.badge}.** Source declared: *{b.source_note or 'n/a'}*. The pipeline cannot verify where "
                "the files came from; findings apply to these files under the stated protocol only.", icon="ℹ️")
    else:
        st.error(f"**{b.badge}.** Nothing here is presented as a LANL result. Declare the source "
                 "(`--data-kind lanl --source-note ...`) to label a run as LANL.", icon="🚫")


# --------------------------------------------------------------------------------------- обзор
def overview(b: Bundle) -> None:
    m, pm = b.manifest, b.processed_manifest or {}
    c = st.columns(4)
    c[0].metric("Hosts in cohort", m["n_hosts"])
    c[1].metric("Methods evaluated", len(m["methods"]))
    c[2].metric("Embedding dim", m["config"]["dim"])
    c[3].metric("Bootstrap draws", m["config"]["bootstrap"])
    coh = b.cohort()
    if coh:
        st.markdown("##### Cohort")
        order = [w for w in ("validation", "test") if w in coh["n_evaluable"]]
        cov = pd.DataFrame({"evaluable hosts": coh["n_evaluable"], "coverage": coh["coverage"]}).loc[order]
        cov.index.name = "window"
        c2 = st.columns([1, 2])
        c2[0].metric("Cohort policy", coh["policy"])
        c2[1].dataframe(cov.style.format({"coverage": "{:.0%}"}), width="stretch")
        if coh.get("selection"):
            with st.expander("Selection bias: what this cohort conditions on"):
                st.write(coh["selection"]["note"])
                st.caption(f"Hosts active in every window: {coh['selection']['n_hosts_active_in_every_window']} of "
                           f"{coh['n_cohort']}. " + coh["selection"].get("interpretation", ""))
    st.markdown("##### What is evaluated")
    st.markdown(
        "Embeddings are computed from the **train** window only. They are scored against references built from a "
        "**later** window: behavioural neighbours, future communication, shared-peer structure, and (when "
        "available) external labels. Each view answers a different question - see *Methodology*.")
    miss = m.get("missing_expected_methods") or []
    if miss:
        st.info("**Not evaluated in this run** (no embeddings supplied; nothing is simulated): " + ", ".join(miss), icon="ℹ️")
    if pm:
        st.markdown("##### Data quality")
        w = pd.DataFrame(pm["windows"]).T[["start", "end", "n_flows", "graph_edges"]]
        w.index.name = "window"
        st.dataframe(w.loc[[x for x in ("train", "validation", "test") if x in w.index]], width="stretch")
        inv = pm.get("invalid_rows", {})
        cnt = pm.get("counts", {})
        q = st.columns(3)
        q[0].metric("Invalid rows dropped", int(sum(inv.values())) if inv else 0)
        q[1].metric("Boundary-straddling flows dropped", int(sum(v for k, v in cnt.items() if k.startswith("straddling"))))
        q[2].metric("Cohort size", pm["cohort"]["n_hosts"])
        if inv:
            st.caption("Invalid rows by reason: " + ", ".join(f"{k}={v}" for k, v in inv.items()))


# --------------------------------------------------------------------------------------- таблица результатов
def leaderboard(b: Bundle) -> None:
    target = st.radio("Reference window", b.targets, horizontal=True, key="lb_target",
                      help="Tune on validation; final numbers are reported on test.")
    group = st.selectbox("Metric family", list(METRIC_GROUPS), key="lb_group")
    keys = [k for k, _ in METRIC_GROUPS[group]]
    board = b.board(target, keys)
    show_ci = st.toggle("Show 95% bootstrap intervals", value=False)
    if board.isna().all().all():
        st.warning("None of these metrics is available for this run.")
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
            st.caption("Interval method: " + "; ".join(methods_used) + ". Intervals cover evaluation-set sampling only.")
    else:
        honest = board.drop(index=[m for m in board.index if m in LEAKY])
        mask = pd.DataFrame(False, index=board.index, columns=board.columns)
        for k in keys:
            if honest[k].notna().any():
                mask.loc[honest[k].idxmax(), k] = True
        st.dataframe(_text_table(board.rename(columns=LABELS), highlight=mask.rename(columns=LABELS)), width="stretch")
        best = [f"{LABELS[k]}: **{honest[k].idxmax()}** {honest[k].max():.3f}" for k in keys if honest[k].notna().any()]
        if best:
            st.caption("Best among methods that do not leak - " + " · ".join(best))
    st.caption("The reference ceiling uses the target-window features themselves and is excluded from 'best'. "
               "Chance: AUC/accuracy 0.5, lift 1.0, kNN ≈ k/(n-1).")
    shown = _methods_for_chart(b, list(board.index), "lb_ceiling")
    fig = go.Figure()
    for m in shown:
        err = [[(b.ci(target, m, k) or (np.nan, np.nan))[i] for k in keys] for i in (0, 1)]
        vals = board.loc[m, keys].to_numpy(float)
        fig.add_bar(name=m, x=[LABELS[k] for k in keys], y=vals, marker_color=_color(m),
                    error_y=dict(type="data", symmetric=False, array=np.array(err[1]) - vals,
                                 arrayminus=vals - np.array(err[0]), thickness=1))
    fig.update_layout(barmode="group", yaxis_title="score", xaxis_tickangle=-30)
    st.plotly_chart(_style(fig, 460, legend_y=-0.42), width="stretch")
    un = b.unavailable(target)
    if un:
        with st.expander(f"Metrics not available ({len(un)})"):
            for u in un:
                st.write(f"- {u}")


# --------------------------------------------------------------------------------------- прогноз связей
def links(b: Bundle) -> None:
    ctx, audit = b.link_context(), b.audit()
    board = b.board("test", ["link_roc_auc", "link_learned_roc_auc", "link_new_roc_auc", "link_pr_auc_lift"])
    if board["link_roc_auc"].isna().all():
        st.warning("Future-link candidates (edges.csv) are not part of this run.")
        return
    c = st.columns(4)
    c[0].metric("Positive rate in the pool", f"{ctx['sampled_positive_rate']:.2f}")
    c[1].metric("Natural prevalence", f"{ctx['natural_prevalence']:.3f}")
    c[2].metric("Positives already linked in source", f"{ctx['share_positives_seen_in_source']:.0%}")
    c[3].metric("Persistence predictor ROC-AUC", f"{ctx['persistence_roc_auc']:.3f}")
    shown = _methods_for_chart(b, list(board.index), "link_ceiling")
    fig = go.Figure()
    for key, name, col in (("link_roc_auc", "cosine similarity", "#9aa0a6"), ("link_learned_roc_auc", "learned edge classifier", "#1a73e8")):
        fig.add_bar(name=name, x=shown, y=board.loc[shown, key], marker_color=col)
    ref_lines = [("chance (0.5)", 0.5, "#d93025", "dot")]
    if ctx["persistence_roc_auc"]:
        ref_lines.append(("persistence: pair seen in source", ctx["persistence_roc_auc"], "#188038", "dash"))
    for name, y, col, dash in ref_lines:
        fig.add_hline(y=y, line_dash=dash, line_color=col)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=name, line=dict(color=col, dash=dash)))
    if audit and "link_learned_roc_auc" in audit["random_null"]["metrics"]:
        n = audit["random_null"]["metrics"]["link_learned_roc_auc"]
        fig.add_hrect(y0=n["mean"] - 2 * n["std"], y1=n["mean"] + 2 * n["std"], fillcolor="#1a73e8", opacity=0.12, line_width=0)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="random embeddings, learned scorer (±2 sd)",
                                 marker=dict(symbol="square", size=12, color="rgba(26,115,232,0.25)")))
    fig.update_layout(barmode="group", yaxis=dict(range=[0.35, 0.85], title="ROC-AUC"))
    st.plotly_chart(_style(fig, 480, legend_y=-0.28), width="stretch")
    st.markdown(
        "**How to read this.** Candidates are pairs that did (positive) or did not (negative) communicate in the "
        "later window. *Cosine* is a parameter-free score and is **below chance** for client-server traffic: linked "
        "hosts are dissimilar while look-alike clients are never linked. The *learned* scorer fits a classifier on "
        "`[|a-b|, a*b]` using the earlier validation candidates and is scored on test. Random embeddings are "
        "above 0.5 under the learned scorer (they act as host identifiers and the classifier memorises hubs), so "
        "always compare against the random band and the persistence line. PR-AUC depends on prevalence: use "
        "lift or the natural-prevalence value.")
    st.dataframe(_text_table(board.rename(columns=LABELS)), width="stretch")


# --------------------------------------------------------------------------------------- соседство
@st.cache_data(show_spinner=False)
def _projection(_b: Bundle, key: str, method: str, k: int) -> pd.DataFrame:
    return _b.projection(method, k)


def neighborhoods(b: Bundle) -> None:
    emb_methods = [m for m in b.methods if (b.run / "embeddings" / m / "train.csv").exists()]
    if not emb_methods or not b.has_neighbors():
        st.warning("This bundle has no saved embeddings / reference neighbours, so host-level views are unavailable.")
        return
    c = st.columns([2, 1, 3])
    method = c[0].selectbox("Method", emb_methods, index=emb_methods.index("behavioral") if "behavioral" in emb_methods else 0)
    k = c[1].select_slider("k", [5, 10, 20], value=10)
    proj = _projection(b, str(b.root), method, k)
    hosts = sorted(proj.index)
    best = proj["agreement"].idxmax()
    host = c[2].selectbox("Host", hosts, index=hosts.index(best),
                          help="Compare a host's embedding neighbours with the frozen reference. "
                               "Default: the host with the highest agreement for this method.")
    nb, ref = b.neighborhood(method, host, k), b.reference_list(host, k)
    overlap = int(nb["in_reference_top_k"].sum())
    st.caption(f"Mean per-host agreement of **{method}** at k={k}: {proj['agreement'].mean():.3f}. "
               f"For **{host}**: {overlap}/{k} embedding neighbours are also reference neighbours "
               f"(agreement of this host: {proj.loc[host, 'agreement']:.2f}).")
    d = proj.reset_index()
    fig = px.scatter(d, x="PC1", y="PC2", color="agreement", hover_name="host_id", range_color=[0, 1],
                     color_continuous_scale="Viridis", opacity=0.75)
    for ids, name, sym, col in ((list(nb.host_id), "embedding neighbours", "circle-open", "#e8710a"),
                                (list(ref.host_id), "reference neighbours", "diamond-open", "#d93025")):
        s = proj.loc[ids]
        fig.add_trace(go.Scatter(x=s.PC1, y=s.PC2, mode="markers", name=name, text=ids, hoverinfo="text",
                                 marker=dict(symbol=sym, size=14, line=dict(width=2, color=col), color=col)))
    s = proj.loc[[host]]
    fig.add_trace(go.Scatter(x=s.PC1, y=s.PC2, mode="markers", name=host, hoverinfo="name",
                             marker=dict(symbol="star", size=18, color="black")))
    fig.update_coloraxes(colorbar=dict(title="agreement", thickness=12, len=0.8))
    st.plotly_chart(_style(fig, 520), width="stretch")
    t1, t2 = st.columns(2)
    nbt = nb.copy()
    nbt["in_reference_top_k"] = nbt["in_reference_top_k"].map({True: "yes", False: "no"})
    nbt["reference_rank"] = nbt["reference_rank"].map(lambda v: "-" if pd.isna(v) else str(int(v)))
    nbt["cosine"] = nbt["cosine"].map(lambda v: f"{v:.3f}")
    t1.markdown("**Embedding neighbours** (cosine)")
    t1.dataframe(nbt, width="stretch", hide_index=True)
    t2.markdown("**Frozen reference neighbours** (later-window behaviour)")
    t2.dataframe(ref, width="stretch", hide_index=True)
    st.caption("The 2D PCA projection is a lossy view of the scored embedding; distances in the plane are not the "
               "distances used by the metrics. Colour = per-host Consistency@k against the frozen reference.")


# --------------------------------------------------------------------------------------- время
def temporal(b: Bundle) -> None:
    st.markdown("##### Validation vs test")
    keys = [k for g in METRIC_GROUPS.values() for k, _ in g]
    avail = [k for k in keys if not b.board("test", [k]).isna().all().all()]
    metric = st.selectbox("Metric", avail, format_func=lambda k: LABELS[k], index=avail.index("consistency@10_mean"))
    if "validation" in b.targets:
        d = pd.concat([b.board(t, [metric])[metric].rename(t) for t in ("validation", "test")], axis=1)
        d = d.loc[_methods_for_chart(b, list(d.index), "temp_ceiling")]
        fig = go.Figure()
        for t, col in (("validation", "#9aa0a6"), ("test", "#1a73e8")):
            fig.add_bar(name=t, x=list(d.index), y=d[t], marker_color=col)
        if metric in CHANCE:
            fig.add_hline(y=CHANCE[metric], line_dash="dot", line_color="#d93025", annotation_text="chance")
        fig.update_layout(barmode="group")
        st.plotly_chart(_style(fig, 340), width="stretch")
        audit = b.audit()
        if audit and audit["split_agreement"].get(metric) is not None:
            st.caption(f"Ranking agreement between validation and test (Kendall tau): {audit['split_agreement'][metric]:.2f}. "
                       "Where every method is near chance the ordering is noise.")
    st.markdown("##### Host identity across windows (train → test)")
    ident = b.identity("test")
    if ident:
        fig = go.Figure()
        methods = list(ident)
        raw = [ident[m]["none"].get("top1") if ident[m].get("shared_coordinate_system_declared") else None for m in methods]
        pro = [ident[m]["procrustes"].get("top1") for m in methods]
        fig.add_bar(name="raw coordinates (shared-coordinate methods only)", x=methods, y=raw, marker_color="#1a73e8")
        fig.add_bar(name="Procrustes-aligned", x=methods, y=pro, marker_color="#e8710a")
        chance = ident[methods[0]]["procrustes"]["chance_top1"]
        fig.add_hline(y=chance, line_dash="dot", line_color="#d93025", annotation_text="chance")
        fig.update_layout(barmode="group", yaxis_title="Top-1 identity retrieval")
        st.plotly_chart(_style(fig, 340), width="stretch")
        st.caption("Same host ID in two windows is the judgment: this measures stability *and* discriminability, not "
                   "behavioural similarity. Independently fitted embeddings rotate between windows, so raw cosine is "
                   "meaningless for them; alignment is fitted on one half of the hosts and scored on the other.")
    audit = b.audit()
    if audit:
        st.markdown("##### How much does behaviour itself drift?")
        s = audit["reference_stability"]
        d = pd.DataFrame({p: v for p, v in s["pairs"].items()}).T
        chron = {"train": 0, "validation": 1, "test": 2}
        d = d.loc[sorted(d.index, key=lambda p: tuple(chron[w] for w in p.split("->")))]
        d.index.name = "windows"
        st.dataframe(_text_table(d), width="stretch")
        st.caption(f"Agreement of behavioural neighbour lists built from two windows (chance Consistency@{s['k']} = "
                   f"{s['chance_consistency']:.3f}). It bounds what an earlier-window embedding can reach on the kNN metric.")


# --------------------------------------------------------------------------------------- аудит
def audit_view(b: Bundle) -> None:
    a = b.audit()
    if not a:
        st.warning("No audit.json in this bundle (run without --no-audit).")
        return
    st.markdown("##### Chance bands (random embeddings, test reference)")
    rows = {LABELS.get(k, k): v for k, v in a["random_null"]["metrics"].items() if k in LABELS}
    nulls = pd.DataFrame(rows).T[["mean", "std", "p5", "p95"]]
    st.dataframe(_text_table(nulls), width="stretch", height=_rows(nulls))
    st.caption(f"{a['random_null']['n_draws']} random draws. Use these bands, not 0.5, to judge significance.")
    w = a["weight_sensitivity"]
    st.markdown("##### Sensitivity of the behavioural proxy to feature weights")
    c = st.columns(3)
    c[0].metric("Mean neighbour agreement", f"{w['mean_reference_agreement']:.2f}", help=f"chance {w['chance_agreement']:.3f}")
    c[1].metric("Mean ranking Kendall tau", f"{w['mean_kendall_tau']:.2f}")
    c[2].metric("Worst ranking tau", f"{w['min_kendall_tau']:.2f}")
    cfg = pd.DataFrame(w["configs"])
    fig = px.bar(cfg, x="config", y="kendall_tau", color=cfg["dropped_block"].fillna("all blocks"),
                 labels={"kendall_tau": "Kendall tau of method ranking", "config": "weight configuration", "color": "removed block"})
    fig.update_layout(yaxis_range=[-1, 1])
    st.plotly_chart(_style(fig, 300), width="stretch")
    pv = a.get("prevalence")
    if pv:
        st.markdown("##### Prevalence of the candidate pools")
        fl, pr = pv["future_links"], pv["pairs"]
        rows = {"future links": {"in the pool": fl["sampled_positive_rate"], "natural (all pairs)": fl["natural_prevalence"]},
                "shared-peer pairs": {"in the pool": pr["sampled_positive_rate"],
                                      "among judged pairs": pr.get("judged_population_prevalence"),
                                      "among all pairs": pr.get("population_positive_rate")}}
        st.dataframe(_text_table(pd.DataFrame(rows).T), width="stretch")
        st.caption(pr["note"][0].upper() + pr["note"][1:] + ". For links every pair is either linked or not, so the natural prevalence is a "
                   "population value; for shared-peer pairs the judged prevalence is conditional.")
    st.markdown("##### Information shared by construction")
    ov = pd.DataFrame(a["information_overlap"]).T
    ov.index.name = "method"
    st.table(ov)                              # длинный текст переносится, а не обрезается
    st.caption("A method whose input overlaps a reference's source scores well on it for a trivial reason. "
               "These rows are calibration points, not independent evidence.")


# --------------------------------------------------------------------------------------- происхождение
def provenance(b: Bundle) -> None:
    m = b.manifest
    st.markdown("##### Run")
    st.json({k: m[k] for k in ("data_kind", "synthetic", "n_hosts", "config", "methods", "external_methods",
                               "missing_expected_methods", "environment")}, expanded=False)
    st.markdown("##### Reference artifacts")
    for t in b.targets:
        ref = b.reference_manifest(t)
        if not ref:
            continue
        with st.expander(f"reference/{t}"):
            st.write(f"kind: `{ref.get('kind')}` · ground truth: `{ref.get('ground_truth')}` · "
                     f"provenance verified: `{ref.get('provenance_verified')}`")
            st.write(f"source window: `{ref.get('source_window')}` → target window: `{ref.get('target_window')}`")
            st.json(ref.get("files_sha256", {}), expanded=False)
            if ref.get("judgments"):
                st.json(ref["judgments"], expanded=False)
            if ref.get("reference_views"):
                st.json(ref["reference_views"], expanded=False)
    st.caption("Every result carries SHA-256 fingerprints of all reference files; the leaderboard builder refuses to "
               "mix different references, cohorts, splits, or synthetic and real data.")
