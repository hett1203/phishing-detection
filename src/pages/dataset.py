"""Dataset Overview page — class balance + per-feature distributions."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

from theme import render_metric_card

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "Dataset.csv"


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH).dropna().reset_index(drop=True)
    return df


def _dark_layout(fig) -> None:
    """Apply the dark cybersecurity theme to a plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13, 19, 32, 0.6)",
        font=dict(color="#e6edf3", family="Inter, sans-serif"),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
    )


def render() -> None:
    st.markdown(
        """
        <div class="page-hero fade-in">
            <h1><span class="page-icon icon-cyan"><span class="material-symbols-rounded">database</span></span> Dataset Overview</h1>
            <div class="subtitle">Class balance, feature distributions, and dataset at a glance</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_dataset()

    # ---- KPI strip ----
    st.markdown('<div class="section-divider"><h2>Dataset Snapshot</h2></div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(render_metric_card("Total URLs", f"{len(df):,}", "After dedup + dropna", "info"), unsafe_allow_html=True)
    c2.markdown(render_metric_card("Features", f"{df.shape[1] - 1}", "Numeric + categorical", "info"), unsafe_allow_html=True)
    c3.markdown(render_metric_card("Legitimate", f"{(df['label']==0).sum():,}", "~86%", "success"), unsafe_allow_html=True)
    c4.markdown(render_metric_card("Phishing", f"{(df['label']==1).sum():,}", "~14%", "danger"), unsafe_allow_html=True)
    c5.markdown(render_metric_card("Missing", "0", "Cleaned dataset", "success"), unsafe_allow_html=True)

    # ---- Class balance ----
    st.markdown('<div class="section-divider"><h2>Class Balance</h2></div>', unsafe_allow_html=True)
    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        counts = df["label"].value_counts().rename(index={0: "Legitimate", 1: "Phishing"}).reset_index()
        counts.columns = ["Class", "Count"]
        fig = px.bar(
            counts, x="Class", y="Count",
            color="Class",
            color_discrete_map={"Legitimate": "#2ed573", "Phishing": "#ff4757"},
            text="Count",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside", textfont=dict(color="#e6edf3"))
        _dark_layout(fig)
        fig.update_yaxes(title="Count")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig2 = px.pie(
            counts, values="Count", names="Class",
            color="Class",
            color_discrete_map={"Legitimate": "#2ed573", "Phishing": "#ff4757"},
            hole=0.55,
        )
        _dark_layout(fig2)
        fig2.update_traces(textfont=dict(color="#e6edf3", size=13))
        st.plotly_chart(fig2, use_container_width=True)

    # ---- Feature picker for distribution ----
    st.markdown('<div class="section-divider"><h2>Feature Distribution by Class</h2></div>', unsafe_allow_html=True)
    numeric_cols = [
        c for c in df.columns
        if c not in ("url", "dom", "tld", "label") and df[c].dtype.kind in "iuf"
    ]
    default_feat = "url_len" if "url_len" in numeric_cols else numeric_cols[0]
    selected = st.selectbox("Choose a feature to inspect", numeric_cols, index=numeric_cols.index(default_feat))

    col_l, col_r = st.columns([1.3, 1])
    with col_l:
        df_plot = df.copy()
        df_plot["Class"] = df_plot["label"].map({0: "Legitimate", 1: "Phishing"})
        fig3 = px.histogram(
            df_plot, x=selected, color="Class",
            color_discrete_map={"Legitimate": "#2ed573", "Phishing": "#ff4757"},
            nbins=60, opacity=0.7, marginal="violin",
        )
        _dark_layout(fig3)
        fig3.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig3, use_container_width=True)
    with col_r:
        # Box plot side-by-side
        fig4 = px.box(
            df_plot, x="Class", y=selected,
            color="Class",
            color_discrete_map={"Legitimate": "#2ed573", "Phishing": "#ff4757"},
        )
        _dark_layout(fig4)
        st.plotly_chart(fig4, use_container_width=True)

    # ---- TLD top-20 ----
    st.markdown('<div class="section-divider"><h2>Top-20 Top-Level Domains</h2></div>', unsafe_allow_html=True)
    top_tlds = df["tld"].value_counts().head(20).reset_index()
    top_tlds.columns = ["TLD", "Count"]
    fig5 = px.bar(
        top_tlds, x="TLD", y="Count",
        color_discrete_sequence=["#00d4ff"],
    )
    _dark_layout(fig5)
    fig5.update_xaxes(tickangle=-45)
    st.plotly_chart(fig5, use_container_width=True)

    # ---- Sample data preview ----
    st.markdown('<div class="section-divider"><h2>Sample Rows</h2></div>', unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True, height=400)
