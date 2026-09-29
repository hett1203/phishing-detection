"""EDA Dashboard — correlation heatmap + top discriminative features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

from theme import render_metric_card

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "Dataset.csv"
NUMERIC_COLS = [
    "url_len", "dom_len", "is_ip", "tld_len", "subdom_cnt",
    "letter_cnt", "digit_cnt", "special_cnt", "eq_cnt", "qm_cnt",
    "amp_cnt", "dot_cnt", "dash_cnt", "under_cnt",
    "letter_ratio", "digit_ratio", "spec_ratio", "is_https",
    "slash_cnt", "entropy", "path_len", "query_len",
]


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH).dropna().reset_index(drop=True)


@st.cache_data(show_spinner=False)
def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = NUMERIC_COLS + ["label"]
    return df[cols].corr()


def _dark_layout(fig) -> None:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13, 19, 32, 0.6)",
        font=dict(color="#e6edf3", family="Inter, sans-serif"),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )


def render() -> None:
    st.markdown(
        """
        <div class="page-hero fade-in">
            <h1><span class="page-icon icon-green"><span class="material-symbols-rounded">analytics</span></span> EDA Dashboard</h1>
            <div class="subtitle">Correlations, discriminative features, and phishing signal patterns</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_dataset()
    corr = correlation_matrix(df)

    # ---- Top correlated features with label ----
    st.markdown('<div class="section-divider"><h2>Correlation with Phishing Label</h2></div>', unsafe_allow_html=True)
    label_corr = corr["label"].drop("label").sort_values(key=lambda s: s.abs(), ascending=False).reset_index()
    label_corr.columns = ["Feature", "Correlation"]

    col_l, col_r = st.columns([1, 1.4])
    with col_l:
        st.markdown(
            """
            <div class="metric-card info" style="padding:1rem 1.2rem;">
                <div style="font-size:0.85rem; color:#8b949e; margin-bottom:0.5rem;">TOP 8 DISCRIMINATIVE FEATURES</div>
            """,
            unsafe_allow_html=True,
        )
        for _, row in label_corr.head(8).iterrows():
            color = "#ff4757" if row["Correlation"] > 0 else "#2ed573"
            direction = "phishing ↑" if row["Correlation"] > 0 else "phishing ↓"
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; padding:0.5rem 0; border-bottom:1px solid rgba(255,255,255,0.05);">
                    <span style="color:#e6edf3; font-weight:600;">{row['Feature']}</span>
                    <span style="color:{color}; font-weight:700;">{row['Correlation']:+.3f}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
    with col_r:
        fig = px.bar(
            label_corr.head(15), x="Correlation", y="Feature", orientation="h",
            color="Correlation",
            color_continuous_scale=["#2ed573", "#0d1320", "#ff4757"],
            range_color=[-0.5, 0.5],
        )
        _dark_layout(fig)
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # ---- Full correlation heatmap ----
    st.markdown('<div class="section-divider"><h2>Feature Correlation Heatmap</h2></div>', unsafe_allow_html=True)
    corr_no_label = corr.drop(columns=["label"], errors="ignore").drop(index="label", errors="ignore")

    fig2 = go.Figure(data=go.Heatmap(
        z=corr_no_label.values,
        x=corr_no_label.columns,
        y=corr_no_label.index,
        colorscale=[[0, "#2ed573"], [0.5, "#0d1320"], [1, "#ff4757"]],
        zmin=-1, zmax=1,
        hovertemplate="%{y} ↔ %{x}<br>ρ = %{z:.3f}<extra></extra>",
        colorbar=dict(title="ρ", tickcolor="#e6edf3"),
    ))
    _dark_layout(fig2)
    fig2.update_layout(height=720, xaxis=dict(tickangle=-45), font=dict(size=9))
    st.plotly_chart(fig2, use_container_width=True)

    # ---- Mean by class for top features ----
    st.markdown('<div class="section-divider"><h2>Class-Wise Feature Means</h2></div>', unsafe_allow_html=True)
    means = df.groupby("label")[NUMERIC_COLS].mean().T
    means.columns = ["Legitimate", "Phishing"]
    means["Ratio"] = (means["Phishing"] / means["Legitimate"].replace(0, np.nan)).round(2)
    means = means.reset_index().rename(columns={"index": "Feature"})

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        name="Legitimate", x=means["Feature"], y=means["Legitimate"],
        marker_color="#2ed573", opacity=0.85,
    ))
    fig3.add_trace(go.Bar(
        name="Phishing", x=means["Feature"], y=means["Phishing"],
        marker_color="#ff4757", opacity=0.85,
    ))
    _dark_layout(fig3)
    fig3.update_layout(barmode="group", height=450, xaxis=dict(tickangle=-45),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig3, use_container_width=True)

    # ---- Stats table ----
    st.markdown('<div class="section-divider"><h2>Numeric Statistics</h2></div>', unsafe_allow_html=True)
    stats = df[NUMERIC_COLS].describe().T.reset_index()
    stats.columns = ["Feature", "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    st.dataframe(stats.style.format({
        "Mean": "{:.3f}", "Std": "{:.3f}", "Min": "{:.3f}",
        "25%": "{:.3f}", "50%": "{:.3f}", "75%": "{:.3f}", "Max": "{:.3f}",
    }), use_container_width=True, height=500)
