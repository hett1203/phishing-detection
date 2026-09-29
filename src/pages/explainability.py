"""Explainability page — SHAP global summary + feature importance."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from predict import load_best_meta, load_shap_artifacts
from theme import render_metric_card

import os
import joblib
from pathlib import Path


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
            <h1><span class="page-icon icon-yellow"><span class="material-symbols-rounded">lightbulb</span></span> Explainability</h1>
            <div class="subtitle">SHAP global feature importance — what drives the model's decisions</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    meta = load_best_meta()
    feature_names = meta["feature_columns"]

    # Load SHAP artifacts (may not exist if SHAP step failed at training time)
    shap_path = Path(__file__).resolve().parent.parent.parent / "artifacts" / "shap"
    if not (shap_path / "shap_values.npy").exists():
        st.markdown(
            """
            <div class="alert-banner alert-info">
            SHAP values were not computed during training. Re-run <code>python src/train.py</code> to regenerate them.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    shap_values = np.load(shap_path / "shap_values.npy")
    sample_X = pd.read_csv(shap_path / "shap_sample.csv")

    # Align columns to feature_names (in case the saved sample has a different order)
    if list(sample_X.columns) != feature_names:
        # Reorder — the saved sample may include TLD dummy columns already
        common = [c for c in feature_names if c in sample_X.columns]
        sample_X = sample_X[common]
        shap_values = shap_values[:, :len(common)]
        feature_names_used = common
    else:
        feature_names_used = feature_names

    # ---- KPI ----
    st.markdown('<div class="section-divider"><h2>SHAP Summary</h2></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown(render_metric_card("Samples Explained", f"{shap_values.shape[0]:,}",
                                    "Held-out test sample", "info"), unsafe_allow_html=True)
    c2.markdown(render_metric_card("Features", f"{shap_values.shape[1]}",
                                    "Model-ready features", "info"), unsafe_allow_html=True)
    c3.markdown(render_metric_card("Best Model", meta["best_model"],
                                    "Currently deployed", "success"), unsafe_allow_html=True)

    # ---- Mean |SHAP| bar ----
    st.markdown('<div class="section-divider"><h2>Global Feature Importance — Mean(|SHAP|)</h2></div>', unsafe_allow_html=True)
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature": feature_names_used,
        "Mean_Abs_SHAP": mean_abs,
    }).sort_values("Mean_Abs_SHAP", ascending=True).tail(20)

    fig = px.bar(
        importance_df, x="Mean_Abs_SHAP", y="Feature", orientation="h",
        color="Mean_Abs_SHAP",
        color_continuous_scale=["#0d1320", "#00d4ff"],
    )
    _dark_layout(fig)
    fig.update_layout(height=600, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # ---- SHAP beeswarm-style scatter (top 10 features) ----
    st.markdown('<div class="section-divider"><h2>SHAP Beeswarm — Top 10 Features</h2></div>', unsafe_allow_html=True)
    top10_idx = np.argsort(mean_abs)[-10:][::-1]

    rows = []
    for fi in top10_idx:
        feat_name = feature_names_used[fi]
        feats_vals = sample_X.iloc[:, fi].values
        shap_vals = shap_values[:, fi]
        for fv, sv in zip(feats_vals, shap_vals):
            rows.append({"Feature": feat_name, "Feature_Value": float(fv), "SHAP_Value": float(sv)})
    bee_df = pd.DataFrame(rows)

    fig2 = px.scatter(
        bee_df, x="SHAP_Value", y="Feature", color="Feature_Value",
        color_continuous_scale=["#2ed573", "#0d1320", "#ff4757"],
        opacity=0.55, size_max=8,
    )
    _dark_layout(fig2)
    fig2.update_layout(height=550, coloraxis_colorbar=dict(title="Feature\nValue"))
    st.plotly_chart(fig2, use_container_width=True)

    # ---- Interpretation panel ----
    st.markdown('<div class="section-divider"><h2>Interpreting the Results</h2></div>', unsafe_allow_html=True)
    top_features = importance_df.sort_values("Mean_Abs_SHAP", ascending=False).head(5)["Feature"].tolist()
    explain_html = []
    for feat in top_features:
        display = feat.replace("tld_", "TLD: ") if feat.startswith("tld_") else feat.replace("_", " ").title()
        explain_html.append(
            f"<li><strong>{display}</strong> — among the strongest signals driving the model toward a phishing verdict.</li>"
        )
    st.markdown(
        f"""
        <div class="metric-card info" style="padding:1.5rem 2rem;">
            <div style="font-size:0.95rem; color:#c9d1d9; line-height:1.7; margin-bottom:1rem;">
            <strong style="color:#00d4ff">Top contributing features</strong> are those whose SHAP values
            consistently shift the model's output. Higher <em>mean |SHAP|</em> means the feature
            drives more of the model's decisions across the test set.
            </div>
            <ul class="explain-list">{"".join(explain_html)}</ul>
            <div style="font-size:0.85rem; color:#8b949e; margin-top:1rem; line-height:1.6;">
            A <strong>red point</strong> in the beeswarm = high feature value;
            a <strong>green point</strong> = low feature value.
            Positive SHAP → pushes model toward <span style="color:#ff4757">phishing</span>;
            negative SHAP → pushes toward <span style="color:#2ed573">safe</span>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
