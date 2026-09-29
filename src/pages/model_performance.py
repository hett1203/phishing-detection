"""Model Performance Dashboard — comparison table + ROC + confusion matrices."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from predict import load_metrics, load_roc_curves, load_confusion_matrices, load_best_meta
from theme import render_metric_card


def _dark_layout(fig) -> None:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13, 19, 32, 0.6)",
        font=dict(color="#e6edf3", family="Inter, sans-serif"),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )


# Color palette for ROC curves
COLORS = ["#00d4ff", "#2ed573", "#ff4757", "#fbbf24", "#a78bfa",
          "#f472b6", "#22d3ee", "#84cc16", "#fb923c", "#94a3b8"]


def render() -> None:
    st.markdown(
        """
        <div class="page-hero fade-in">
            <h1><span class="page-icon icon-red"><span class="material-symbols-rounded">monitoring</span></span> Model Performance</h1>
            <div class="subtitle">Side-by-side comparison of all trained classifiers</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_metrics()
    roc = load_roc_curves()
    cms = load_confusion_matrices()
    meta = load_best_meta()
    best_name = meta["best_model"]

    # ---- Best model KPI ----
    best_row = df[df["model"] == best_name].iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(render_metric_card("Best Model", best_name, f"{best_row['train_seconds']:.1f}s train time", "success"), unsafe_allow_html=True)
    c2.markdown(render_metric_card("Accuracy", f"{best_row['accuracy']*100:.2f}%", "Overall", "info"), unsafe_allow_html=True)
    c3.markdown(render_metric_card("Phishing Recall", f"{best_row['recall_phish']*100:.2f}%", "Most important", "danger"), unsafe_allow_html=True)
    c4.markdown(render_metric_card("F1 Score", f"{best_row['f1_phish']*100:.2f}%", "Phishing class", "info"), unsafe_allow_html=True)
    c5.markdown(render_metric_card("ROC-AUC", f"{best_row['roc_auc']:.4f}", "Discrimination", "info"), unsafe_allow_html=True)

    # ---- Comparison table ----
    st.markdown('<div class="section-divider"><h2>Comparison Table</h2></div>', unsafe_allow_html=True)
    display_df = df.copy()
    # Multiply rate metrics by 100 for display
    for col in ["accuracy", "precision_phish", "recall_phish", "f1_phish"]:
        display_df[col] = (display_df[col] * 100).round(2)
    display_df["roc_auc"] = display_df["roc_auc"].round(4)
    display_df["pr_auc"] = display_df["pr_auc"].round(4)
    display_df["mcc"] = display_df["mcc"].round(4)
    display_df["log_loss"] = display_df["log_loss"].round(4)
    rename = {
        "accuracy": "Accuracy (%)",
        "precision_phish": "Precision_Phish (%)",
        "recall_phish": "Recall_Phish (%)",
        "f1_phish": "F1_Phish (%)",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "mcc": "MCC",
        "log_loss": "Log Loss",
        "train_seconds": "Train (s)",
        "tp": "TP", "fp": "FP", "tn": "TN", "fn": "FN",
    }
    display_df = display_df.rename(columns=rename)

    # Highlight best model
    def _highlight_best(row):
        if row["model"] == best_name:
            return ["background-color: rgba(46, 213, 115, 0.15); color: #2ed573; font-weight:700"] * len(row)
        return [""] * len(row)

    styled = display_df.style.apply(_highlight_best, axis=1)
    st.dataframe(styled, use_container_width=True, height=400, hide_index=True)

    # ---- Tabs for ROC + Confusion ----
    st.markdown('<div class="section-divider"><h2>Visual Comparison</h2></div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["ROC Curves", "Confusion Matrices", "Metric Bar Charts"])

    # ---- ROC ----
    with tab1:
        fig = go.Figure()
        for i, (name, data) in enumerate(roc.items()):
            color = COLORS[i % len(COLORS)]
            lw = 3 if name == best_name else 1.5
            fig.add_trace(go.Scatter(
                x=data["fpr"], y=data["tpr"],
                mode="lines",
                name=f"{name} (AUC={data['auc']:.4f})",
                line=dict(color=color, width=lw),
            ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Random (AUC=0.5)",
            line=dict(color="#6e7681", dash="dash", width=1),
        ))
        _dark_layout(fig)
        fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=550,
            legend=dict(orientation="v", xanchor="left", x=1.02, yanchor="top", y=1,
                        font=dict(size=11), bgcolor="rgba(13,19,32,0.8)"),
            title=dict(text="ROC Curves — All Models", x=0.5, font=dict(size=16)),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---- Confusion matrices ----
    with tab2:
        model_names = list(cms.keys())
        n = len(model_names)
        # Grid of confusion matrix heatmaps
        cols = st.columns(min(3, n))
        for i, name in enumerate(model_names):
            cm = np.array(cms[name])
            col = cols[i % len(cols)]
            with col:
                fig = go.Figure(data=go.Heatmap(
                    z=cm[::-1],
                    x=["Pred Safe", "Pred Phishing"],
                    y=["True Phishing", "True Safe"],
                    colorscale=[[0, "#0d1320"], [0.5, "#0096c7"], [1, "#00d4ff"]],
                    showscale=False,
                    text=cm[::-1],
                    texttemplate="%{text}",
                    textfont=dict(color="#e6edf3", size=14),
                ))
                _dark_layout(fig)
                fig.update_layout(
                    title=dict(text=name, x=0.5, font=dict(size=12, color="#e6edf3")),
                    height=300,
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

    # ---- Bar charts ----
    with tab3:
        metrics_to_plot = ["accuracy", "recall_phish", "f1_phish", "roc_auc"]
        col_a, col_b = st.columns(2)
        for i, metric in enumerate(metrics_to_plot):
            target_col = col_a if i % 2 == 0 else col_b
            with target_col:
                plot_df = df.sort_values(metric, ascending=True).copy()
                colors = ["#2ed573" if m == best_name else "#00d4ff" for m in plot_df["model"]]
                fig = go.Figure(data=[go.Bar(
                    x=plot_df[metric],
                    y=plot_df["model"],
                    orientation="h",
                    marker_color=colors,
                    text=plot_df[metric].round(4),
                    textposition="outside",
                    textfont=dict(color="#e6edf3"),
                )])
                _dark_layout(fig)
                fig.update_layout(
                    title=dict(text=metric.upper().replace("_", " "), x=0.5,
                              font=dict(size=12, color="#e6edf3")),
                    height=400,
                    xaxis=dict(range=[0, 1.05]),
                )
                st.plotly_chart(fig, use_container_width=True)
