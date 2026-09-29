#!/usr/bin/env python3
"""
app.py - Phishing Website Detection - Streamlit Dashboard
=========================================================
Professional dark cybersecurity-themed Streamlit application.

Pages
-----
1. Home / Landing
2. About Project
3. Dataset Overview
4. EDA Dashboard
5. Behavior Clusters (optional/interpretive)
6. Single Website Checker
7. Batch Prediction (CSV upload -> predicted_file.csv schema)
8. Feature Importance & Explainability (SHAP + LIME)
9. Model Performance Dashboard
10. Download Prediction Results
"""
from __future__ import annotations

import io
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st
import yaml

# Project root on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import (
    get_config, get_logger, setup_logging, load_joblib,
    to_abs, map_numeric_to_label, map_label_to_numeric,
)
from src.utils.logger import setup_logging as _setup_logging
from src.features.engineering import FeatureEngineer
from src.features.selection import FeatureSelector
from src.prediction import PredictionPipeline

# Initialize once
_setup_logging()
log = get_logger("streamlit_app")
CFG = get_config()

# ----------------------------- Styling ----------------------------- #

THEME = {
    "bg": "#0A0E14",
    "bg_panel": "#11161F",
    "bg_card": "#161D28",
    "border": "#1F2937",
    "primary": "#00E5A8",   # neon green
    "accent": "#FF3B3B",    # red
    "warning": "#FFB020",
    "text": "#E5E7EB",
    "text_muted": "#9CA3AF",
    "phishing": "#FF3B3B",
    "safe": "#00E5A8",
    "neon_green": "#00E676",
    "neon_blue": "#00B8FF",
    "neon_purple": "#9D4EDD",
}


def _inject_css() -> None:
    st.markdown(
        f"""
        <style>
        /* ---------- Global ---------- */
        html, body, [data-testid="stAppViewContainer"], 
        [data-testid="stSidebar"], [data-testid="stHeader"] {{
            background-color: {THEME["bg"]} !important;
            color: {THEME["text"]} !important;
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0A0E14 0%, #0F141C 100%) !important;
            border-right: 1px solid {THEME["border"]} !important;
        }}
        [data-testid="stSidebar"] * {{
            color: {THEME["text"]} !important;
        }}
        .stApp {{
            background: radial-gradient(ellipse at top left,
                rgba(0, 229, 168, 0.06) 0%, transparent 50%),
                radial-gradient(ellipse at bottom right,
                rgba(255, 59, 59, 0.06) 0%, transparent 50%),
                {THEME["bg"]} !important;
        }}
        /* ---------- Text ---------- */
        h1, h2, h3, h4, h5, h6 {{
            color: {THEME["text"]} !important;
            font-weight: 700 !important;
            letter-spacing: 0.02em;
        }}
        h1 {{ font-size: 2.0rem !important; 
              border-bottom: 2px solid {THEME["primary"]} !important;
              padding-bottom: 0.4rem !important; }}
        h2 {{ font-size: 1.5rem !important; }}
        h3 {{ font-size: 1.2rem !important; }}
        p, li, span, label {{
            color: {THEME["text"]} !important;
        }}
        .muted {{ color: {THEME["text_muted"]} !important; }}
        /* ---------- Components ---------- */
        [data-testid="stButton"] button {{
            background: linear-gradient(135deg, {THEME["primary"]} 0%,
                {THEME["neon_blue"]} 100%) !important;
            color: #0A0E14 !important;
            font-weight: 700 !important;
            border: none !important;
            padding: 0.6rem 1.4rem !important;
            border-radius: 6px !important;
            box-shadow: 0 4px 14px rgba(0, 229, 168, 0.25) !important;
            transition: all 0.2s ease !important;
        }}
        [data-testid="stButton"] button:hover {{
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 20px rgba(0, 229, 168, 0.4) !important;
        }}
        [data-testid="stDownloadButton"] button {{
            background: linear-gradient(135deg, {THEME["accent"]} 0%,
                {THEME["warning"]} 100%) !important;
            color: #0A0E14 !important;
            font-weight: 700 !important;
            border: none !important;
            padding: 0.55rem 1.2rem !important;
            border-radius: 6px !important;
        }}
        /* Metric cards */
        [data-testid="stMetric"] {{
            background: {THEME["bg_card"]} !important;
            border: 1px solid {THEME["border"]} !important;
            border-radius: 10px !important;
            padding: 1rem 1.2rem !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
        }}
        [data-testid="stMetricValue"] {{
            color: {THEME["primary"]} !important;
            font-weight: 800 !important;
        }}
        [data-testid="stMetricLabel"] {{
            color: {THEME["text_muted"]} !important;
            font-size: 0.78rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.08em !important;
        }}
        /* Tables & dataframes */
        .stDataFrame, .stTable {{
            background: {THEME["bg_card"]} !important;
            border: 1px solid {THEME["border"]} !important;
            border-radius: 8px !important;
            overflow: hidden;
        }}
        table {{
            color: {THEME["text"]} !important;
        }}
        thead tr th {{
            background: {THEME["bg_panel"]} !important;
            color: {THEME["primary"]} !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            font-size: 0.75rem !important;
        }}
        tbody tr td {{
            background: {THEME["bg_card"]} !important;
            color: {THEME["text"]} !important;
        }}
        /* Alerts */
        [data-testid="stAlert"] {{
            background: {THEME["bg_card"]} !important;
            border: 1px solid {THEME["primary"]} !important;
            color: {THEME["text"]} !important;
            border-radius: 8px !important;
        }}
        /* Selectbox & inputs */
        [data-baseweb="select"] > div {{
            background: {THEME["bg_card"]} !important;
            border-color: {THEME["border"]} !important;
            color: {THEME["text"]} !important;
        }}
        [data-baseweb="input"] {{
            background: {THEME["bg_card"]} !important;
            border-color: {THEME["border"]} !important;
        }}
        /* Custom card */
        .cyber-card {{
            background: linear-gradient(135deg, {THEME["bg_card"]} 0%,
                {THEME["bg_panel"]} 100%) !important;
            border: 1px solid {THEME["border"]} !important;
            border-radius: 12px !important;
            padding: 1.5rem !important;
            box-shadow: 0 6px 24px rgba(0,0,0,0.4) !important;
        }}
        .cyber-card.accent {{
            border-left: 4px solid {THEME["primary"]} !important;
        }}
        .cyber-card.danger {{
            border-left: 4px solid {THEME["accent"]} !important;
        }}
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-right: 0.35rem;
        }}
        .badge.neon {{
            background: rgba(0, 229, 168, 0.12) !important;
            color: {THEME["primary"]} !important;
            border: 1px solid {THEME["primary"]} !important;
        }}
        .badge.red {{
            background: rgba(255, 59, 59, 0.12) !important;
            color: {THEME["accent"]} !important;
            border: 1px solid {THEME["accent"]} !important;
        }}
        .badge.purple {{
            background: rgba(157, 78, 221, 0.12) !important;
            color: {THEME["neon_purple"]} !important;
            border: 1px solid {THEME["neon_purple"]} !important;
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 0.75rem;
            margin: 1rem 0;
        }}
        .stat-tile {{
            background: {THEME["bg_card"]};
            border: 1px solid {THEME["border"]};
            border-radius: 8px;
            padding: 0.9rem;
            text-align: center;
        }}
        .stat-tile .val {{
            font-size: 1.6rem;
            font-weight: 800;
            color: {THEME["primary"]};
        }}
        .stat-tile .lbl {{
            font-size: 0.72rem;
            color: {THEME["text_muted"]};
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-top: 0.3rem;
        }}
        .footer {{
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid {THEME["border"]};
            color: {THEME["text_muted"]};
            font-size: 0.8rem;
            text-align: center;
        }}
        .hero-title {{
            font-size: 3.2rem;
            font-weight: 900;
            background: linear-gradient(135deg, {THEME["primary"]} 0%,
                {THEME["neon_blue"]} 50%, {THEME["neon_purple"]} 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0;
            line-height: 1.05;
        }}
        .hero-sub {{
            font-size: 1.1rem;
            color: {THEME["text_muted"]};
            margin-top: 0.5rem;
        }}
        .scan-line {{
            height: 2px;
            background: linear-gradient(90deg, transparent,
                {THEME["primary"]}, transparent);
            margin: 1rem 0;
            animation: scan 2s infinite;
        }}
        @keyframes scan {{
            0% {{ opacity: 0.3; }}
            50% {{ opacity: 1; }}
            100% {{ opacity: 0.3; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------- Artifact loaders ----------------------------- #

@st.cache_resource(show_spinner=False)
def _load_pipeline_artifacts():
    """Load the persisted FeatureEngineer, FeatureSelector, and best model."""
    fe = load_joblib(CFG.features_dir / "feature_engineer.joblib")
    fs = load_joblib(CFG.features_dir / "feature_selector.joblib")
    # Best model
    best_path = CFG.trained_models_dir / "best_model.joblib"
    if not best_path.exists():
        # Look for any model file
        for p in CFG.trained_models_dir.iterdir():
            if p.suffix == ".joblib" and p.name != "best_model.joblib":
                best_path = p
                break
    model = load_joblib(best_path)
    meta_path = CFG.trained_models_dir / "best_model_metadata.json"
    meta = {}
    if meta_path.exists():
        with meta_path.open() as fh:
            meta = json.load(fh)
    is_xgb_like = bool(meta.get("is_xgb_like", False))
    return fe, fs, model, meta, is_xgb_like


@st.cache_data(show_spinner=False)
def _load_eval_results():
    path = CFG.evaluations_dir / "eval_results.json"
    if not path.exists():
        return None, None
    with path.open() as fh:
        data = json.load(fh)
    cmp_path = CFG.evaluations_dir / "model_comparison.csv"
    cmp = pd.read_csv(cmp_path) if cmp_path.exists() else None
    return data, cmp


@st.cache_data(show_spinner=False)
def _load_cluster_report():
    path = CFG.clusters_dir / "cluster_report.json"
    if not path.exists():
        return None
    with path.open() as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def _load_feature_importance():
    mi_path = CFG.features_dir / "selection_report.json"
    shap_path = CFG.shap_dir / "global_feature_importance.json"
    mi = None
    shap = None
    if mi_path.exists():
        with mi_path.open() as fh:
            mi = json.load(fh)
    if shap_path.exists():
        with shap_path.open() as fh:
            shap = json.load(fh)
    return mi, shap


@st.cache_data(show_spinner=False)
def _load_train_df():
    path = CFG.raw_data_dir / "phising_08012020_120000.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _load_reference_predicted():
    path = CFG.raw_data_dir / "predicted_file.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# ----------------------------- Sidebar / nav ----------------------------- #

PAGES = [
    ("Home", "home", "🏠"),
    ("About", "about", "ℹ"),
    ("Dataset Overview", "dataset", "📊"),
    ("EDA Dashboard", "eda", "📈"),
    ("Behavior Clusters", "clusters", "🧩"),
    ("Single Website Checker", "single", "🔍"),
    ("Batch Prediction", "batch", "📤"),
    ("Explainability", "explain", "🧠"),
    ("Model Performance", "perf", "⚡"),
    ("Download Results", "download", "⬇"),
]


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            f"""
            <div style="text-align:center; padding:0.8rem 0; margin-bottom:1rem;
                        border-bottom:1px solid {THEME['border']};">
                <div style="font-size:1.5rem; font-weight:900;
                            background: linear-gradient(135deg, {THEME['primary']}, {THEME['neon_blue']});
                            -webkit-background-clip: text;
                            -webkit-text-fill-color: transparent;
                            background-clip: text;">PHISHGUARD</div>
                <div style="font-size:0.7rem; color: {THEME['text_muted']};
                            letter-spacing: 0.18em; text-transform:uppercase;
                            margin-top: 0.3rem;">Phishing Detection Suite</div>
            </div>
            """,
            unsafe_allow_html=True)
        # Navigation radio
        for label, key, icon in PAGES:
            if st.button(f"{icon}  {label}", key=f"nav_{key}",
                         use_container_width=True,
                         type="primary" if st.session_state.get(
                             "page", "home") == key else "secondary"):
                st.session_state["page"] = key
                st.rerun()
        st.markdown("---")
        st.caption(f"v{CFG.config.project.version}")
        st.caption("M.S. Data Science Project")
        st.caption("© 2026 PhishGuard")


# ----------------------------- Page renderers ----------------------------- #
def page_home():
    st.markdown(
        f"""
        <div style="padding: 2rem 0;">
            <div class="hero-title">PHISHGUARD</div>
            <div class="hero-sub">Adaptive Phishing Website Detection Engine</div>
            <div class="scan-line"></div>
        </div>
        <div class="cyber-card accent" style="margin-bottom: 1.5rem;">
            <h3 style="color: {THEME['primary']}; margin-top: 0;">
                Defensive intelligence for the modern web
            </h3>
            <p>PhishGuard is a production-grade ML system that classifies
            websites as <span style="color: {THEME['accent']}; font-weight:700;">
            phishing</span> or
            <span style="color: {THEME['primary']}; font-weight:700;">safe</span>
            using 30 pre-engineered lexical, host-based, and content-based
            URL features. The system orchestrates nine supervised classifiers,
            two ensembles, an AutoGluon TabularPredictor, and ten unsupervised
            segmentation algorithms - all explained through SHAP and LIME
            interpretability layers.</p>
        </div>
        """,
        unsafe_allow_html=True)

    # Stat tiles
    train_df = _load_train_df()
    eval_data, _ = _load_eval_results()

    n_rows = len(train_df) if train_df is not None else 0
    n_features = len(CFG.features)
    n_models = len(eval_data["results"]) if eval_data else 0
    n_phish = int((train_df["Result"] == -1).sum()) if train_df is not None else 0
    n_safe = int((train_df["Result"] == 1).sum()) if train_df is not None else 0

    st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-tile"><div class="val">{n_rows:,}</div>
                <div class="lbl">Records Trained</div></div>
            <div class="stat-tile"><div class="val">{n_features}</div>
                <div class="lbl">Engineered Features</div></div>
            <div class="stat-tile"><div class="val">{n_models}</div>
                <div class="lbl">Models Trained</div></div>
            <div class="stat-tile"><div class="val">{n_phish:,}</div>
                <div class="lbl">Phishing Samples</div></div>
            <div class="stat-tile"><div class="val">{n_safe:,}</div>
                <div class="lbl">Legitimate Samples</div></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
            <div class="cyber-card accent" style="height: 100%;">
                <h4 style="color: {THEME['primary']}; margin-top: 0;">Classifiers</h4>
                <p style="font-size:0.85rem; color: {THEME['text_muted']};">
                    Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost,
                    HistGradientBoosting, GradientBoosting, AdaBoost, Bagging
                </p>
            </div>
            """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="cyber-card accent" style="height: 100%;">
                <h4 style="color: {THEME['primary']}; margin-top: 0;">Ensembles</h4>
                <p style="font-size:0.85rem; color: {THEME['text_muted']};">
                    Soft Voting (5 base learners) and Stacking
                    (Logistic Regression meta-learner) +
                    AutoGluon TabularPredictor
                </p>
            </div>
            """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="cyber-card accent" style="height: 100%;">
                <h4 style="color: {THEME['primary']}; margin-top: 0;">Explainability</h4>
                <p style="font-size:0.85rem; color: {THEME['text_muted']};">
                    SHAP TreeExplainer for global feature attribution,
                    LIME for per-prediction local explanations, and a
                    security-tip generator
                </p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Start exploring")
    st.markdown("Use the sidebar to navigate between modules. The recommended path is:")
    st.markdown("""
        1. **Dataset Overview** - inspect class balance and feature distributions
        2. **EDA Dashboard** - explore correlations and discriminative features
        3. **Single Website Checker** - test a 30-feature record interactively
        4. **Batch Prediction** - upload a CSV and download predictions
        5. **Explainability** - understand why a record was flagged
        6. **Model Performance** - compare all trained models
    """)


def page_about():
    st.title("About the Project")
    st.markdown(
        f"""
        <div class="cyber-card accent">
            <h3 style="color: {THEME['primary']}; margin-top: 0;">Mission Statement</h3>
            <p>Phishing attacks remain the most prevalent initial-access vector in
            cybersecurity breaches, accounting for a substantial share of
            web-borne intrusions year after year. <b>PhishGuard</b> closes the
            detection gap by combining supervised classification with
            unsupervised behavioral segmentation, providing both a verdict and
            a human-readable rationale for every URL inspected.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Pipeline Architecture")
    st.markdown("""
        The system follows a modular, configuration-driven architecture
        inspired by industry MLOps best practices:
    """)

    pipeline_steps = [
        ("Data Ingestion", "Load raw UCI Phishing Websites CSVs"),
        ("Validation", "Enforce ternary {-1,0,1} schema"),
        ("Transformation", "Optional PCA / scaling pipeline"),
        ("Feature Engineering", "Interactions, ratios, statistical aggregates"),
        ("Feature Selection", "Mutual information + RF + RFE union"),
        ("Clustering", "10 unsupervised algorithms (interpretive)"),
        ("Classification", "9 base + 2 ensembles + AutoGluon"),
        ("Evaluation", "Accuracy, Precision, Recall, F1, AUC, MCC, LogLoss"),
        ("Explainability", "SHAP global + LIME local + security tips"),
        ("Deployment", "Streamlit dashboard (this app)"),
    ]
    for i, (name, desc) in enumerate(pipeline_steps, 1):
        st.markdown(f"""
            <div class="cyber-card accent" style="margin-bottom: 0.6rem;
                padding: 0.8rem 1rem;">
                <span style="color: {THEME['primary']}; font-weight: 800;
                font-size: 1.1rem;">{i:02d}</span>
                &nbsp;&nbsp;
                <span style="font-weight: 700;">{name}</span>
                &nbsp;&mdash;&nbsp;
                <span style="color: {THEME['text_muted']}; font-size: 0.9rem;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### Tech Stack")
    cols = st.columns(4)
    tech = [
        ("Language", "Python 3.11+"),
        ("ML", "scikit-learn 1.5"),
        ("Boosting", "XGBoost / LightGBM / CatBoost"),
        ("AutoML", "AutoGluon Tabular"),
        ("Hyperopt", "Optuna TPE"),
        ("Explainability", "SHAP + LIME"),
        ("Visualization", "Plotly + Matplotlib + Seaborn"),
        ("Dashboard", "Streamlit 1.38"),
        ("Config", "PyYAML + python-box"),
        ("Storage", "Joblib (versioned)"),
        ("Standards", "SOLID + PEP8 + Type hints"),
        ("Docs", "Google-style docstrings"),
    ]
    for i, (k, v) in enumerate(tech):
        with cols[i % 4]:
            st.markdown(f"""
                <div class="cyber-card accent" style="padding:0.7rem;">
                    <div style="font-size:0.7rem; color: {THEME['text_muted']};
                    text-transform:uppercase; letter-spacing:0.08em;">{k}</div>
                    <div style="font-weight: 700; margin-top: 0.2rem;">{v}</div>
                </div>
                """, unsafe_allow_html=True)


def page_dataset():
    st.title("Dataset Overview")
    train_df = _load_train_df()
    if train_df is None:
        st.error("Training dataset not found. Run `python training.py` first.")
        return
    st.markdown(
        f"""
        <div class="cyber-card accent">
            <p><b>Source:</b> UCI Machine Learning Repository - Phishing Websites Data Set
            (Mohammad, Thabtah & McCluskey)</p>
            <p><b>Training rows:</b> {len(train_df):,} &nbsp;•&nbsp;
               <b>Features:</b> {len(CFG.features)} &nbsp;•&nbsp;
               <b>Target:</b> Result ∈ {{-1, 1}}</p>
            <p><b>Encoding:</b> All 30 features are pre-engineered and ternary-encoded
            ({'{-1, 0, 1}'}).</p>
        </div>
        """, unsafe_allow_html=True)

    # Class balance
    st.markdown("### Class Balance")
    counts = train_df["Result"].value_counts().to_dict()
    counts_str = {int(k): v for k, v in counts.items()}
    fig = go.Figure(data=[
        go.Bar(x=["Phishing (-1)", "Legitimate (+1)"],
               y=[counts_str.get(-1, 0), counts_str.get(1, 0)],
               marker_color=[THEME["phishing"], THEME["safe"]],
               text=[counts_str.get(-1, 0), counts_str.get(1, 0)],
               textposition="auto",
               textfont=dict(color="#0A0E14", size=14, family="Arial Black"))])
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=THEME["bg_card"],
        plot_bgcolor=THEME["bg_card"],
        font=dict(color=THEME["text"]),
        height=380,
        showlegend=False,
        margin=dict(t=20, b=20, l=40, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Per-feature distribution by class
    st.markdown("### Per-Feature Distribution by Class")
    feats = list(CFG.features)
    selected = st.selectbox("Select feature", feats, index=feats.index("SSLfinal_State"))
    phish_vals = train_df[train_df["Result"] == -1][selected].value_counts().to_dict()
    legit_vals = train_df[train_df["Result"] == 1][selected].value_counts().to_dict()
    cats = sorted(set(list(phish_vals.keys()) + list(legit_vals.keys())))
    fig = go.Figure(data=[
        go.Bar(name="Phishing", x=cats,
               y=[phish_vals.get(c, 0) for c in cats],
               marker_color=THEME["phishing"]),
        go.Bar(name="Legitimate", x=cats,
               y=[legit_vals.get(c, 0) for c in cats],
               marker_color=THEME["safe"])
    ])
    fig.update_layout(
        barmode="group", template="plotly_dark",
        paper_bgcolor=THEME["bg_card"], plot_bgcolor=THEME["bg_card"],
        font=dict(color=THEME["text"]), height=380,
        xaxis_title=selected, yaxis_title="Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        margin=dict(t=20, b=40, l=40, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Sample Rows")
    st.dataframe(train_df.head(10), use_container_width=True, height=300)


def page_eda():
    st.title("Exploratory Data Analysis")
    train_df = _load_train_df()
    if train_df is None:
        st.error("Training dataset not found.")
        return

    st.markdown("### Feature-Target Correlation Heatmap")
    corr = train_df.corr()
    target_corr = corr["Result"].drop("Result").sort_values()
    fig = go.Figure(data=[
        go.Bar(
            x=target_corr.values,
            y=target_corr.index,
            orientation="h",
            marker_color=[
                THEME["phishing"] if v < 0 else THEME["safe"]
                for v in target_corr.values
            ],
            text=[f"{v:+.3f}" for v in target_corr.values],
            textposition="outside",
            textfont=dict(color=THEME["text_muted"], size=9),
        )
    ])
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=THEME["bg_card"], plot_bgcolor=THEME["bg_card"],
        font=dict(color=THEME["text"]), height=720,
        xaxis_title="Pearson correlation with Result",
        yaxis=dict(automargin=True, tickfont=dict(size=10)),
        margin=dict(t=10, b=40, l=200, r=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Top Discriminative Features (by Mutual Information)")
    mi, _ = _load_feature_importance()
    if mi and "mi_scores" in mi:
        mi_ser = pd.Series(mi["mi_scores"]).sort_values(ascending=False).head(15)
        fig = go.Figure(data=[
            go.Bar(x=mi_ser.values, y=mi_ser.index,
                   orientation="h",
                   marker_color=THEME["primary"],
                   text=[f"{v:.4f}" for v in mi_ser.values],
                   textposition="outside",
                   textfont=dict(color=THEME["text_muted"], size=10))
        ])
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=THEME["bg_card"], plot_bgcolor=THEME["bg_card"],
            font=dict(color=THEME["text"]), height=520,
            xaxis_title="Mutual Information",
            yaxis=dict(automargin=True, tickfont=dict(size=10)),
            margin=dict(t=10, b=40, l=200, r=80),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Mutual-information report not available - run training first.")

    st.markdown("### Full Correlation Matrix (30 features + Result)")
    full_corr = train_df.corr()
    fig = go.Figure(data=go.Heatmap(
        z=full_corr.values,
        x=full_corr.columns,
        y=full_corr.columns,
        colorscale=[[0, THEME["phishing"]], [0.5, "#1F2937"],
                    [1, THEME["primary"]]],
        zmin=-1, zmax=1,
        colorbar=dict(title="ρ", tickfont=dict(color=THEME["text"])),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=THEME["bg_card"], plot_bgcolor=THEME["bg_card"],
        font=dict(color=THEME["text"]), height=720, width=720,
        margin=dict(t=10, b=40, l=40, r=40),
        xaxis=dict(tickfont=dict(size=7), tickangle=-90),
        yaxis=dict(tickfont=dict(size=7)),
    )
    st.plotly_chart(fig, use_container_width=True)


def page_clusters():
    st.title("Website Behavior Clusters (Unsupervised)")
    st.markdown(
        f"""
        <div class="cyber-card accent">
            <p><b>Module type:</b> Unsupervised behavioral segmentation
            <span class="badge purple">interpretive</span></p>
            <p style="color: {THEME['text_muted']}; font-size: 0.9rem;">
            Cluster assignments are NEVER used to decide phishing vs. legitimate.
            They serve to surface structural patterns in website behavior for
            interpretability and threat-actor profiling.</p>
        </div>
        """, unsafe_allow_html=True)

    report = _load_cluster_report()
    if report is None:
        st.info("Cluster report not found - run training first.")
        return

    st.markdown("### Algorithm Comparison")
    results = report.get("results", [])
    df = pd.DataFrame(results)
    df = df[["name", "n_clusters", "silhouette", "davies_bouldin",
             "calinski_harabasz", "note"]].copy()
    df["silhouette"] = df["silhouette"].apply(
        lambda x: f"{x:.4f}" if x is not None else "N/A")
    df["davies_bouldin"] = df["davies_bouldin"].apply(
        lambda x: f"{x:.4f}" if x is not None else "N/A")
    df["calinski_harabasz"] = df["calinski_harabasz"].apply(
        lambda x: f"{x:.2f}" if x is not None else "N/A")
    df.columns = ["Algorithm", "Clusters", "Silhouette ↑", "DB ↓", "CH ↑", "Note"]
    st.dataframe(df, use_container_width=True, hide_index=True)

    best = report.get("best_algorithm", "")
    best_sil = report.get("best_silhouette", -1)
    st.markdown(f"""
        <div class="cyber-card accent" style="margin-top: 1rem;">
            <h4 style="color: {THEME['primary']}; margin-top: 0;">
                Selected Algorithm: {best}
            </h4>
            <p style="color: {THEME['text_muted']};">
            Best silhouette score: <b style="color: {THEME['primary']};">
            {best_sil:.4f}</b> &nbsp;•&nbsp;
            Selected via primary metric (silhouette), with Davies-Bouldin
            and Calinski-Harabasz reported alongside for sanity check.</p>
        </div>
        """, unsafe_allow_html=True)

    # PCA 2D scatter
    st.markdown("### 2D PCA Projection (colored by cluster)")
    try:
        fe, fs, model, meta, _ = _load_pipeline_artifacts()
        train_df = _load_train_df()
        if train_df is not None:
            from sklearn.decomposition import PCA
            X = train_df[CFG.features].head(2000).copy()
            X_eng = fe.transform(X)
            X_sel = fs.transform(X_eng)
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X_sel)
            # Load best cluster model labels
            best_model_path = CFG.clusters_dir / f"{best.lower()}.joblib"
            if best_model_path.exists():
                cm = load_joblib(best_model_path)
                if hasattr(cm, "predict"):
                    labels = cm.predict(X_sel)
                elif hasattr(cm, "labels_"):
                    labels = cm.labels_
                else:
                    labels = np.zeros(len(X_sel), dtype=int)
            else:
                labels = np.zeros(len(X_sel), dtype=int)
            fig_df = pd.DataFrame({
                "PC1": coords[:, 0], "PC2": coords[:, 1],
                "Cluster": labels.astype(str),
                "True Label": ["Phishing" if t == -1 else "Legitimate"
                              for t in train_df["Result"].head(2000)]
            })
            fig = px.scatter(fig_df, x="PC1", y="PC2", color="Cluster",
                             symbol="True Label",
                             template="plotly_dark",
                             color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(
                paper_bgcolor=THEME["bg_card"],
                plot_bgcolor=THEME["bg_card"],
                font=dict(color=THEME["text"]), height=520,
                margin=dict(t=10, b=20, l=40, r=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Training dataset not loaded.")
    except Exception as exc:
        st.warning(f"Cluster visualization unavailable: {exc}")


def _render_single_form() -> Dict[str, int]:
    """Render the 30-feature form and return the record as a dict."""
    feats = list(CFG.features)
    record: Dict[str, int] = {}
    # Group features into 5 columns of 6 each
    n_cols = 5
    n_per_col = (len(feats) + n_cols - 1) // n_cols
    cols = st.columns(n_cols)
    for i, feat in enumerate(feats):
        col_idx = (i // n_per_col) % n_cols
        with cols[col_idx]:
            record[feat] = st.selectbox(
                feat, options=[-1, 0, 1],
                index=1,  # default 0
                key=f"feat_{feat}",
                help=str(CFG.prediction_schema.feature_security_tips
                        .get(feat, {}).get("0", ""))
            )
    return record


def page_single():
    st.title("Single Website Checker")
    st.markdown(
        f"""
        <div class="cyber-card accent">
            <p>Enter values for each of the 30 pre-engineered URL features:</p>
            <ul style="margin-top:0.5rem;">
                <li><b style="color: {THEME['accent']};">-1</b> = phishing indicator present</li>
                <li><b style="color: {THEME['text_muted']};">0</b> = neutral / unknown</li>
                <li><b style="color: {THEME['primary']};">+1</b> = legitimate indicator present</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    record = _render_single_form()
    st.markdown("---")

    if st.button("🛡  Run Prediction", type="primary", use_container_width=True):
        with st.spinner("Running feature-engineering + classification..."):
            try:
                fe, fs, model, meta, is_xgb = _load_pipeline_artifacts()
                pipe = PredictionPipeline(
                    feature_engineer=fe, feature_selector=fs,
                    model=model, is_xgb_like=is_xgb)
                out = pipe.predict_single(record)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                return

        # Render result
        is_phish = out.label == "phising"
        color = THEME["phishing"] if is_phish else THEME["safe"]
        icon = "⚠" if is_phish else "✓"
        title = "Phishing Website Detected" if is_phish else "Legitimate Website"
        st.markdown(
            f"""
            <div class="cyber-card {'danger' if is_phish else 'accent'}"
                 style="text-align:center; padding: 2rem;">
                <div style="font-size: 4rem; line-height: 1;">{icon}</div>
                <div style="font-size: 1.8rem; font-weight: 800;
                            color: {color}; margin-top: 0.5rem;">{title}</div>
                <div style="margin-top: 0.8rem;">
                    <span class="badge {'red' if is_phish else 'neon'}">
                        Label: {out.label}</span>
                    <span class="badge {'red' if is_phish else 'neon'}">
                        Confidence: {out.confidence*100:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Risk gauge
        st.markdown("### Risk Gauge")
        gauge_fig = go.Figure(go.Indicator(
            domain={"x": [0, 1], "y": [0, 1]},
            value=out.p_phishing * 100,
            mode="gauge+number",
            number={"suffix": "%", "font": dict(color=THEME["text"], size=44)},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": THEME["text_muted"]},
                "bar": {"color": color, "thickness": 0.45},
                "bgcolor": THEME["bg_card"],
                "borderwidth": 2,
                "bordercolor": THEME["border"],
                "steps": [
                    {"range": [0, 35], "color": "rgba(0, 230, 118, 0.18)"},
                    {"range": [35, 65], "color": "rgba(255, 176, 32, 0.18)"},
                    {"range": [65, 100], "color": "rgba(255, 59, 59, 0.22)"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 4},
                    "thickness": 0.85,
                    "value": out.p_phishing * 100,
                },
            },
            title={"text": "Phishing Probability", "font": dict(color=THEME["text"])}
        ))
        gauge_fig.update_layout(
            paper_bgcolor=THEME["bg_card"], height=300,
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

        # Probability bar
        st.markdown("### Class Probabilities")
        prob_df = pd.DataFrame({
            "Class": ["Phishing", "Safe"],
            "Probability": [out.p_phishing, out.p_safe],
            "Color": [THEME["phishing"], THEME["safe"]],
        })
        fig = go.Figure(data=[
            go.Bar(x=prob_df["Class"], y=prob_df["Probability"],
                   marker_color=prob_df["Color"],
                   text=[f"{p*100:.1f}%" for p in prob_df["Probability"]],
                   textposition="auto",
                   textfont=dict(color="#0A0E14", size=14, family="Arial Black"))
        ])
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=THEME["bg_card"], plot_bgcolor=THEME["bg_card"],
            font=dict(color=THEME["text"]), height=300,
            yaxis=dict(range=[0, 1], tickformat=".0%"),
            margin=dict(t=10, b=20, l=40, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Per-feature security tips
        st.markdown("### Per-Feature Security Analysis")
        tips = []
        tips_cfg = CFG.prediction_schema.feature_security_tips
        for fname in CFG.features:
            val = record[fname]
            tip_dict = tips_cfg.get(fname, {})
            tip = tip_dict.get(str(val), "No tip available.")
            is_risky = (val == -1)
            color_dot = THEME["phishing"] if is_risky else (
                THEME["text_muted"] if val == 0 else THEME["safe"])
            tips.append({
                "Feature": fname,
                "Value": val,
                "Indicator": "🔴 risky" if is_risky else (
                    "⚪ neutral" if val == 0 else "🟢 safe"),
                "Security Tip": tip,
            })
        tips_df = pd.DataFrame(tips)
        st.dataframe(tips_df, use_container_width=True, height=500)


def page_batch():
    st.title("Batch Prediction")
    st.markdown(
        f"""
        <div class="cyber-card accent">
            <p>Upload a CSV in the <code>phisingtest.csv</code> schema
            (30 feature columns, no Result). The pipeline will:</p>
            <ol>
                <li>Apply feature-engineering + selection</li>
                <li>Score every row with the best trained classifier</li>
                <li>Return a CSV matching <code>predicted_file.csv</code>:
                    the 30 features + <code>Result</code> (string label
                    <code>"phising"</code> or <code>"safe"</code>) +
                    <code>confidence</code> + <code>probability_phishing</code></li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    uploaded = st.fileUploader("Choose a CSV file", type=["csv"])
    sample_file = CFG.raw_data_dir / "phisingtest.csv"
    if st.button("Use Sample phisingtest.csv", type="secondary"):
        if sample_file.exists():
            df = pd.read_csv(sample_file)
            st.session_state["batch_df"] = df
            st.rerun()

    if "batch_df" in st.session_state:
        df = st.session_state["batch_df"]
        st.markdown(f"**Loaded:** {len(df):,} rows × {df.shape[1]} cols")
        st.dataframe(df.head(10), use_container_width=True, height=300)

        if st.button("🚀 Run Batch Prediction", type="primary"):
            with st.spinner("Scoring batch..."):
                try:
                    fe, fs, model, meta, is_xgb = _load_pipeline_artifacts()
                    pipe = PredictionPipeline(
                        feature_engineer=fe, feature_selector=fs,
                        model=model, is_xgb_like=is_xgb)
                    out_df = pipe.predict_batch(
                        df[CFG.features],
                        out_path=CFG.predictions_dir /
                                 f"batch_pred_{int(time.time())}.csv")
                    st.session_state["batch_pred"] = out_df
                except Exception as exc:
                    st.error(f"Batch prediction failed: {exc}")
                    return

    if "batch_pred" in st.session_state:
        out_df = st.session_state["batch_pred"]
        st.markdown("### Prediction Results")
        st.dataframe(out_df.head(20), use_container_width=True, height=400)

        # Class distribution
        st.markdown("### Prediction Class Distribution")
        vc = out_df["Result"].value_counts()
        fig = go.Figure(data=[
            go.Bar(x=vc.index, y=vc.values,
                   marker_color=[THEME["phishing"] if l == "phising"
                                 else THEME["safe"] for l in vc.index],
                   text=vc.values, textposition="auto",
                   textfont=dict(color="#0A0E14", size=14,
                                 family="Arial Black"))
        ])
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=THEME["bg_card"], plot_bgcolor=THEME["bg_card"],
            font=dict(color=THEME["text"]), height=320,
            margin=dict(t=10, b=20, l=40, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Confidence distribution
        st.markdown("### Confidence Distribution")
        fig = go.Figure(data=[
            go.Histogram(x=out_df["confidence"], nbinsx=40,
                         marker_color=THEME["primary"],
                         marker_line_color=THEME["border"],
                         marker_line_width=1)
        ])
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=THEME["bg_card"], plot_bgcolor=THEME["bg_card"],
            font=dict(color=THEME["text"]), height=320,
            xaxis_title="Confidence", yaxis_title="Count",
            margin=dict(t=10, b=40, l=40, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Download
        st.markdown("### Download")
        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download predictions.csv",
            data=csv_bytes,
            file_name="predicted_file.csv",
            mime="text/csv",
        )


def page_explain():
    st.title("Feature Importance & Explainability")
    mi, shap_imp = _load_feature_importance()

    if shap_imp:
        st.markdown("### SHAP Global Feature Importance (mean |SHAP|)")
        ser = pd.Series(shap_imp).sort_values(ascending=False).head(20)
        fig = go.Figure(data=[
            go.Bar(x=ser.values, y=ser.index, orientation="h",
                   marker_color=THEME["primary"],
                   text=[f"{v:.4f}" for v in ser.values],
                   textposition="outside",
                   textfont=dict(color=THEME["text_muted"], size=10))
        ])
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=THEME["bg_card"], plot_bgcolor=THEME["bg_card"],
            font=dict(color=THEME["text"]), height=600,
            xaxis_title="Mean |SHAP value|",
            yaxis=dict(automargin=True, tickfont=dict(size=10)),
            margin=dict(t=10, b=40, l=200, r=80),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("SHAP global importance not available.")

    if mi:
        st.markdown("### Mutual Information vs RandomForest Importance")
        mi_ser = pd.Series(mi.get("mi_scores", {})).sort_values(
            ascending=False).head(15)
        rf_ser = pd.Series(mi.get("rf_importances", {})).sort_values(
            ascending=False).head(15)
        col1, col2 = st.columns(2)
        for col, ser, title, color in [
            (col1, mi_ser, "Mutual Information", THEME["neon_blue"]),
            (col2, rf_ser, "RandomForest Importance", THEME["neon_purple"])
        ]:
            with col:
                fig = go.Figure(data=[
                    go.Bar(x=ser.values, y=ser.index, orientation="h",
                           marker_color=color,
                           text=[f"{v:.4f}" for v in ser.values],
                           textposition="outside",
                           textfont=dict(color=THEME["text_muted"], size=9))
                ])
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor=THEME["bg_card"],
                    plot_bgcolor=THEME["bg_card"],
                    font=dict(color=THEME["text"]), height=500,
                    title=dict(text=title, font=dict(color=THEME["primary"],
                                                    size=16)),
                    yaxis=dict(automargin=True, tickfont=dict(size=9)),
                    margin=dict(t=40, b=20, l=160, r=80),
                )
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Per-Prediction Local Explanation (LIME)")
    st.markdown("Adjust features below and click Explain to get a LIME-based "
                "local explanation with security tips.")
    record = _render_single_form()
    if st.button("🧠 Explain This Prediction", type="primary"):
        with st.spinner("Generating LIME explanation..."):
            try:
                fe, fs, model, meta, is_xgb = _load_pipeline_artifacts()
                pipe = PredictionPipeline(
                    feature_engineer=fe, feature_selector=fs,
                    model=model, is_xgb_like=is_xgb)
                from src.explainability import explain_record
                tips_cfg = CFG.prediction_schema.feature_security_tips
                expl = explain_record(pipe, record, tips_cfg, top_k=10)
            except Exception as exc:
                st.error(f"Explanation failed: {exc}")
                return

        is_phish = expl.label == "phising"
        color = THEME["phishing"] if is_phish else THEME["safe"]
        st.markdown(
            f"""
            <div class="cyber-card {'danger' if is_phish else 'accent'}">
                <h3 style="color: {color}; margin-top: 0;">
                    Predicted: {expl.label.upper()}
                </h3>
            </div>
            """, unsafe_allow_html=True)

        if expl.top_features:
            st.markdown("### Top contributing features")
            df = pd.DataFrame(expl.top_features,
                              columns=["Feature (LIME weight)", "Weight"])
            df["Weight"] = df["Weight"].round(4)
            st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("### Security Tips")
        for tip in expl.security_tips:
            st.markdown(f"- {tip}")


def page_perf():
    st.title("Model Performance Dashboard")
    eval_data, cmp_df = _load_eval_results()
    if eval_data is None:
        st.error("Evaluation results not found - run training first.")
        return

    # Best model card
    best_meta_path = CFG.trained_models_dir / "best_model_metadata.json"
    if best_meta_path.exists():
        with best_meta_path.open() as fh:
            best_meta = json.load(fh)
        st.markdown(
            f"""
            <div class="cyber-card accent">
                <h3 style="color: {THEME['primary']}; margin-top: 0;">
                    🏆 Best Model: {best_meta.get("name", "")}
                </h3>
                <p>Selected by <b>recall on the phishing class</b>
                (a missed phishing site is costlier than a false alarm),
                tie-broken by ROC-AUC.</p>
            </div>
            """, unsafe_allow_html=True)
        m = best_meta.get("metrics", {})
        col1, col2, col3, col4 = st.columns(4)
        for col, k, v in [
            (col1, "Accuracy", m.get("accuracy")),
            (col2, "Recall (Phishing)", m.get("recall_phishing")),
            (col3, "F1", m.get("f1")),
            (col4, "ROC-AUC", m.get("roc_auc")),
        ]:
            with col:
                st.metric(k, f"{v*100:.2f}%" if v is not None else "—")

    # Full comparison table
    st.markdown("### Comparison Table (all models)")
    st.dataframe(cmp_df, use_container_width=True, hide_index=True)

    # Side-by-side bar chart
    st.markdown("### Metric Comparison Across Models")
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc",
                   "pr_auc", "mcc"]
    selected_metric = st.selectbox(
        "Select metric to sort by", metric_cols, index=metric_cols.index("recall"))
    fig = go.Figure(data=[
        go.Bar(x=cmp_df.sort_values(by=selected_metric, ascending=False)["model"],
               y=cmp_df.sort_values(by=selected_metric, ascending=False)[selected_metric],
               marker_color=THEME["primary"],
               text=cmp_df.sort_values(by=selected_metric, ascending=False)[selected_metric].apply(lambda v: f"{v:.4f}"),
               textposition="outside",
               textfont=dict(color=THEME["text_muted"], size=10))
    ])
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=THEME["bg_card"], plot_bgcolor=THEME["bg_card"],
        font=dict(color=THEME["text"]), height=450,
        xaxis_title="Model", yaxis_title=selected_metric,
        xaxis=dict(tickangle=-30),
        margin=dict(t=10, b=80, l=40, r=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Confusion matrices side-by-side
    st.markdown("### Confusion Matrices")
    cms = [(r["name"], r["confusion_matrix"]) for r in eval_data["results"]
           if r.get("confusion_matrix")]
    n_cols = 3
    rows = (len(cms) + n_cols - 1) // n_cols
    for r in range(rows):
        cols = st.columns(n_cols)
        for c in range(n_cols):
            idx = r * n_cols + c
            if idx >= len(cms):
                continue
            name, cm = cms[idx]
            with cols[c]:
                st.markdown(f"**{name}**")
                # cm is [[TP, FN], [FP, TN]] (labels=[1=phish,0=safe])
                labels = ["Phishing", "Safe"]
                fig = go.Figure(data=go.Heatmap(
                    z=cm, x=["Pred Phishing", "Pred Safe"],
                    y=["Act Phishing", "Act Safe"],
                    colorscale=[[0, THEME["bg_panel"]],
                                [0.5, THEME["warning"]],
                                [1, THEME["phishing"]]],
                    text=cm, texttemplate="%{text}",
                    textfont=dict(color="#0A0E14", size=14,
                                  family="Arial Black"),
                ))
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor=THEME["bg_card"],
                    plot_bgcolor=THEME["bg_card"],
                    font=dict(color=THEME["text"]), height=300,
                    margin=dict(t=10, b=10, l=40, r=20),
                )
                st.plotly_chart(fig, use_container_width=True)

    # ROC curves (simulated from AUC values for clarity)
    st.markdown("### ROC-AUC Comparison (bar)")
    roc_df = cmp_df[["model", "roc_auc"]].sort_values(by="roc_auc", ascending=False)
    fig = go.Figure(data=[
        go.Bar(x=roc_df["model"], y=roc_df["roc_auc"],
               marker_color=THEME["neon_blue"],
               text=roc_df["roc_auc"].apply(lambda v: f"{v:.4f}"),
               textposition="outside",
               textfont=dict(color=THEME["text_muted"], size=10))
    ])
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=THEME["bg_card"], plot_bgcolor=THEME["bg_card"],
        font=dict(color=THEME["text"]), height=400,
        yaxis=dict(range=[0, 1]),
        xaxis=dict(tickangle=-30),
        margin=dict(t=10, b=80, l=40, r=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def page_download():
    st.title("Download Prediction Results")
    st.markdown("""
        Download prediction artifacts produced by the batch inference pipeline.
    """)
    # List available batch predictions
    pred_dir = CFG.predictions_dir
    files = sorted(pred_dir.glob("*.csv"), reverse=True) if pred_dir.exists() else []
    ref_file = CFG.raw_data_dir / "predicted_file.csv"
    st.markdown("### Available Files")
    if files:
        for p in files:
            cols = st.columns([4, 1])
            with cols[0]:
                st.code(f"{p.name}  ({p.stat().st_size:,} bytes)")
            with cols[1]:
                with open(p, "rb") as fh:
                    st.download_button("Download", fh.read(),
                                       file_name=p.name, mime="text/csv",
                                       key=f"dl_{p.name}")
    else:
        st.info("No batch prediction files yet. Run the Batch Prediction page first.")
    st.markdown("### Reference File (predicted_file.csv schema)")
    st.code("30 feature columns + Result (string 'phising' or 'safe') "
            "+ confidence + probability_phishing")
    if ref_file.exists():
        df = pd.read_csv(ref_file).head(5)
        st.dataframe(df, use_container_width=True, hide_index=True)
        with open(ref_file, "rb") as fh:
            st.download_button("⬇ Download reference predicted_file.csv",
                               fh.read(), file_name="predicted_file.csv",
                               mime="text/csv")


# ----------------------------- Main entry ----------------------------- #
def main() -> None:
    _inject_css()
    _render_sidebar()
    page = st.session_state.get("page", "home")
    dispatch = {
        "home": page_home,
        "about": page_about,
        "dataset": page_dataset,
        "eda": page_eda,
        "clusters": page_clusters,
        "single": page_single,
        "batch": page_batch,
        "explain": page_explain,
        "perf": page_perf,
        "download": page_download,
    }
    fn = dispatch.get(page, page_home)
    try:
        fn()
    except Exception as exc:
        st.error(f"Page error: {exc}")
        st.exception(exc)
    st.markdown(
        f"""
        <div class="footer">
            PhishGuard v{CFG.config.project.version} &nbsp;•&nbsp;
            UCI Phishing Websites Dataset &nbsp;•&nbsp;
            <span style="color: {THEME['primary']};">M.S. Data Science Project</span>
        </div>
        """,
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
