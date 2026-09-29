"""Home / landing page."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from predict import load_best_meta, load_metrics
from theme import render_metric_card, render_alert


def render() -> None:
    meta = load_best_meta()
    metrics = load_metrics()
    best = meta["best_model"]
    m = meta["metrics"]

    # Hero
    st.markdown(
        """
        <div class="page-hero fade-in">
            <h1><span class="hero-icon hero-icon-shield"><span class="material-symbols-rounded">shield</span></span>PhishGuard</h1>
            <div class="subtitle">AI-powered Phishing URL Detection · Master's Project in Data Science</div>
            <div style="margin-top:1rem;">
                <span class="chip chip-cyan">{best}</span>
                <span class="chip chip-green">{acc:.2f}% Accuracy</span>
                <span class="chip chip-red">{recall:.2f}% Phishing Recall</span>
                <span class="chip chip-yellow">AUC {auc:.3f}</span>
            </div>
        </div>
        """.format(
            best=best,
            acc=m["accuracy"] * 100,
            recall=m["recall_phish"] * 100,
            auc=m["roc_auc"],
        ),
        unsafe_allow_html=True,
    )

    # KPI strip
    st.markdown('<div class="section-divider"><h2>Key Metrics</h2></div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(render_metric_card("Accuracy", f"{m['accuracy']*100:.2f}%",
                                        "Overall classification rate", "success"), unsafe_allow_html=True)
    with col2:
        st.markdown(render_metric_card("Phishing Recall", f"{m['recall_phish']*100:.2f}%",
                                        "Sensitivity to phishing sites", "danger"), unsafe_allow_html=True)
    with col3:
        st.markdown(render_metric_card("F1 Score (phishing)", f"{m['f1_phish']*100:.2f}%",
                                        "Harmonic mean of precision/recall", "info"), unsafe_allow_html=True)
    with col4:
        st.markdown(render_metric_card("ROC-AUC", f"{m['roc_auc']:.4f}",
                                        "Area under ROC curve", "info"), unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.markdown(render_metric_card("Precision (phishing)", f"{m['precision_phish']*100:.2f}%",
                                        "True positives / predicted positives", "info"), unsafe_allow_html=True)
    with col6:
        st.markdown(render_metric_card("PR-AUC", f"{m['pr_auc']:.4f}",
                                        "Area under PR curve", "info"), unsafe_allow_html=True)
    with col7:
        st.markdown(render_metric_card("MCC", f"{m['mcc']:.4f}",
                                        "Matthews correlation coefficient", "info"), unsafe_allow_html=True)
    with col8:
        st.markdown(render_metric_card("Log Loss", f"{m['log_loss']:.4f}",
                                        "Cross-entropy loss", "warning"), unsafe_allow_html=True)

    # About the system
    st.markdown('<div class="section-divider"><h2>About the System</h2></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="metric-card fade-in" style="padding:1.5rem 2rem;">
        <p style="font-size:1rem; line-height:1.7; color:#c9d1d9; margin:0;">
        <strong style="color:#00d4ff">PhishGuard</strong> is an end-to-end machine-learning system that
        automatically distinguishes phishing websites from legitimate ones using
        pre-engineered <strong style="color:#2ed573">lexical</strong>,
        <strong style="color:#2ed573">host-based</strong>, and
        <strong style="color:#2ed573">content-based</strong> URL features.
        </p>
        <p style="font-size:0.95rem; line-height:1.7; color:#8b949e; margin-top:1rem;">
        The pipeline ingests a URL, extracts 22 numeric features and a categorical TLD signal,
        encodes the TLD with top-50 frequency bucketing, then runs the trained
        <strong style="color:#00d4ff">{best}</strong> classifier to output a probability,
        a verdict (<span style="color:#2ed573">Safe</span> / <span style="color:#ff4757">Phishing</span>),
        and a plain-language explanation. Model selection prioritizes <strong style="color:#ff4757">phishing recall</strong>
        — a missed phishing site is far costlier than a false alarm.
        </p>
        </div>
        """.format(best=best),
        unsafe_allow_html=True,
    )

    # Quick-start / navigation
    st.markdown('<div class="section-divider"><h2>Get Started</h2></div>', unsafe_allow_html=True)
    cols = st.columns(3)
    cards = [
        ("Single URL Checker", "Paste a URL and get an instant verdict, risk gauge, and per-feature explanation.", "cyan", "search"),
        ("Batch Prediction", "Upload a CSV of URLs and download a CSV of predictions + confidence scores.", "green", "folder_open"),
        ("Model Performance", "Compare all trained models on accuracy, recall, F1, ROC curves, confusion matrices.", "red", "monitoring"),
    ]
    for col, (title, desc, color, icon) in zip(cols, cards):
        col.markdown(
            f"""
            <div class="metric-card chip-{color}" style="height:140px; display:flex; flex-direction:column; justify-content:center;">
                <div style="display:flex; align-items:center; gap:0.55rem; font-size:1.1rem; font-weight:700; color:#e6edf3; margin-bottom:0.4rem;"><span class="quick-icon icon-{color}"><span class="material-symbols-rounded">{icon}</span></span>{title}</div>
                <div style="font-size:0.85rem; color:#8b949e; line-height:1.4;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Pipeline overview
    st.markdown('<div class="section-divider"><h2>Pipeline Overview</h2></div>', unsafe_allow_html=True)
    steps = [
        ("Data Ingestion", "116,586 labeled URLs · 25 numeric + categorical features"),
        ("Preprocessing", "Top-50 TLD one-hot bucketing · numeric passthrough"),
        ("Model Training", "10 classifiers · class-weight balanced · stratified split"),
        ("Evaluation", "Accuracy · Precision · Recall · F1 · AUC · MCC · LogLoss"),
        ("Deployment", "Streamlit dashboard · dark cybersecurity UI"),
    ]
    cols = st.columns(len(steps))
    for col, (title, desc) in zip(cols, steps):
        col.markdown(
            f"""
            <div class="metric-card info" style="text-align:center; padding:1rem;">
                <div style="font-size:0.95rem; font-weight:700; color:#00d4ff;">{title}</div>
                <div style="font-size:0.75rem; color:#8b949e; margin-top:0.4rem; line-height:1.4;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Footer note
    st.markdown(
        """
        <div style="margin-top:2rem;">
        <div class="alert-banner alert-info">
        <span class="inline-icon icon-cyan"><span class="material-symbols-rounded">database</span></span> <strong>Dataset:</strong> {n_train:,} training URLs · {n_test:,} held-out test URLs ·
        trained on a stratified 80/20 split.  Use the sidebar to explore each module.
        </div>
        </div>
        """.format(n_train=meta["n_train"], n_test=meta["n_test"]),
        unsafe_allow_html=True,
    )
