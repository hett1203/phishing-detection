"""Batch Prediction page — upload CSV, get predictions, download results."""
from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd
import streamlit as st

from features import extract_url_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from predict import Predictor
from theme import render_metric_card, render_alert


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
            <h1><span class="page-icon icon-green"><span class="material-symbols-rounded">folder_open</span></span> Batch Prediction</h1>
            <div class="subtitle">Upload a CSV of URLs and download predictions with confidence scores</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="alert-banner alert-info">
        <span class="inline-icon icon-cyan"><span class="material-symbols-rounded">table_view</span></span> <strong>Expected schema:</strong> CSV must contain a <code>url</code> column. The system will
        extract 22 features automatically and return a downloadable CSV with columns
        <code>url</code>, <code>label</code> (<span style="color:#2ed573">safe</span> / <span style="color:#ff4757">phishing</span>),
        <code>probability</code>, and <code>risk_score</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Upload section
    uploaded = st.file_uploader(
        "Upload your CSV file",
        type=["csv"],
        help="CSV must contain a 'url' column.",
    )

    # Sample template download
    col_dl, _ = st.columns([1, 3])
    with col_dl:
        sample_df = pd.DataFrame({
            "url": [
                "https://www.google.com/",
                "http://paypal-secure-login.com/account/verify?id=123",
                "https://github.com/",
                "http://192.168.1.1/admin/login.php",
                "https://en.wikipedia.org/wiki/Phishing",
            ]
        })
        csv_template = sample_df.to_csv(index=False).encode()
        st.download_button(
            "Download sample template",
            csv_template,
            file_name="batch_template.csv",
            mime="text/csv",
        )

    if uploaded is None:
        st.markdown(
            """
            <div class="metric-card info" style="padding:1.5rem 2rem; text-align:center; margin-top:1rem;">
                <div style="font-size:1rem; color:#8b949e;"><span class="inline-icon icon-green"><span class="material-symbols-rounded">upload_file</span></span> Drop your CSV file above to begin batch prediction</div>
                <div style="font-size:0.85rem; color:#6e7681; margin-top:0.5rem;">
                Need a starting point? Download the sample template, edit the URLs, and re-upload.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Read CSV
    try:
        df_in = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return

    if "url" not in df_in.columns:
        st.markdown(render_alert(
            f"The uploaded CSV does not contain a 'url' column. Found columns: {list(df_in.columns)}",
            "danger",
        ), unsafe_allow_html=True)
        return

    st.markdown('<div class="section-divider"><h2>Input Preview</h2></div>', unsafe_allow_html=True)
    st.dataframe(df_in.head(10), use_container_width=True, height=280)

    if st.button(":material/play_arrow: Run Batch Prediction", use_container_width=True):
        with st.spinner(f"Processing {len(df_in):,} URLs..."):
            t0 = time.time()
            urls = df_in["url"].astype(str).tolist()

            # Extract features in chunks
            feature_rows = [extract_url_features(u) for u in urls]
            features_df = pd.DataFrame(feature_rows)

            # Run predictions
            predictor = Predictor.get()
            probas = predictor.predict_features_df(features_df)
            elapsed = time.time() - t0

        # Build output DataFrame
        labels = np.where(probas >= 0.5, "phishing", "safe")
        result_df = df_in.copy()
        result_df["label"] = labels
        result_df["probability"] = (probas * 100).round(2)
        result_df["risk_score"] = (probas * 100).astype(int)

        # Summary KPIs
        n = len(result_df)
        n_phish = int((result_df["label"] == "phishing").sum())
        n_safe = int((result_df["label"] == "safe").sum())
        avg_prob = float(result_df["probability"].mean())

        st.markdown('<div class="section-divider"><h2>Batch Results</h2></div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(render_metric_card("URLs Processed", f"{n:,}", f"in {elapsed:.1f}s", "info"), unsafe_allow_html=True)
        c2.markdown(render_metric_card("Phishing", f"{n_phish:,}", f"{n_phish/n*100:.1f}%", "danger"), unsafe_allow_html=True)
        c3.markdown(render_metric_card("Safe", f"{n_safe:,}", f"{n_safe/n*100:.1f}%", "success"), unsafe_allow_html=True)
        c4.markdown(render_metric_card("Avg Probability", f"{avg_prob:.2f}%", "Mean phishing score", "warning"), unsafe_allow_html=True)

        # Distribution chart
        import plotly.express as px
        hist_df = pd.DataFrame({"Probability (%)": result_df["probability"].values})
        fig = px.histogram(
            hist_df, x="Probability (%)", nbins=20,
            color_discrete_sequence=["#00d4ff"],
        )
        _dark_layout(fig)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Preview results
        st.markdown('<div class="section-divider"><h2>Predictions</h2></div>', unsafe_allow_html=True)
        st.dataframe(
            result_df.head(50),
            use_container_width=True,
            height=400,
            hide_index=True,
        )

        # Download
        st.markdown('<div class="section-divider"><h2>Download Results</h2></div>', unsafe_allow_html=True)
        csv_out = result_df.to_csv(index=False).encode()
        st.download_button(
            "Download predictions.csv",
            csv_out,
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
