"""Single-URL checker page — interactive phishing prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from features import extract_url_features
from predict import Predictor, get_security_tip, load_best_meta
from theme import render_metric_card, render_alert


SAMPLE_URLS = [
    "https://www.google.com/search?q=streamlit",
    "https://bit.ly/3OuSYSo",
    "https://ln.run/nTbnB",
    "https://www.umfk.edu/",
    "https://valndave1400.wixsite.com/my-site-1",
    "https://rakuten-cord-co-jp.968bet.info/inmax/",
    "http://www.ibm.com/",
    "https://l.ead.me/bfYk65",
]


def _risk_color(prob: float) -> str:
    if prob < 0.3:
        return "#2ed573"   # green
    if prob < 0.6:
        return "#fbbf24"   # yellow
    return "#ff4757"       # red


def render() -> None:
    st.markdown(
        """
        <div class="page-hero fade-in">
            <h1><span class="page-icon icon-cyan"><span class="material-symbols-rounded">search</span></span> Single URL Checker</h1>
            <div class="subtitle">Paste any URL and get an instant verdict, risk score, and per-feature explanation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Input section
    col_input, col_sample = st.columns([3, 1])
    with col_sample:
        st.markdown("##### Try a sample")
        sample = st.selectbox(
            "Pick an example URL",
            options=SAMPLE_URLS,
            index=0,
            label_visibility="collapsed",
        )

    with col_input:
        url = st.text_input(
            "Enter the URL to inspect",
            value=sample,
            placeholder="https://example.com/path?query=1",
        )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    analyze = st.button(":material/search: Analyze URL", use_container_width=True)

    if not analyze:
        st.markdown(
            """
            <div class="metric-card info" style="padding:1.5rem 2rem; text-align:center;">
                <div style="font-size:1rem; color:#8b949e;"><span class="inline-icon icon-cyan"><span class="material-symbols-rounded">touch_app</span></span> Enter a URL above and click <strong style="color:#00d4ff">Analyze URL</strong> to begin</div>
                <div style="font-size:0.85rem; color:#6e7681; margin-top:0.5rem;">
                The model will extract 22 lexical + host features and output a phishing probability,
                a risk gauge, and a plain-language explanation of the verdict.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if not url or not url.strip():
        st.markdown(render_alert("Please enter a URL first.", "warning"), unsafe_allow_html=True)
        return

    with st.spinner("Extracting features and running prediction..."):
        # Extract features
        features = extract_url_features(url)
        features_df = pd.DataFrame([features])

        predictor = Predictor.get()
        proba = float(predictor.predict_features_df(features_df)[0])
        contributions = predictor.contributions_for(features_df)

    # Verdict banner
    is_phishing = proba >= 0.5
    verdict = "PHISHING DETECTED" if is_phishing else "SAFE / LEGITIMATE"
    alert_variant = "danger" if is_phishing else "success"
    pulse_class = " pulse-danger" if is_phishing else ""
    st.markdown(
        f"""
        <div class="alert-banner alert-{alert_variant}{pulse_class}" style="font-size:1.3rem; text-align:center; padding:1.5rem;">
            {verdict}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Probability + risk gauge
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_a:
        st.markdown(
            render_metric_card(
                "Phishing Probability",
                f"{proba * 100:.2f}%",
                "Model confidence",
                variant="danger" if is_phishing else "success",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            render_metric_card(
                "Verdict",
                "Phishing" if is_phishing else "Safe",
                f"Threshold: 0.50",
                variant="danger" if is_phishing else "success",
            ),
            unsafe_allow_html=True,
        )

    with col_b:
        risk_score = int(proba * 100)
        thumb_pos = max(0, min(100, risk_score))
        color = _risk_color(proba)
        label_text = "HIGH RISK" if proba >= 0.6 else ("MEDIUM RISK" if proba >= 0.3 else "LOW RISK")
        st.markdown(
            f"""
            <div class="risk-gauge-container">
                <div style="font-size:0.8rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.1em;">Risk Score</div>
                <div style="font-size:3rem; font-weight:800; color:{color}; margin-top:0.3rem; line-height:1;">{risk_score}</div>
                <div style="font-size:0.85rem; color:{color}; font-weight:600; margin-top:0.2rem;">{label_text}</div>
                <div class="risk-gauge-bar">
                    <div class="risk-gauge-thumb" style="left: calc({thumb_pos}% - 14px);"></div>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.7rem; color:#6e7681; margin-top:0.3rem;">
                    <span>SAFE</span>
                    <span>SUSPICIOUS</span>
                    <span>PHISHING</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_c:
        st.markdown(
            render_metric_card(
                "Top Trigger",
                contributions[0][0].replace("tld_", "TLD: ") if contributions else "—",
                "Highest contributing signal",
                variant="danger" if is_phishing else "success",
            ),
            unsafe_allow_html=True,
        )
        # Class probability bars
        st.markdown(
            f"""
            <div class="metric-card info" style="padding:1rem 1.2rem;">
                <div style="font-size:0.78rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.6rem;">Class Probabilities</div>
                <div style="display:flex; align-items:center; margin-bottom:0.4rem;">
                    <div style="width:80px; font-size:0.85rem; color:#2ed573; font-weight:600;">Safe</div>
                    <div style="flex:1; height:8px; background:rgba(255,255,255,0.05); border-radius:4px; overflow:hidden; margin:0 0.6rem;">
                        <div style="width:{(1-proba)*100:.1f}%; height:100%; background:#2ed573;"></div>
                    </div>
                    <div style="width:50px; text-align:right; font-size:0.85rem; color:#2ed573; font-weight:700;">{(1-proba)*100:.1f}%</div>
                </div>
                <div style="display:flex; align-items:center;">
                    <div style="width:80px; font-size:0.85rem; color:#ff4757; font-weight:600;">Phishing</div>
                    <div style="flex:1; height:8px; background:rgba(255,255,255,0.05); border-radius:4px; overflow:hidden; margin:0 0.6rem;">
                        <div style="width:{proba*100:.1f}%; height:100%; background:#ff4757;"></div>
                    </div>
                    <div style="width:50px; text-align:right; font-size:0.85rem; color:#ff4757; font-weight:700;">{proba*100:.1f}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Extracted features table
    st.markdown('<div class="section-divider"><h2>Extracted Features</h2></div>', unsafe_allow_html=True)
    feat_df = pd.DataFrame(
        [(k, v) for k, v in features.items() if k != "tld"],
        columns=["Feature", "Value"]
    )
    feat_df["Feature"] = feat_df["Feature"].str.replace("_", " ").str.title()
    st.dataframe(feat_df, use_container_width=True, height=420, hide_index=True)

    # Per-feature explanations
    st.markdown('<div class="section-divider"><h2>Why this Verdict? — Plain-Language Explanation</h2></div>', unsafe_allow_html=True)
    top_contribs = contributions[:6]
    explain_items = []
    for feat, score in top_contribs:
        if score < 0.001:
            continue
        tip = get_security_tip(feat)
        display = feat.replace("tld_", "TLD: ") if feat.startswith("tld_") else feat.replace("_", " ").title()
        explain_items.append(f"<li><strong>{display}</strong> · value signal {score:.2f} — {tip}</li>")
    if not explain_items:
        explain_items.append("<li>No single dominant signal — the URL closely matches legitimate patterns.</li>")
    st.markdown(
        f'<ul class="explain-list">{"".join(explain_items)}</ul>',
        unsafe_allow_html=True,
    )

    # Security tip
    if is_phishing:
        st.markdown(
            """
            <div class="alert-banner alert-danger" style="margin-top:1rem;">
            <span class="inline-icon icon-red"><span class="material-symbols-rounded">warning</span></span> <strong>Recommendation:</strong> Do not enter credentials, payment info, or personal data on this site.
            Verify the URL through official channels before interacting.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="alert-banner alert-success" style="margin-top:1rem;">
            <span class="inline-icon icon-green"><span class="material-symbols-rounded">check_circle</span></span> <strong>This URL appears legitimate.</strong> Always remain cautious and verify the
            certificate, domain spelling, and content before entering sensitive information.
            </div>
            """,
            unsafe_allow_html=True,
        )
