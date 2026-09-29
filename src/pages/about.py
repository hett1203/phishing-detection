"""About page — project overview, methodology, references."""
from __future__ import annotations

import streamlit as st

from predict import load_best_meta, load_metrics


def render() -> None:
    meta = load_best_meta()
    best = meta["best_model"]
    m = meta["metrics"]

    st.markdown(
        """
        <div class="page-hero fade-in">
            <h1><span class="page-icon icon-cyan"><span class="material-symbols-rounded">info</span></span> About the Project</h1>
            <div class="subtitle">Phishing URL Detection — methodology, technology stack, and references</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Mission
    st.markdown('<div class="section-divider"><h2>Mission</h2></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="metric-card info" style="padding:1.5rem 2rem;">
        <p style="font-size:1rem; line-height:1.7; color:#c9d1d9; margin:0;">
        PhishGuard is an end-to-end <strong style="color:#00d4ff">machine-learning system</strong>
        designed to automatically distinguish phishing websites from legitimate ones using
        pre-engineered lexical, host-based, and content-based URL features.
        </p>
        <p style="font-size:0.95rem; line-height:1.7; color:#8b949e; margin-top:1rem;">
        The system ingests a raw URL string, extracts 22 numeric features and a categorical
        top-level-domain signal, runs a trained <strong style="color:#2ed573">{best}</strong>
        classifier, and outputs a verdict with a probability, a risk score, and a
        plain-language explanation of why the URL was flagged.
        </p>
        </div>
        """.format(best=best),
        unsafe_allow_html=True,
    )

    # Tech stack
    st.markdown('<div class="section-divider"><h2>Technology Stack</h2></div>', unsafe_allow_html=True)
    cols = st.columns(4)
    stack = [
        ("Language", "Python 3.11+", "cyan"),
        ("ML Framework", "scikit-learn · XGBoost · LightGBM · CatBoost", "green"),
        ("Explainability", "SHAP · TreeExplainer", "cyan"),
        ("Visualization", "Plotly · Streamlit", "green"),
        ("Data", "Pandas · NumPy", "cyan"),
        ("Storage", "Joblib · JSON · CSV artifacts", "green"),
        ("UI Theme", "Dark Cybersecurity · Neon accent", "cyan"),
        ("Deployment", "Local · Streamlit dashboard", "green"),
    ]
    for col, (label, value, color) in zip(cols, stack):
        col.markdown(
            f"""
            <div class="metric-card chip-{color}" style="padding:1rem; text-align:center;">
                <div style="font-size:0.7rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em;">{label}</div>
                <div style="font-size:0.85rem; color:#e6edf3; font-weight:600; margin-top:0.4rem; line-height:1.3;">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Methodology
    st.markdown('<div class="section-divider"><h2>Methodology</h2></div>', unsafe_allow_html=True)
    steps = [
        ("1. Data Ingestion",
         "116,586 labeled URLs (after dropping 14 NaN rows) loaded from the uploaded CSV. Features include URL length, domain length, character counts, ratios, entropy, and the top-level domain (TLD) as a categorical variable."),
        ("2. Preprocessing",
         "Top-50 most-frequent TLDs get their own one-hot slot; rare TLDs collapse into an 'other' bucket. Numeric features pass through unchanged. The fitted preprocessor is saved as a joblib artifact."),
        ("3. Train/Test Split",
         "Stratified 80/20 split preserving the ~14% phishing class ratio. 93,268 training rows · 23,318 test rows."),
        ("4. Model Training",
         "10 classifiers trained: RandomForest, ExtraTrees, XGBoost, LightGBM, CatBoost, GradientBoosting, HistGradientBoosting, AdaBoost, Bagging, LogisticRegression. All configured for class imbalance via class_weight='balanced' or scale_pos_weight."),
        ("5. Evaluation",
         "Each model scored on Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, MCC, Log Loss, and Confusion Matrix. Best model selected by phishing-recall priority (missing a phishing site is the worst failure)."),
        ("6. Explainability",
         "SHAP TreeExplainer computed on 2,000-sample held-out subset. Mean |SHAP| per feature surfaces the global drivers of the model's decisions."),
        ("7. Deployment",
         "Streamlit dashboard with dark cybersecurity theme — 8 pages covering home, dataset overview, EDA, single URL checker, batch prediction, explainability, model performance, and about."),
    ]
    for title, desc in steps:
        st.markdown(
            f"""
            <div class="metric-card info" style="padding:1rem 1.4rem; margin-bottom:0.6rem;">
                <div style="font-size:0.95rem; font-weight:700; color:#00d4ff; margin-bottom:0.4rem;">{title}</div>
                <div style="font-size:0.88rem; color:#c9d1d9; line-height:1.6;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Best model summary
    st.markdown('<div class="section-divider"><h2>Best Model Summary</h2></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="metric-card success" style="padding:1.5rem 2rem;">
            <div style="font-size:1.3rem; font-weight:800; color:#2ed573; margin-bottom:0.5rem;">{best}</div>
            <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:1rem; margin-top:1rem;">
                <div>
                    <div style="font-size:0.7rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em;">Accuracy</div>
                    <div style="font-size:1.2rem; font-weight:700; color:#e6edf3;">{m['accuracy']*100:.2f}%</div>
                </div>
                <div>
                    <div style="font-size:0.7rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em;">Recall (Phish)</div>
                    <div style="font-size:1.2rem; font-weight:700; color:#ff4757;">{m['recall_phish']*100:.2f}%</div>
                </div>
                <div>
                    <div style="font-size:0.7rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em;">F1 Score</div>
                    <div style="font-size:1.2rem; font-weight:700; color:#e6edf3;">{m['f1_phish']*100:.2f}%</div>
                </div>
                <div>
                    <div style="font-size:0.7rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em;">ROC-AUC</div>
                    <div style="font-size:1.2rem; font-weight:700; color:#00d4ff;">{m['roc_auc']:.4f}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Limitations
    st.markdown('<div class="section-divider"><h2>Limitations & Future Scope</h2></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="metric-card warning" style="padding:1.5rem 2rem;">
        <ul class="explain-list" style="margin:0;">
            <li><strong>Static features only</strong> — model uses pre-extracted lexical features. Live URL feature extraction (HTTP fetch, WHOIS, DNS) would improve detection of new phishing techniques.</li>
            <li><strong>Dataset size</strong> — 116k rows is reasonable but a larger, more diverse sample would improve generalization to underrepresented TLDs and novel attack patterns.</li>
            <li><strong>Class imbalance</strong> — ~14% phishing class means recall priority is critical. Class weighting helps but a cost-sensitive threshold tuned to a target false-negative rate could be added in production.</li>
            <li><strong>Adversarial drift</strong> — phishing tactics evolve; periodic retraining with newly labeled URLs is recommended.</li>
            <li><strong>Future scope</strong> — browser extension deployment, real-time threat-intel feed integration, and benchmarking against published UCI Phishing Websites Dataset results.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # References
    st.markdown('<div class="section-divider"><h2>References</h2></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="metric-card info" style="padding:1.5rem 2rem;">
        <ul class="explain-list" style="margin:0;">
            <li><strong>scikit-learn</strong> — Pedregosa et al., 2011. <em>Scikit-learn: Machine Learning in Python.</em> JMLR 12, pp. 2825–2830.</li>
            <li><strong>XGBoost</strong> — Chen & Guestrin, 2016. <em>XGBoost: A Scalable Tree Boosting System.</em> KDD.</li>
            <li><strong>LightGBM</strong> — Ke et al., 2017. <em>LightGBM: A Highly Efficient Gradient Boosting Decision Tree.</em> NeurIPS.</li>
            <li><strong>CatBoost</strong> — Prokhorenkova et al., 2018. <em>CatBoost: unbiased boosting with categorical features.</em> NeurIPS.</li>
            <li><strong>SHAP</strong> — Lundberg & Lee, 2017. <em>A Unified Approach to Interpreting Model Predictions.</em> NeurIPS.</li>
            <li><strong>HistGradientBoosting</strong> — scikit-learn implementation based on LightGBM algorithm.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
