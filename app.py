"""Streamlit app entry point — Phishing URL Detection dashboard.

Run with:
    streamlit run /home/z/my-project/phishing_app/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make `src/` importable.
SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from pages import (
    home,
    dataset,
    eda,
    single_checker,
    batch,
    explainability,
    model_performance,
    about,
)

st.set_page_config(
    page_title="PhishGuard | Phishing URL Detection",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject the global dark cybersecurity theme CSS.
from theme import apply_global_theme  # noqa: E402
apply_global_theme()

# Sidebar navigation
with st.sidebar:
    st.markdown(
        """
        <div class="brand-lockup">
            <div class="brand-mark"><span class="material-symbols-rounded">shield</span></div>
            <div>
                <div class="sidebar-logo">PHISHGUARD</div>
                <div class="sidebar-subtitle">Threat intelligence workspace</div>
            </div>
        </div>
        <div class="sidebar-status"><span class="status-dot"></span> Detection engine online</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

PAGES = {
    "Home": home.render,
    "Dataset Overview": dataset.render,
    "EDA Dashboard": eda.render,
    "Single URL Checker": single_checker.render,
    "Batch Prediction": batch.render,
    "Explainability": explainability.render,
    "Model Performance": model_performance.render,
    "About": about.render,
}

PAGE_ICONS = {
    "Home": ":material/home:",
    "Dataset Overview": ":material/database:",
    "EDA Dashboard": ":material/analytics:",
    "Single URL Checker": ":material/search:",
    "Batch Prediction": ":material/folder_open:",
    "Explainability": ":material/lightbulb:",
    "Model Performance": ":material/monitoring:",
    "About": ":material/info:",
}

st.sidebar.markdown('<div class="sidebar-section-label">Workspace</div>', unsafe_allow_html=True)
if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Home"

for page_name, page_icon in PAGE_ICONS.items():
    is_selected = st.session_state.selected_page == page_name
    if st.sidebar.button(
        page_name,
        key=f"nav_{page_name.lower().replace(' ', '_')}",
        icon=page_icon,
        type="primary" if is_selected else "secondary",
        use_container_width=True,
    ):
        st.session_state.selected_page = page_name
        st.rerun()

selection = st.session_state.selected_page

st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

# Footer in sidebar
st.sidebar.markdown(
    """
    <div class="sidebar-footer">
        <div class="sidebar-footer-title">PHISHGUARD OPS</div>
        <div>Model: HistGradientBoosting</div>
        <div>Coverage: 116,586 URLs</div>
        <div class="sidebar-footer-meta">v1.0 · Local analysis console</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Render the selected page
PAGES[selection]()
