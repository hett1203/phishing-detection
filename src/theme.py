"""Global dark cybersecurity theme for the Streamlit app."""
from __future__ import annotations

import streamlit as st


# Palette
# Background       #0a0e14   (deep near-black blue)
# Card background   #111824   (panel)
# Accent red        #ff4757   (phishing / danger)
# Accent green       #2ed573   (safe / neon)
# Accent cyan       #00d4ff   (info / charts)
# Text primary       #e6edf3   (off-white)
# Text secondary     #8b949e   (muted)

GLOBAL_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,400,0,0');

    .material-symbols-rounded {
        font-family: 'Material Symbols Rounded';
        font-weight: normal;
        font-style: normal;
        font-size: 1.25rem;
        line-height: 1;
        letter-spacing: normal;
        text-transform: none;
        display: inline-block;
        white-space: nowrap;
        word-wrap: normal;
        direction: ltr;
        -webkit-font-feature-settings: 'liga';
        -webkit-font-smoothing: antialiased;
        font-feature-settings: 'liga';
    }

    /* ---------- Base ---------- */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', 'Roboto', sans-serif;
    }

    .stApp {
        background: #0a0e14;
        color: #e6edf3;
    }

    /* Hide default Streamlit top padding */
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 4rem;
        max-width: 1280px;
    }

    /* ---------- Headings ---------- */
    h1, h2, h3, h4 {
        color: #e6edf3 !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    h1 { font-size: 2.2rem !important; }
    h2 { font-size: 1.6rem !important; }
    h3 { font-size: 1.25rem !important; }

    /* ---------- Cards ---------- */
    .metric-card {
        background: #111824;
        border: 1px solid rgba(0, 212, 255, 0.15);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 212, 255, 0.4);
    }

    .metric-card .label {
        font-size: 0.78rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }

    .metric-card .value {
        font-size: 2rem;
        font-weight: 800;
        margin-top: 0.25rem;
        letter-spacing: -0.02em;
    }

    .metric-card .sublabel {
        font-size: 0.78rem;
        color: #8b949e;
        margin-top: 0.25rem;
    }

    .metric-card.danger { border-color: rgba(255, 71, 87, 0.4); }
    .metric-card.danger .value { color: #ff4757; }
    .metric-card.success { border-color: rgba(46, 213, 115, 0.4); }
    .metric-card.success .value { color: #2ed573; }
    .metric-card.info { border-color: rgba(0, 212, 255, 0.4); }
    .metric-card.info .value { color: #00d4ff; }
    .metric-card.warning { border-color: rgba(251, 191, 36, 0.4); }
    .metric-card.warning .value { color: #fbbf24; }

    /* ---------- Alert banners ---------- */
    .alert-banner {
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        font-weight: 600;
        border-left: 4px solid;
        backdrop-filter: blur(8px);
    }
    .alert-danger {
        background: rgba(255, 71, 87, 0.1);
        border-left-color: #ff4757;
        color: #ffb3bd;
    }
    .alert-success {
        background: rgba(46, 213, 115, 0.1);
        border-left-color: #2ed573;
        color: #aef5c8;
    }
    .alert-info {
        background: rgba(0, 212, 255, 0.08);
        border-left-color: #00d4ff;
        color: #b8eaff;
    }

    /* ---------- Risk gauge ---------- */
    .risk-gauge-container {
        background: rgba(17, 24, 36, 0.8);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(0, 212, 255, 0.15);
    }
    .risk-gauge-bar {
        height: 12px;
        border-radius: 6px;
        background: #2ed573;
        margin: 1rem 0 0.5rem 0;
        position: relative;
    }
    .risk-gauge-thumb {
        position: absolute;
        top: -8px;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: #e6edf3;
        border: 3px solid #0a0e14;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.6);
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: #0b111b;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 1.25rem;
    }

    .brand-lockup {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        padding: 0.2rem 0.15rem;
    }

    .brand-mark {
        display: grid;
        place-items: center;
        width: 1.8rem;
        height: 1.8rem;
        color: #00d4ff;
    }

    .page-icon, .quick-icon, .inline-icon {
        display: inline-grid;
        place-items: center;
        vertical-align: middle;
        color: #00d4ff !important;
        background: transparent;
        line-height: 1;
    }

    .page-icon {
        width: 2.65rem;
        height: 2.65rem;
        margin-right: 0.55rem;
        border: 1px solid rgba(41, 199, 232, 0.45);
        border-radius: 10px;
        background: rgba(41, 199, 232, 0.1);
    }
    .page-icon .material-symbols-rounded {
        font-size: 1.55rem;
    }
    .page-icon.icon-green {
        border-color: rgba(57, 217, 138, 0.45);
        background: rgba(57, 217, 138, 0.1);
    }
    .page-icon.icon-red {
        border-color: rgba(242, 118, 120, 0.45);
        background: rgba(242, 118, 120, 0.1);
    }
    .page-icon.icon-yellow {
        border-color: rgba(242, 184, 75, 0.45);
        background: rgba(242, 184, 75, 0.1);
    }

    .quick-icon {
        width: 2.35rem;
        height: 2.35rem;
        border-radius: 9px;
        border: 1px solid currentColor;
        background: rgba(255, 255, 255, 0.04);
    }
    .quick-icon .material-symbols-rounded {
        font-size: 1.35rem;
    }

    .inline-icon {
        min-width: 1.65rem;
        height: 1.5rem;
        padding: 0;
        margin-right: 0.25rem;
        border-radius: 5px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid currentColor;
    }
    .inline-icon .material-symbols-rounded {
        font-size: 1rem;
    }

    .icon-cyan { color: #00d4ff !important; }
    .icon-green { color: #2ed573 !important; }
    .icon-red { color: #ff6675 !important; }
    .icon-yellow { color: #fbbf24 !important; }

    .hero-icon {
        display: grid;
        place-items: center;
        width: 2.65rem;
        height: 2.65rem;
        border: 1px solid rgba(41, 199, 232, 0.45);
        border-radius: 10px;
        background: rgba(41, 199, 232, 0.1);
        color: #00d4ff;
    }
    .hero-icon .material-symbols-rounded {
        font-size: 1.55rem;
    }

    .sidebar-logo {
        font-size: 1.12rem;
        color: #00d4ff;
        font-weight: 800;
        letter-spacing: 0.1em;
    }

    .sidebar-subtitle {
        font-size: 0.68rem;
        color: #8b949e;
        margin-top: 0.25rem;
    }

    .sidebar-status {
        margin-top: 0.85rem;
        color: #aef5c8;
        font-size: 0.7rem;
        font-weight: 600;
    }

    .status-dot {
        display: inline-block;
        width: 7px;
        height: 7px;
        margin-right: 0.4rem;
        border-radius: 50%;
        background: #2ed573;
        box-shadow: 0 0 8px rgba(46, 213, 115, 0.8);
    }

    .sidebar-section-label {
        color: #6e7681;
        font-size: 0.64rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin: 0.4rem 0 0.55rem;
    }

    .sidebar-divider {
        height: 1px;
        background: rgba(255, 255, 255, 0.08);
        margin: 0.85rem 0;
    }

    .sidebar-footer {
        margin: 0.75rem 0.15rem 0;
        font-size: 0.68rem;
        color: #6e7681;
        line-height: 1.55;
    }
    .sidebar-footer-title {
        color: #aab6c4;
        font-size: 0.65rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        margin-bottom: 0.2rem;
    }
    .sidebar-footer-meta { color: #4f5c6d; margin-top: 0.25rem; }

    /* In-app navigation rows avoid full browser reloads and URL flashes. */
    .stSidebar .stButton {
        margin: 0 !important;
    }
    .stSidebar .stButton > button {
        justify-content: flex-start;
        min-height: 2rem;
        margin: 0 !important;
        padding: 0.35rem 0.55rem !important;
        border: 0 !important;
        border-left: 2px solid transparent !important;
        border-radius: 0 6px 6px 0 !important;
        background: transparent !important;
        color: #aab6c4 !important;
        box-shadow: none !important;
        transform: none !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        text-align: left;
    }
    .stSidebar .stButton > button:hover {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #e6edf3 !important;
    }
    .stSidebar .stButton > button[kind="primary"] {
        background: rgba(0, 212, 255, 0.1) !important;
        border-left-color: #00d4ff !important;
        color: #e6edf3 !important;
    }
    .stSidebar .st-key-nav_home [data-testid="stIconMaterial"] { color: #29c7e8 !important; }
    .stSidebar .st-key-nav_dataset_overview [data-testid="stIconMaterial"] { color: #39d98a !important; }
    .stSidebar .st-key-nav_eda_dashboard [data-testid="stIconMaterial"] { color: #f2b84b !important; }
    .stSidebar .st-key-nav_single_url_checker [data-testid="stIconMaterial"] { color: #f27678 !important; }
    .stSidebar .st-key-nav_batch_prediction [data-testid="stIconMaterial"] { color: #39d98a !important; }
    .stSidebar .st-key-nav_explainability [data-testid="stIconMaterial"] { color: #af91ee !important; }
    .stSidebar .st-key-nav_model_performance [data-testid="stIconMaterial"] { color: #f27678 !important; }
    .stSidebar .st-key-nav_about [data-testid="stIconMaterial"] { color: #29c7e8 !important; }

    /* ---------- Buttons ---------- */
    .stButton > button {
        background: #00b8d4 !important;
        color: #0a0e14 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: #00c2dc !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 212, 255, 0.3);
    }

    /* ---------- Inputs ---------- */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > div {
        background-color: #111824 !important;
        color: #e6edf3 !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        border-radius: 8px !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #00d4ff !important;
        box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.15) !important;
    }

    /* ---------- Tables ---------- */
    .stDataFrame, .stTable {
        background: transparent;
    }
    .stDataFrame table, .stTable table {
        color: #e6edf3;
    }
    .stDataFrame thead th, .stTable thead th {
        background: #111824 !important;
        color: #00d4ff !important;
        font-weight: 700;
        border-bottom: 1px solid rgba(0, 212, 255, 0.3);
    }
    .stDataFrame tbody tr, .stTable tbody tr {
        background: rgba(17, 24, 36, 0.5) !important;
    }
    .stDataFrame tbody tr:hover, .stTable tbody tr:hover {
        background: rgba(0, 212, 255, 0.08) !important;
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: rgba(17, 24, 36, 0.6);
        border-radius: 10px;
        padding: 0.3rem;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8b949e !important;
        padding: 0.5rem 1.2rem !important;
        border-radius: 8px !important;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #00b8d4 !important;
        color: #0a0e14 !important;
    }

    /* ---------- Spinner ---------- */
    .stSpinner > div > div {
        border-color: #00d4ff transparent transparent transparent !important;
    }

    /* ---------- Page hero ---------- */
    .page-hero {
        background: #111824;
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
    }

    .page-hero h1 {
        display: block;
        white-space: nowrap;
        color: #39d98a;
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: currentColor;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem;
    }

    .page-hero h1 > span {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        white-space: nowrap;
    }

    .page-hero .subtitle {
        color: #8b949e;
        font-size: 1rem;
        margin-bottom: 0;
    }

    /* ---------- Tag chips ---------- */
    .chip {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 100px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .chip-red { background: rgba(255, 71, 87, 0.15); color: #ff7a85; border: 1px solid rgba(255, 71, 87, 0.3); }
    .chip-green { background: rgba(46, 213, 115, 0.15); color: #5be089; border: 1px solid rgba(46, 213, 115, 0.3); }
    .chip-cyan { background: rgba(0, 212, 255, 0.15); color: #5bd9ff; border: 1px solid rgba(0, 212, 255, 0.3); }
    .chip-yellow { background: rgba(251, 191, 36, 0.15); color: #fcd34d; border: 1px solid rgba(251, 191, 36, 0.3); }

    /* ---------- Section dividers ---------- */
    .section-divider {
        display: flex;
        align-items: center;
        margin: 2rem 0 1rem 0;
    }
    .section-divider::after {
        content: '';
        flex: 1;
        height: 1px;
        background: rgba(0, 212, 255, 0.3);
        margin-left: 1rem;
    }
    .section-divider h2 {
        margin: 0;
        color: #00d4ff !important;
        font-size: 1.1rem !important;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* ---------- Plotly fix ---------- */
    .js-plotly-plot .plotly .modebar {
        background: transparent !important;
    }

    /* ---------- Code blocks ---------- */
    .stCode > div {
        background: #0d1320 !important;
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 8px;
    }

    /* ---------- File uploader ---------- */
    .stFileUploader {
        border: 2px dashed rgba(0, 212, 255, 0.3) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        background: rgba(17, 24, 36, 0.5) !important;
    }

    /* ---------- Tooltip ---------- */
    .tooltip-text {
        color: #8b949e;
        font-size: 0.8rem;
        font-style: italic;
    }

    /* ---------- Plain language explanation list ---------- */
    .explain-list {
        list-style: none;
        padding: 0;
        margin: 1rem 0;
    }
    .explain-list li {
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        background: rgba(17, 24, 36, 0.6);
        border-left: 3px solid #00d4ff;
        border-radius: 6px;
        font-size: 0.92rem;
        line-height: 1.4;
        color: #c9d1d9;
    }
    .explain-list li strong {
        color: #00d4ff;
    }

    /* ---------- Animations ---------- */
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 0 0 rgba(255, 71, 87, 0.4); }
        50% { box-shadow: 0 0 0 8px rgba(255, 71, 87, 0); }
    }
    .pulse-danger {
        animation: pulse-glow 2s infinite;
    }
    @keyframes fade-in {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in { animation: fade-in 0.4s ease-out; }
</style>
"""


def apply_global_theme() -> None:
    """Inject the global CSS into the Streamlit app."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def render_metric_card(
    label: str,
    value: str,
    sublabel: str = "",
    variant: str = "info",
) -> str:
    """Return HTML for a single metric card."""
    return f"""
    <div class="metric-card {variant} fade-in">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {f'<div class="sublabel">{sublabel}</div>' if sublabel else ''}
    </div>
    """


def render_alert(message: str, variant: str = "info") -> str:
    """Return HTML for an alert banner."""
    return f'<div class="alert-banner alert-{variant}">{message}</div>'
