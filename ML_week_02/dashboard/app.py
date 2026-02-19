"""
app.py – Main entry point for the Explainable AI Dashboard.

Run with:  streamlit run app.py --server.port 8502
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.utils.session_state import init_state

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="XAI Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load custom CSS ───────────────────────────────────────────
CSS_PATH = Path(__file__).parent / "assets" / "styles.css"
if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)

# ── Initialise session state ─────────────────────────────────
init_state()

# ── Navigation ────────────────────────────────────────────────
PAGES = {
    "🏠 Overview": "pages/1_overview.py",
    "🌐 Global Explanations": "pages/2_global_explanations.py",
    "🔬 Local Explanations": "pages/3_local_explanations.py",
    "⚖️ Fairness Analysis": "pages/4_fairness_analysis.py",
    "📊 Reports": "pages/5_reports.py",
}

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("XAI Dashboard")
    st.caption("Explainable AI · Fairness · Reports")
    st.divider()

    page = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")

    st.divider()

    # Pipeline status indicators
    from src.utils.session_state import stage_ready
    stages = [
        ("📁 Data", "data"),
        ("🤖 Model", "model"),
        ("🧠 Explain", "explain"),
        ("⚖️ Fairness", "fairness"),
        ("📊 Report", "report"),
    ]
    st.markdown("**Pipeline Status**")
    for label, stage in stages:
        icon = "✅" if stage_ready(stage) else "⬜"
        st.caption(f"{icon} {label}")

# ── Run selected page ─────────────────────────────────────────
page_path = Path(__file__).parent / PAGES[page]
if page_path.exists():
    exec(page_path.read_text(), {"__name__": "__page__"})
else:
    st.error(f"Page not found: {page_path}")
