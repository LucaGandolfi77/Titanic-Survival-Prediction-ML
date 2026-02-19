"""
ML Playground – Interactive Machine Learning Dashboard
=====================================================

Main Streamlit application entry-point (Home page).
Run with:
    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from src.utils.session_state import init_session_state
from src.ui.sidebar import render_sidebar
from src.ui.data_upload import render_upload_section, render_target_selector

# ── Page configuration ────────────────────────────────────────
st.set_page_config(
    page_title="ML Playground",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_path = _ROOT / "assets" / "styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# Initialise session state
init_session_state()

# ── Sidebar ───────────────────────────────────────────────────
render_sidebar()

# ── Main content ──────────────────────────────────────────────
st.title("🧪 ML Playground")
st.markdown(
    """
    Welcome to the **Interactive ML Playground** — a no-code dashboard for
    exploring data, training machine learning models, and comparing results.

    ### 🚀 Getting Started

    1. **Upload** your own CSV/Excel dataset or pick a sample below.
    2. **Explore** your data in the 📊 **Data Explorer** page.
    3. **Train** multiple models with one click on the 🤖 **Model Training** page.
    4. **Compare** performance on the 📈 **Results** dashboard.
    5. **Predict** new observations on the 🔮 **Predictions** page.

    ---
    """
)

# ── Data loading section ──────────────────────────────────────
render_upload_section()

st.markdown("---")

# ── Target selection ──────────────────────────────────────────
render_target_selector()

# ── Quick preview ─────────────────────────────────────────────
if st.session_state.get("df") is not None:
    st.markdown("---")
    st.subheader("👀 Dataset Preview")
    st.dataframe(st.session_state.df.head(20), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", st.session_state.df.shape[0])
    with col2:
        st.metric("Columns", st.session_state.df.shape[1])
    with col3:
        missing = st.session_state.df.isnull().sum().sum()
        st.metric("Missing values", int(missing))

    st.info("👉 Head to **📊 Data Explorer** in the sidebar to dig deeper!")
