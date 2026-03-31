"""
Real-time Anomaly Detection Dashboard
Main Streamlit application entry point.

This is a thin orchestrator — all logic lives in the app/ package:
  app/config.py     – YAML loading, simulator/model factories
  app/state.py      – Session state initialization, calibration
  app/sidebar.py    – Sidebar controls and settings
  app/streaming.py  – Real-time streaming loop, static rendering
"""

import streamlit as st

from app.config import load_config
from app.state import init_session_state
from app.sidebar import render_sidebar
from app.streaming import run_streaming_loop, render_static


def main():
    st.set_page_config(
        page_title="Real-time Anomaly Detector",
        page_icon="📡",
        layout="wide",
    )
    st.title("📡 Real-time Anomaly Detection Dashboard")

    # Load config and initialize session state
    cfg = load_config()
    init_session_state(cfg)

    # Sidebar controls
    settings = render_sidebar(cfg)

    # Layout: metrics row → chart → alerts table
    col1, col2, col3, col4 = st.columns(4)
    placeholders = {
        "total": col1.empty(),
        "rate": col2.empty(),
        "score": col3.empty(),
        "alerts": col4.empty(),
        "chart": st.empty(),
    }
    st.subheader("Recent Alerts")
    placeholders["alerts_table"] = st.empty()

    # Run or display static
    if st.session_state.running:
        run_streaming_loop(placeholders, cfg, settings.speed_ms, settings.batch_size)
    else:
        render_static(placeholders)


if __name__ == "__main__":
    main()
