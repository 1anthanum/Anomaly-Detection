"""
Sidebar controls: start/pause/reset buttons, settings sliders,
and optional Chronos baseline toggle.
"""

import logging
from dataclasses import dataclass

import streamlit as st

from .state import calibrate

logger = logging.getLogger(__name__)


@dataclass
class SidebarSettings:
    """Values returned by the sidebar controls."""
    speed_ms: int
    batch_size: int


def _init_chronos():
    """Lazily initialise the Chronos detector into session state."""
    if "chronos_detector" in st.session_state:
        return  # already initialised (or explicitly set to None)

    try:
        from src.models.chronos_baseline import ChronosBaseline, CHRONOS_AVAILABLE

        if not CHRONOS_AVAILABLE:
            st.session_state.chronos_detector = None
            return

        with st.spinner("Loading Chronos model (first time only)..."):
            st.session_state.chronos_detector = ChronosBaseline(
                model_name="amazon/chronos-t5-tiny",
                context_length=64,
            )
    except Exception as e:
        logger.warning("Failed to load Chronos: %s", e)
        st.session_state.chronos_detector = None


def render_sidebar(cfg: dict) -> SidebarSettings:
    """Render the sidebar and return user-selected settings."""
    with st.sidebar:
        st.header("Controls")

        if st.button("▶️ Start / Resume"):
            if not st.session_state.calibrated:
                with st.spinner("Calibrating model on normal data..."):
                    calibrate(cfg)
            st.session_state.running = True

        if st.button("⏸️ Pause"):
            st.session_state.running = False

        if st.button("🔄 Reset"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.divider()
        st.subheader("Settings")
        speed = st.slider("Update speed (ms)", 50, 1000, cfg["dashboard"]["update_interval_ms"])
        batch_size = st.slider("Points per update", 1, 10, 1)

        st.divider()
        st.subheader("Model")
        st.text(f"Type: {cfg['model']['default'].upper()}")
        st.text(f"Window: {cfg['model']['window_size']}")
        st.text(f"Threshold: p{cfg['detection']['threshold_percentile']}")

        # Chronos baseline toggle
        st.divider()
        st.subheader("Baseline Comparison")
        chronos_enabled = st.checkbox(
            "Enable Chronos overlay",
            value=False,
            help="Show Chronos foundation model anomaly markers alongside the autoencoder. "
                 "Requires chronos-forecasting to be installed.",
        )
        if chronos_enabled:
            _init_chronos()
            if st.session_state.get("chronos_detector") is None:
                st.warning("Chronos unavailable. Install: `pip install chronos-forecasting`")
        else:
            st.session_state.chronos_detector = None

    return SidebarSettings(speed_ms=speed, batch_size=batch_size)
