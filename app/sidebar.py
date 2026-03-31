"""
Sidebar controls: start/pause/reset buttons and settings sliders.
"""

from dataclasses import dataclass
import streamlit as st

from .state import calibrate


@dataclass
class SidebarSettings:
    """Values returned by the sidebar controls."""
    speed_ms: int
    batch_size: int


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

    return SidebarSettings(speed_ms=speed, batch_size=batch_size)
