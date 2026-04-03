"""
Real-time streaming loop and static (paused) rendering.

This module owns the core detection cycle:
generate → window → score → alert → render.
"""

import time
import numpy as np
import pandas as pd
import torch
import streamlit as st

from src.visualization import create_timeseries_chart, render_metrics, format_alert_table


def _update_display(placeholders: dict, score: float):
    """Push current state to all Streamlit placeholders."""
    dash = st.session_state.dash_state
    alert_engine = st.session_state.alert_engine

    m = render_metrics(
        st.session_state.total_points,
        st.session_state.detected_anomalies,
        score,
        alert_engine.get_stats(),
    )
    placeholders["total"].metric("Total Points", m["total_points"])
    placeholders["rate"].metric("Anomaly Rate", m["anomaly_rate"])
    placeholders["score"].metric("Current Score", m["current_score"])
    placeholders["alerts"].metric("Alerts", m["total_alerts"])

    placeholders["chart"].plotly_chart(
        create_timeseries_chart(dash),
        use_container_width=True,
        key=f"chart_{dash.steps[-1] if dash.steps else 0}",
    )

    recent = alert_engine.get_recent(15)
    if recent:
        placeholders["alerts_table"].dataframe(
            pd.DataFrame(format_alert_table(recent)),
            use_container_width=True,
            hide_index=True,
        )
    else:
        placeholders["alerts_table"].info("No alerts yet.")


def run_streaming_loop(placeholders: dict, cfg: dict, speed_ms: int, batch_size: int):
    """Run the real-time detection loop (blocking while running)."""
    window_size = cfg["model"]["window_size"]

    while st.session_state.running:
        sim = st.session_state.simulator
        windower = st.session_state.windower
        model = st.session_state.model
        scorer = st.session_state.scorer
        alert_engine = st.session_state.alert_engine
        dash = st.session_state.dash_state

        score = 0.0

        for _ in range(batch_size):
            point = sim.generate_point()
            value = point["value"]
            step = point["step"]
            st.session_state.value_buffer.append(value)
            if len(st.session_state.value_buffer) > window_size * 2:
                st.session_state.value_buffer = st.session_state.value_buffer[-window_size:]
            st.session_state.total_points += 1

            score = 0.0
            threshold = 0.0
            is_detected = False

            # Score when we have enough data for a window
            if len(st.session_state.value_buffer) >= window_size:
                window_data = np.array(st.session_state.value_buffer[-window_size:])
                normalized = windower.normalize(window_data)
                tensor = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(-1)

                with torch.no_grad():
                    raw_error = float(model.anomaly_score(tensor)[0])

                result = scorer.score(raw_error)
                score = result.score
                threshold = result.threshold_score
                is_detected = result.is_anomaly

                if is_detected:
                    st.session_state.detected_anomalies += 1
                    alert_engine.check(result, step, value)

            dash.append(step, value, score, threshold, is_detected)

        _update_display(placeholders, score)
        time.sleep(speed_ms / 1000)


def render_static(placeholders: dict):
    """Render the dashboard in its current (paused) state."""
    dash = st.session_state.dash_state

    if dash.steps:
        last_score = dash.scores[-1] if dash.scores else 0
        _update_display(placeholders, last_score)
    else:
        st.info("Click **▶️ Start / Resume** in the sidebar to begin streaming.")
