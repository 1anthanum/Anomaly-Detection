"""
Session state initialization and model calibration.

Manages the lifecycle of all Streamlit session-state objects
(simulator, model, scorer, alert engine, dashboard state).
"""

import torch
import streamlit as st

from src.data import TimeSeriesSimulator, SimulatorConfig, AnomalyConfig, TimeSeriesWindower
from src.detection import AnomalyScorer, AlertEngine
from src.visualization import DashboardState

from .config import build_simulator, build_model


def init_session_state(cfg: dict):
    """Initialize all Streamlit session state objects (idempotent)."""
    if "initialized" in st.session_state:
        return

    model_cfg = cfg["model"]
    det_cfg = cfg["detection"]
    dash_cfg = cfg["dashboard"]

    # Core components
    st.session_state.simulator = build_simulator(cfg)
    st.session_state.model = build_model(cfg)
    st.session_state.model.eval()

    st.session_state.windower = TimeSeriesWindower(window_size=model_cfg["window_size"])
    st.session_state.scorer = AnomalyScorer(
        threshold_percentile=det_cfg["threshold_percentile"],
        window_size=det_cfg["history_window"],
    )
    st.session_state.alert_engine = AlertEngine(
        cooldown_steps=det_cfg["alert_cooldown_steps"],
        max_alerts=dash_cfg["max_alerts_display"],
    )

    # Dashboard state
    st.session_state.dash_state = DashboardState.create(max_display=dash_cfg["chart_history"])
    st.session_state.value_buffer = []
    st.session_state.detected_anomalies = 0
    st.session_state.total_points = 0
    st.session_state.running = False
    st.session_state.calibrated = False

    st.session_state.initialized = True


def calibrate(cfg: dict):
    """Calibrate the model and scorer using simulated normal data."""
    windower = st.session_state.windower
    model = st.session_state.model
    scorer = st.session_state.scorer

    # Generate calibration data (normal traffic, no anomalies)
    sim_cfg = cfg["simulator"]
    cal_sim = TimeSeriesSimulator(
        SimulatorConfig(
            base_value=sim_cfg["base_value"],
            daily_amplitude=sim_cfg["daily_amplitude"],
            weekly_amplitude=sim_cfg["weekly_amplitude"],
            noise_std=sim_cfg["noise_std"],
            sampling_rate=sim_cfg["sampling_rate"],
            anomaly=AnomalyConfig(point_prob=0, contextual_prob=0, collective_prob=0),
        )
    )
    cal_batch = cal_sim.generate_batch(1000)
    cal_values = cal_batch["values"]

    # Fit windower and create windows
    windows = windower.prepare(cal_values, fit=True)
    tensors = torch.FloatTensor(windows).unsqueeze(-1)

    # Compute reconstruction errors on normal data
    model.eval()
    with torch.no_grad():
        errors = model.anomaly_score(tensors)

    # Calibrate scorer
    scorer.calibrate(errors)
    st.session_state.calibrated = True
