"""
Session state initialization and model calibration.

Manages the lifecycle of all Streamlit session-state objects
(simulator, model, scorer, alert engine, dashboard state).
"""

import logging
from pathlib import Path

import torch
import streamlit as st

from src.data import TimeSeriesWindower
from src.detection import AnomalyScorer, AlertEngine
from src.models import build_simulator_from_config, LSTMAutoencoder, TransformerDetector
from src.visualization import DashboardState

from .config import build_simulator, build_model

logger = logging.getLogger(__name__)


def _load_or_build_model(cfg: dict):
    """Try to load a pre-trained model from weights/; fall back to building a fresh one."""
    weights_path = Path("weights/best_model.pt")
    model_cfg = cfg["model"]
    if weights_path.exists():
        try:
            model_type = model_cfg["default"]
            cls = LSTMAutoencoder if model_type == "lstm" else TransformerDetector
            if model_type == "lstm":
                p = model_cfg["lstm"]
                model = cls.load(str(weights_path), hidden_dim=p["hidden_dim"],
                                 latent_dim=p["latent_dim"], n_layers=p["n_layers"])
            else:
                p = model_cfg["transformer"]
                model = cls.load(str(weights_path), d_model=p["d_model"], n_heads=p["n_heads"],
                                 n_encoder_layers=p["n_encoder_layers"],
                                 n_decoder_layers=p["n_decoder_layers"],
                                 dim_feedforward=p["dim_feedforward"])
            logger.info("Loaded pre-trained model from %s", weights_path)
            return model
        except Exception as e:
            logger.warning("Failed to load model from %s: %s. Building fresh model.", weights_path, e)
    return build_model(cfg)


def init_session_state(cfg: dict):
    """Initialize all Streamlit session state objects (idempotent)."""
    if "initialized" in st.session_state:
        return

    model_cfg = cfg["model"]
    det_cfg = cfg["detection"]
    dash_cfg = cfg["dashboard"]

    # Core components
    st.session_state.simulator = build_simulator(cfg)
    st.session_state.model = _load_or_build_model(cfg)
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
    cal_sim = build_simulator_from_config(cfg, with_anomalies=False)
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
