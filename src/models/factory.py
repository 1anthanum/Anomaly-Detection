"""
Model and simulator factory functions.

Single source of truth for constructing components from config dicts,
used by both the training pipeline and the Streamlit app.
"""

from src.data import TimeSeriesSimulator, SimulatorConfig, AnomalyConfig
from .lstm_autoencoder import LSTMAutoencoder
from .transformer_detector import TransformerDetector
from .base import AnomalyDetector


def build_model_from_config(cfg: dict) -> AnomalyDetector:
    """Construct an anomaly detection model from the full config dict."""
    model_cfg = cfg["model"]
    window_size = model_cfg["window_size"]

    if model_cfg["default"] == "lstm":
        p = model_cfg["lstm"]
        return LSTMAutoencoder(
            window_size=window_size,
            hidden_dim=p["hidden_dim"],
            latent_dim=p["latent_dim"],
            n_layers=p["n_layers"],
            dropout=p["dropout"],
        )
    else:
        p = model_cfg["transformer"]
        return TransformerDetector(
            window_size=window_size,
            d_model=p["d_model"],
            n_heads=p["n_heads"],
            n_encoder_layers=p["n_encoder_layers"],
            n_decoder_layers=p["n_decoder_layers"],
            dim_feedforward=p["dim_feedforward"],
            dropout=p["dropout"],
        )


def build_simulator_from_config(cfg: dict, with_anomalies: bool = True) -> TimeSeriesSimulator:
    """Construct a TimeSeriesSimulator from the full config dict.

    Args:
        cfg: Full application config dict.
        with_anomalies: If False, creates a clean simulator (for calibration/training).
    """
    sim_cfg = cfg["simulator"]

    if with_anomalies:
        anom = sim_cfg["anomaly"]
        anomaly_config = AnomalyConfig(
            point_prob=anom["point_prob"],
            contextual_prob=anom["contextual_prob"],
            collective_prob=anom["collective_prob"],
        )
    else:
        anomaly_config = AnomalyConfig(point_prob=0, contextual_prob=0, collective_prob=0)

    return TimeSeriesSimulator(
        SimulatorConfig(
            base_value=sim_cfg["base_value"],
            daily_amplitude=sim_cfg["daily_amplitude"],
            weekly_amplitude=sim_cfg["weekly_amplitude"],
            noise_std=sim_cfg["noise_std"],
            sampling_rate=sim_cfg["sampling_rate"],
            anomaly=anomaly_config,
        )
    )
