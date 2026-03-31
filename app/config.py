"""
Configuration loading and component factory functions.

Responsible for reading YAML config and constructing
simulator / model instances from configuration dicts.
"""

import yaml
import streamlit as st

from src.data import TimeSeriesSimulator, SimulatorConfig, AnomalyConfig
from src.models import LSTMAutoencoder, TransformerDetector


@st.cache_data
def load_config(path: str = "configs/default.yaml") -> dict:
    """Load and cache the YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def build_simulator(cfg: dict) -> TimeSeriesSimulator:
    """Construct a TimeSeriesSimulator from the config dict."""
    sim_cfg = cfg["simulator"]
    anom = sim_cfg["anomaly"]
    return TimeSeriesSimulator(
        SimulatorConfig(
            base_value=sim_cfg["base_value"],
            daily_amplitude=sim_cfg["daily_amplitude"],
            weekly_amplitude=sim_cfg["weekly_amplitude"],
            noise_std=sim_cfg["noise_std"],
            sampling_rate=sim_cfg["sampling_rate"],
            anomaly=AnomalyConfig(
                point_prob=anom["point_prob"],
                contextual_prob=anom["contextual_prob"],
                collective_prob=anom["collective_prob"],
            ),
        )
    )


def build_model(cfg: dict):
    """Construct an anomaly detection model from the config dict."""
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
