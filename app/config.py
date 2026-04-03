"""
Configuration loading.

Delegates component construction to the shared factory in src.models.factory.
"""

import yaml
import streamlit as st

from src.models.factory import build_model_from_config, build_simulator_from_config

# Re-export factories under short names for app-layer convenience
build_model = build_model_from_config
build_simulator = build_simulator_from_config


@st.cache_data
def load_config(path: str = "configs/default.yaml") -> dict:
    """Load and cache the YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)
