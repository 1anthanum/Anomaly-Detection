"""Abstract base class for anomaly detection models."""

import json
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np


class AnomalyDetector(ABC, nn.Module):
    """Base class for all anomaly detection models."""

    def __init__(self, window_size: int, n_features: int = 1):
        super().__init__()
        self.window_size = window_size
        self.n_features = n_features

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns reconstructed input."""
        pass

    def _get_config(self) -> dict:
        """Return full model config for checkpoint serialization.

        Subclasses should override this to include model-specific
        hyperparameters (hidden_dim, n_layers, etc.) so that checkpoints
        are fully self-describing.
        """
        return {"window_size": self.window_size, "n_features": self.n_features}

    def anomaly_score(self, x: torch.Tensor) -> np.ndarray:
        """Compute anomaly score as reconstruction error."""
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            mse = ((x - reconstructed) ** 2).mean(dim=(1, 2))
        return mse.cpu().numpy()

    @staticmethod
    def _config_path(weights_path: str) -> Path:
        """Derive the JSON config sidecar path from a .pt weights path."""
        p = Path(weights_path)
        return p.with_suffix(".json")

    def save(self, path: str):
        """Save model weights (.pt) and config (.json) as separate files.

        This avoids the ``weights_only=True`` incompatibility that occurs
        when non-tensor objects (dicts, strings) are stored inside a .pt
        checkpoint.
        """
        # Weights — tensors only, safe to load with weights_only=True
        torch.save(self.state_dict(), path)

        # Config sidecar — plain JSON
        config_data = {
            "model_class": type(self).__name__,
            "config": self._get_config(),
        }
        config_path = self._config_path(path)
        config_path.write_text(json.dumps(config_data, indent=2))

    @classmethod
    def load(cls, path: str, **kwargs):
        """Load model from a weights file + JSON config sidecar.

        Falls back to legacy single-file checkpoint format if the JSON
        sidecar is missing (for backward compatibility with older saves).
        kwargs override any values stored in the config.
        """
        config_path = AnomalyDetector._config_path(path)

        if config_path.exists():
            # New format: separate .pt (weights) + .json (config)
            config_data = json.loads(config_path.read_text())
            config = config_data["config"]
            config.update(kwargs)
            model = cls(**config)
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
        else:
            # Legacy format: single .pt with dict containing state_dict + config
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            config = checkpoint["config"]
            config.update(kwargs)
            model = cls(**config)
            model.load_state_dict(checkpoint["state_dict"])

        model.eval()
        return model
