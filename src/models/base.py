"""Abstract base class for anomaly detection models."""

from abc import ABC, abstractmethod
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

    def save(self, path: str):
        """Save model weights and full config to a checkpoint file."""
        torch.save({
            "state_dict": self.state_dict(),
            "config": self._get_config(),
            "model_class": type(self).__name__,
        }, path)

    @classmethod
    def load(cls, path: str, **kwargs):
        """Load model from checkpoint. kwargs override saved config."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        config.update(kwargs)  # allow overrides
        model = cls(**config)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model
