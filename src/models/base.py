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

    def anomaly_score(self, x: torch.Tensor) -> np.ndarray:
        """Compute anomaly score as reconstruction error."""
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            mse = ((x - reconstructed) ** 2).mean(dim=(1, 2))
        return mse.cpu().numpy()

    def save(self, path: str):
        torch.save({"state_dict": self.state_dict(), "config": {"window_size": self.window_size, "n_features": self.n_features}}, path)

    @classmethod
    def load(cls, path: str, **kwargs):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        model = cls(window_size=config["window_size"], n_features=config["n_features"], **kwargs)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model
