"""
LSTM Autoencoder for time series anomaly detection.
Learns to reconstruct normal patterns; high reconstruction error = anomaly.
"""

import torch
import torch.nn as nn
from .base import AnomalyDetector


class LSTMAutoencoder(AnomalyDetector):
    """
    Encoder-Decoder LSTM Autoencoder.

    Architecture:
        Encoder: LSTM that compresses input sequence into a fixed-size latent vector
        Decoder: LSTM that reconstructs the sequence from the latent representation

    Anomaly detection logic:
        Normal data → low reconstruction error
        Anomalous data → high reconstruction error (model hasn't learned these patterns)
    """

    def __init__(self, window_size: int = 50, n_features: int = 1,
                 hidden_dim: int = 64, latent_dim: int = 32, n_layers: int = 2, dropout: float = 0.1):
        super().__init__(window_size, n_features)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0
        )
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0
        )
        self.output_fc = nn.Linear(hidden_dim, n_features)

    def _get_config(self) -> dict:
        return {
            "window_size": self.window_size,
            "n_features": self.n_features,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "n_layers": self.n_layers,
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input sequence to latent representation. x: (batch, seq_len, features)"""
        _, (h_n, _) = self.encoder_lstm(x)
        latent = self.encoder_fc(h_n[-1])  # Use last layer's hidden state
        return latent

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Decode latent vector back to sequence. z: (batch, latent_dim)"""
        h = self.decoder_fc(z)  # (batch, hidden_dim)
        h_repeated = h.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden_dim)
        decoded, _ = self.decoder_lstm(h_repeated)
        output = self.output_fc(decoded)  # (batch, seq_len, features)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode then decode."""
        latent = self.encode(x)
        reconstructed = self.decode(latent, x.size(1))
        return reconstructed
