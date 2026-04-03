"""
Transformer-based Autoencoder for time series anomaly detection.
Uses self-attention to capture long-range temporal dependencies.
"""

import torch
import torch.nn as nn
import math
from .base import AnomalyDetector


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer input."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerDetector(AnomalyDetector):
    """
    Transformer Autoencoder for anomaly detection.

    Uses multi-head self-attention to model temporal dependencies,
    then reconstructs the input. Anomalies produce high reconstruction error.

    The decoder uses learned query tokens instead of the raw input, forcing
    reconstruction to rely solely on the encoder's compressed memory. This
    creates a proper information bottleneck: normal patterns are well-captured
    by the encoder, while anomalous inputs produce high reconstruction error.
    """

    def __init__(self, window_size: int = 50, n_features: int = 1,
                 d_model: int = 64, n_heads: int = 4, n_encoder_layers: int = 2,
                 n_decoder_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__(window_size, n_features)
        self.d_model = d_model

        self.input_projection = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=window_size)

        # Learned query tokens for the decoder (information bottleneck)
        self.decoder_query = nn.Parameter(torch.randn(1, window_size, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

        self.output_projection = nn.Linear(d_model, n_features)

    def _get_config(self) -> dict:
        return {
            "window_size": self.window_size,
            "n_features": self.n_features,
            "d_model": self.d_model,
            "n_heads": self.transformer_encoder.layers[0].self_attn.num_heads,
            "n_encoder_layers": len(self.transformer_encoder.layers),
            "n_decoder_layers": len(self.transformer_decoder.layers),
            "dim_feedforward": self.transformer_encoder.layers[0].linear1.out_features,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, features) -> reconstructed: (batch, seq_len, features)"""
        batch_size = x.size(0)

        # Project to model dimension and encode
        src = self.input_projection(x) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)

        # Decode using learned queries (not the raw input)
        tgt = self.pos_encoder(self.decoder_query.expand(batch_size, -1, -1))
        decoded = self.transformer_decoder(tgt, memory)

        # Project back to feature dimension
        output = self.output_projection(decoded)
        return output
