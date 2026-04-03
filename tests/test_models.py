"""Tests for anomaly detection models."""

import torch

from src.models.lstm_autoencoder import LSTMAutoencoder
from src.models.transformer_detector import TransformerDetector


def test_lstm_forward_shape():
    model = LSTMAutoencoder(window_size=20, hidden_dim=32, latent_dim=16, n_layers=1)
    x = torch.randn(4, 20, 1)  # batch=4, seq=20, features=1
    out = model(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_lstm_encode_decode():
    model = LSTMAutoencoder(window_size=20, hidden_dim=32, latent_dim=16, n_layers=1)
    x = torch.randn(4, 20, 1)
    latent = model.encode(x)
    assert latent.shape == (4, 16)
    decoded = model.decode(latent, seq_len=20)
    assert decoded.shape == (4, 20, 1)


def test_lstm_anomaly_score():
    model = LSTMAutoencoder(window_size=20, hidden_dim=32, latent_dim=16, n_layers=1)
    x = torch.randn(8, 20, 1)
    scores = model.anomaly_score(x)
    assert scores.shape == (8,)
    assert all(s >= 0 for s in scores)


def test_transformer_forward_shape():
    model = TransformerDetector(window_size=20, d_model=32, n_heads=4, n_encoder_layers=1, n_decoder_layers=1)
    x = torch.randn(4, 20, 1)
    out = model(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_transformer_anomaly_score():
    model = TransformerDetector(window_size=20, d_model=32, n_heads=4, n_encoder_layers=1, n_decoder_layers=1)
    x = torch.randn(8, 20, 1)
    scores = model.anomaly_score(x)
    assert scores.shape == (8,)
    assert all(s >= 0 for s in scores)


def test_lstm_save_load(tmp_path):
    path = str(tmp_path / "model.pt")
    model = LSTMAutoencoder(window_size=20, hidden_dim=32, latent_dim=16, n_layers=1)
    x = torch.randn(2, 20, 1)

    original_out = model(x).detach()
    model.save(path)

    loaded = LSTMAutoencoder.load(path, hidden_dim=32, latent_dim=16, n_layers=1)
    loaded_out = loaded(x).detach()

    assert torch.allclose(original_out, loaded_out, atol=1e-6)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    test_lstm_forward_shape()
    test_lstm_encode_decode()
    test_lstm_anomaly_score()
    test_transformer_forward_shape()
    test_transformer_anomaly_score()

    with tempfile.TemporaryDirectory() as td:
        test_lstm_save_load(Path(td))

    print("All model tests passed!")
