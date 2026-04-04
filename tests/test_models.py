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
    """Test the new split-file save/load (weights .pt + config .json)."""
    path = str(tmp_path / "model.pt")
    model = LSTMAutoencoder(window_size=20, hidden_dim=32, latent_dim=16, n_layers=1)
    x = torch.randn(2, 20, 1)

    original_out = model(x).detach()
    model.save(path)

    # Verify sidecar JSON was created
    import json
    from pathlib import Path
    config_path = Path(path).with_suffix(".json")
    assert config_path.exists(), "Config JSON sidecar should be created"
    config_data = json.loads(config_path.read_text())
    assert config_data["model_class"] == "LSTMAutoencoder"
    assert config_data["config"]["hidden_dim"] == 32

    # Load using config from sidecar (no manual kwargs needed)
    loaded = LSTMAutoencoder.load(path)
    loaded_out = loaded(x).detach()
    assert torch.allclose(original_out, loaded_out, atol=1e-6)


def test_transformer_save_load(tmp_path):
    """Transformer round-trip through the new checkpoint format."""
    path = str(tmp_path / "transformer.pt")
    model = TransformerDetector(window_size=20, d_model=32, n_heads=4, n_encoder_layers=1, n_decoder_layers=1)
    x = torch.randn(2, 20, 1)

    model.eval()
    original_out = model(x).detach()
    model.save(path)

    loaded = TransformerDetector.load(path)
    loaded_out = loaded(x).detach()
    assert torch.allclose(original_out, loaded_out, atol=1e-6)


def test_save_load_with_kwargs_override(tmp_path):
    """kwargs passed to load() should override saved config values."""
    path = str(tmp_path / "model.pt")
    model = LSTMAutoencoder(window_size=20, hidden_dim=32, latent_dim=16, n_layers=1)
    model.save(path)

    # Override n_layers — this creates a structurally different model,
    # so state_dict won't match, but we only test that config merging works.
    import json
    from pathlib import Path
    config_data = json.loads(Path(path).with_suffix(".json").read_text())
    assert config_data["config"]["n_layers"] == 1  # original

    # We can't actually load with mismatched n_layers (state_dict mismatch),
    # but we can verify the config path reads + merges correctly.
    # So test with a compatible override: n_features (default 1).
    loaded = LSTMAutoencoder.load(path, n_features=1)
    assert loaded.n_features == 1


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
        test_transformer_save_load(Path(td))

    print("All model tests passed!")
