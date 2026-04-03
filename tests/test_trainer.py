"""Tests for the model trainer."""

import numpy as np
import torch

from src.models.lstm_autoencoder import LSTMAutoencoder
from src.data.preprocessor import create_dataloader
from src.training.trainer import Trainer


def _make_loader(n_windows=50, window_size=20, batch_size=16, shuffle=False):
    windows = np.random.randn(n_windows, window_size).astype(np.float32)
    return create_dataloader(windows, batch_size=batch_size, shuffle=shuffle)


def test_train_epoch():
    model = LSTMAutoencoder(window_size=20, hidden_dim=16, latent_dim=8, n_layers=1)
    trainer = Trainer(model)
    loader = _make_loader()
    loss = trainer.train_epoch(loader)
    assert isinstance(loss, float)
    assert loss > 0


def test_validate():
    model = LSTMAutoencoder(window_size=20, hidden_dim=16, latent_dim=8, n_layers=1)
    trainer = Trainer(model)
    loader = _make_loader()
    loss = trainer.validate(loader)
    assert isinstance(loss, float)
    assert loss > 0


def test_fit_records_history():
    model = LSTMAutoencoder(window_size=20, hidden_dim=16, latent_dim=8, n_layers=1)
    trainer = Trainer(model)
    train_loader = _make_loader()
    val_loader = _make_loader(n_windows=20)
    history = trainer.fit(train_loader, val_loader, epochs=3, verbose=False, patience=0)
    assert len(history["train_loss"]) == 3
    assert len(history["val_loss"]) == 3


def test_early_stopping():
    model = LSTMAutoencoder(window_size=20, hidden_dim=16, latent_dim=8, n_layers=1)
    trainer = Trainer(model)
    train_loader = _make_loader()
    val_loader = _make_loader(n_windows=20)
    # Very aggressive patience: should stop before 100 epochs
    history = trainer.fit(train_loader, val_loader, epochs=100, patience=3, verbose=False)
    assert len(history["train_loss"]) < 100


def test_lr_scheduler_exists():
    model = LSTMAutoencoder(window_size=20, hidden_dim=16, latent_dim=8, n_layers=1)
    trainer = Trainer(model)
    assert hasattr(trainer, "scheduler")
    assert trainer.scheduler is not None


def test_compute_reconstruction_errors():
    model = LSTMAutoencoder(window_size=20, hidden_dim=16, latent_dim=8, n_layers=1)
    trainer = Trainer(model)
    loader = _make_loader(n_windows=30)
    errors = trainer.compute_reconstruction_errors(loader)
    assert errors.shape == (30,)
    assert all(e >= 0 for e in errors)


def test_save_best_model(tmp_path):
    path = str(tmp_path / "best.pt")
    model = LSTMAutoencoder(window_size=20, hidden_dim=16, latent_dim=8, n_layers=1)
    trainer = Trainer(model)
    train_loader = _make_loader()
    val_loader = _make_loader(n_windows=20)
    trainer.fit(train_loader, val_loader, epochs=3, save_path=path, verbose=False, patience=0)
    assert (tmp_path / "best.pt").exists()
