"""
Model trainer: handles training loop, validation, and checkpointing
for anomaly detection autoencoders.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

from src.models.base import AnomalyDetector


class Trainer:
    """Handles training loop for anomaly detection models."""

    def __init__(
        self,
        model: AnomalyDetector,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss

    def validate(self, dataloader: DataLoader) -> float:
        """Run validation. Returns average loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 50,
        save_path: str = None,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Full training loop with optional validation and checkpointing."""
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.history["val_loss"].append(val_loss)

                # Save best model
                if save_path and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.model.save(save_path)

            if verbose and epoch % 10 == 0:
                msg = f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f}"
                if val_loss is not None:
                    msg += f" | Val Loss: {val_loss:.6f}"
                print(msg)

        # Save final model if no validation
        if save_path and val_loader is None:
            self.model.save(save_path)

        return self.history

    def compute_reconstruction_errors(self, dataloader: DataLoader) -> np.ndarray:
        """Compute reconstruction errors for all samples in the dataloader."""
        self.model.eval()
        all_errors = []

        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(self.device)
                errors = self.model.anomaly_score(inputs)
                all_errors.append(errors)

        return np.concatenate(all_errors)
