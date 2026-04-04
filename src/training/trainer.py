"""
Model trainer: handles training loop, validation, checkpointing,
early stopping, learning rate scheduling, and experiment logging
for anomaly detection autoencoders.
"""

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from src.models.base import AnomalyDetector
from src.training.logger import TrainingLogger, NullLogger

logger = logging.getLogger(__name__)


class Trainer:
    """Handles training loop for anomaly detection models."""

    def __init__(
        self,
        model: AnomalyDetector,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        max_grad_norm: float = 1.0,
        device: str = "cpu",
        experiment_logger: TrainingLogger | None = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )
        self.criterion = nn.MSELoss()
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self.exp_logger: TrainingLogger = experiment_logger or NullLogger()

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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
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
        patience: int = 10,
        save_path: str = None,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Full training loop with optional validation, early stopping, and checkpointing.

        Args:
            patience: Stop training if validation loss does not improve for this
                      many consecutive epochs. Set to 0 to disable early stopping.
        """
        # Log hyperparameters at run start
        self.exp_logger.log_params({
            "model_class": type(self.model).__name__,
            "epochs": epochs,
            "patience": patience,
            "lr": self.optimizer.param_groups[0]["lr"],
            "max_grad_norm": self.max_grad_norm,
            "window_size": getattr(self.model, "window_size", "N/A"),
        })

        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)
            self.exp_logger.log_scalar("loss/train", train_loss, epoch)

            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.history["val_loss"].append(val_loss)
                self.exp_logger.log_scalar("loss/val", val_loss, epoch)

                # Step the LR scheduler
                self.scheduler.step(val_loss)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    if save_path:
                        self.model.save(save_path)
                else:
                    epochs_without_improvement += 1

                # Early stopping
                if patience > 0 and epochs_without_improvement >= patience:
                    logger.info(
                        "Early stopping at epoch %d (no improvement for %d epochs)",
                        epoch, patience,
                    )
                    if verbose:
                        print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                    break

            # Log learning rate
            lr = self.optimizer.param_groups[0]["lr"]
            self.exp_logger.log_scalar("lr", lr, epoch)

            if verbose and epoch % 10 == 0:
                msg = f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f}"
                if val_loss is not None:
                    msg += f" | Val Loss: {val_loss:.6f}"
                msg += f" | LR: {lr:.2e}"
                print(msg)
                logger.info(msg)

        # Save final model if no validation
        if save_path and val_loader is None:
            self.model.save(save_path)

        self.exp_logger.close()
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
