"""
End-to-end training pipeline: data generation -> preprocessing -> training -> calibration.
Orchestrates all components to produce a ready-to-deploy model.
"""

import logging
import yaml
import numpy as np
from pathlib import Path

from src.data import TimeSeriesWindower, create_dataloader
from src.models import build_model_from_config, build_simulator_from_config
from src.detection import AnomalyScorer
from .trainer import Trainer

logger = logging.getLogger(__name__)


def compute_metrics(predictions: list[bool], labels: np.ndarray, window_size: int) -> dict:
    """Compute precision, recall, and F1 from window-level predictions vs point-level labels.

    Each window covers points [i, i+window_size). A window is considered truly
    anomalous if any point in that window is labelled anomalous.
    """
    n_windows = len(predictions)
    window_labels = []
    for i in range(n_windows):
        window_labels.append(bool(labels[i : i + window_size].any()))

    tp = sum(p and t for p, t in zip(predictions, window_labels))
    fp = sum(p and not t for p, t in zip(predictions, window_labels))
    fn = sum(not p and t for p, t in zip(predictions, window_labels))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


class TrainingPipeline:
    """Orchestrates the full training workflow."""

    def __init__(self, config_path: str = "configs/default.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

    def generate_training_data(self, n_points: int = 10000) -> dict:
        """Generate normal training data (no anomalies)."""
        sim = build_simulator_from_config(self.cfg, with_anomalies=False)
        return sim.generate_batch(n_points)

    def generate_test_data(self, n_points: int = 2000) -> dict:
        """Generate test data with anomalies for evaluation."""
        sim = build_simulator_from_config(self.cfg, with_anomalies=True)
        return sim.generate_batch(n_points)

    def build_model(self):
        """Build model from config."""
        return build_model_from_config(self.cfg)

    def run(
        self,
        n_train: int = 10000,
        n_test: int = 2000,
        epochs: int = 50,
        batch_size: int = 32,
        save_dir: str = "weights",
        verbose: bool = True,
    ) -> dict:
        """Execute the full pipeline. Returns training results and scorer."""
        window_size = self.cfg["model"]["window_size"]
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        logger.info("Step 1/5: Generating training data...")
        if verbose:
            print("Step 1/5: Generating training data...")
        train_data = self.generate_training_data(n_train)

        logger.info("Step 2/5: Preprocessing...")
        if verbose:
            print("Step 2/5: Preprocessing...")
        windower = TimeSeriesWindower(window_size=window_size)
        train_windows = windower.prepare(train_data["values"], fit=True)

        # Split into train/val (80/20)
        split_idx = int(len(train_windows) * 0.8)
        train_loader = create_dataloader(train_windows[:split_idx], batch_size=batch_size, shuffle=True)
        val_loader = create_dataloader(train_windows[split_idx:], batch_size=batch_size, shuffle=False)

        logger.info("Step 3/5: Training model...")
        if verbose:
            print("Step 3/5: Training model...")
        model = self.build_model()
        trainer = Trainer(model)
        history = trainer.fit(
            train_loader, val_loader,
            epochs=epochs,
            save_path=str(save_path / "best_model.pt"),
            verbose=verbose,
        )

        logger.info("Step 4/5: Calibrating scorer...")
        if verbose:
            print("Step 4/5: Calibrating scorer...")
        cal_errors = trainer.compute_reconstruction_errors(val_loader)
        det_cfg = self.cfg["detection"]
        scorer = AnomalyScorer(
            threshold_percentile=det_cfg["threshold_percentile"],
            window_size=det_cfg["history_window"],
        )
        scorer.calibrate(cal_errors)

        logger.info("Step 5/5: Evaluating on test data...")
        if verbose:
            print("Step 5/5: Evaluating on test data...")
        test_data = self.generate_test_data(n_test)
        test_windows = windower.prepare(test_data["values"])
        test_loader = create_dataloader(test_windows, batch_size=batch_size, shuffle=False)
        test_errors = trainer.compute_reconstruction_errors(test_loader)
        results = scorer.score_batch(test_errors)

        # Evaluation metrics
        predictions = [r.is_anomaly for r in results]
        metrics = compute_metrics(predictions, test_data["is_anomaly"], window_size)

        detected = sum(predictions)
        if verbose:
            print(f"\nTraining complete!")
            print(f"  Final train loss: {history['train_loss'][-1]:.6f}")
            print(f"  Final val loss:   {history['val_loss'][-1]:.6f}")
            print(f"  Test anomalies detected: {detected}/{len(results)}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}")

        logger.info(
            "Training complete. Precision=%.4f Recall=%.4f F1=%.4f",
            metrics["precision"], metrics["recall"], metrics["f1"],
        )

        return {
            "history": history,
            "scorer": scorer,
            "windower": windower,
            "model": model,
            "test_results": results,
            "metrics": metrics,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = TrainingPipeline()
    pipeline.run(epochs=30, verbose=True)
