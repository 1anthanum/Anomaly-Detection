"""
End-to-end training pipeline: data generation -> preprocessing -> training -> calibration.
Orchestrates all components to produce a ready-to-deploy model.
"""

import yaml
import numpy as np
from pathlib import Path

from src.data import TimeSeriesSimulator, SimulatorConfig, AnomalyConfig, TimeSeriesWindower, create_dataloader
from src.models import LSTMAutoencoder, TransformerDetector
from src.detection import AnomalyScorer
from .trainer import Trainer


class TrainingPipeline:
    """Orchestrates the full training workflow."""

    def __init__(self, config_path: str = "configs/default.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

    def generate_training_data(self, n_points: int = 10000) -> dict:
        """Generate normal training data (no anomalies)."""
        sim_cfg = self.cfg["simulator"]
        sim = TimeSeriesSimulator(
            SimulatorConfig(
                base_value=sim_cfg["base_value"],
                daily_amplitude=sim_cfg["daily_amplitude"],
                weekly_amplitude=sim_cfg["weekly_amplitude"],
                noise_std=sim_cfg["noise_std"],
                sampling_rate=sim_cfg["sampling_rate"],
                anomaly=AnomalyConfig(point_prob=0, contextual_prob=0, collective_prob=0),
            )
        )
        return sim.generate_batch(n_points)

    def generate_test_data(self, n_points: int = 2000) -> dict:
        """Generate test data with anomalies for evaluation."""
        sim_cfg = self.cfg["simulator"]
        anom = sim_cfg["anomaly"]
        sim = TimeSeriesSimulator(
            SimulatorConfig(
                base_value=sim_cfg["base_value"],
                daily_amplitude=sim_cfg["daily_amplitude"],
                weekly_amplitude=sim_cfg["weekly_amplitude"],
                noise_std=sim_cfg["noise_std"],
                sampling_rate=sim_cfg["sampling_rate"],
                anomaly=AnomalyConfig(
                    point_prob=anom["point_prob"],
                    contextual_prob=anom["contextual_prob"],
                    collective_prob=anom["collective_prob"],
                ),
            )
        )
        return sim.generate_batch(n_points)

    def build_model(self):
        """Build model from config."""
        model_cfg = self.cfg["model"]
        window_size = model_cfg["window_size"]
        if model_cfg["default"] == "lstm":
            p = model_cfg["lstm"]
            return LSTMAutoencoder(
                window_size=window_size,
                hidden_dim=p["hidden_dim"],
                latent_dim=p["latent_dim"],
                n_layers=p["n_layers"],
                dropout=p["dropout"],
            )
        else:
            p = model_cfg["transformer"]
            return TransformerDetector(
                window_size=window_size,
                d_model=p["d_model"],
                n_heads=p["n_heads"],
                n_encoder_layers=p["n_encoder_layers"],
                n_decoder_layers=p["n_decoder_layers"],
                dim_feedforward=p["dim_feedforward"],
                dropout=p["dropout"],
            )

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

        if verbose:
            print("Step 1/5: Generating training data...")
        train_data = self.generate_training_data(n_train)

        if verbose:
            print("Step 2/5: Preprocessing...")
        windower = TimeSeriesWindower(window_size=window_size)
        train_windows = windower.prepare(train_data["values"], fit=True)

        # Split into train/val (80/20)
        split_idx = int(len(train_windows) * 0.8)
        train_loader = create_dataloader(train_windows[:split_idx], batch_size=batch_size, shuffle=True)
        val_loader = create_dataloader(train_windows[split_idx:], batch_size=batch_size, shuffle=False)

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

        if verbose:
            print("Step 4/5: Calibrating scorer...")
        cal_errors = trainer.compute_reconstruction_errors(val_loader)
        det_cfg = self.cfg["detection"]
        scorer = AnomalyScorer(
            threshold_percentile=det_cfg["threshold_percentile"],
            window_size=det_cfg["history_window"],
        )
        scorer.calibrate(cal_errors)

        if verbose:
            print("Step 5/5: Evaluating on test data...")
        test_data = self.generate_test_data(n_test)
        test_windows = windower.prepare(test_data["values"])
        test_loader = create_dataloader(test_windows, batch_size=batch_size, shuffle=False)
        test_errors = trainer.compute_reconstruction_errors(test_loader)
        results = scorer.score_batch(test_errors)

        detected = sum(1 for r in results if r.is_anomaly)
        if verbose:
            print(f"\nTraining complete!")
            print(f"  Final train loss: {history['train_loss'][-1]:.6f}")
            print(f"  Final val loss:   {history['val_loss'][-1]:.6f}")
            print(f"  Test anomalies detected: {detected}/{len(results)}")

        return {
            "history": history,
            "scorer": scorer,
            "windower": windower,
            "model": model,
            "test_results": results,
        }


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run(epochs=30, verbose=True)
