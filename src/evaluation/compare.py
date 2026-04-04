"""
Model comparison framework.

Runs multiple anomaly detectors on the same test data and produces a
structured comparison of precision / recall / F1 / latency.
"""

import logging
import time
from dataclasses import dataclass, field

from src.data import TimeSeriesWindower, create_dataloader
from src.detection import AnomalyScorer
from src.models import build_model_from_config, build_simulator_from_config
from src.training.pipeline import compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Aggregated result for one detector on one test set."""

    name: str
    precision: float
    recall: float
    f1: float
    n_detected: int
    n_total: int
    elapsed_seconds: float
    extra: dict = field(default_factory=dict)

    def summary_line(self) -> str:
        return (
            f"{self.name:<25s}  P={self.precision:.4f}  R={self.recall:.4f}  "
            f"F1={self.f1:.4f}  detected={self.n_detected}/{self.n_total}  "
            f"time={self.elapsed_seconds:.1f}s"
        )


class ModelComparator:
    """Orchestrates a side-by-side comparison of detectors.

    Workflow
    --------
    1. ``generate_data()`` — creates shared train / test splits.
    2. ``evaluate_autoencoder()`` — train + evaluate an LSTM/Transformer model.
    3. ``evaluate_chronos()`` — zero-shot evaluate Chronos baseline.
    4. ``report()`` — print / return all results.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.results: list[ComparisonResult] = []
        self._test_data: dict | None = None
        self._train_data: dict | None = None

    # ------------------------------------------------------------------
    # Data generation (shared across all detectors)
    # ------------------------------------------------------------------

    def generate_data(
        self,
        n_train: int = 10000,
        n_test: int = 2000,
        verbose: bool = True,
    ):
        """Generate shared train and test datasets."""
        if verbose:
            print("Generating shared training data ...")
        sim_train = build_simulator_from_config(self.cfg, with_anomalies=False)
        self._train_data = sim_train.generate_batch(n_train)

        if verbose:
            print("Generating shared test data ...")
        sim_test = build_simulator_from_config(self.cfg, with_anomalies=True)
        self._test_data = sim_test.generate_batch(n_test)

    # ------------------------------------------------------------------
    # Autoencoder evaluation (LSTM / Transformer)
    # ------------------------------------------------------------------

    def evaluate_autoencoder(
        self,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> ComparisonResult:
        """Train and evaluate the project's autoencoder detector."""
        from src.training.trainer import Trainer

        if self._train_data is None or self._test_data is None:
            raise RuntimeError("Call generate_data() first.")

        model_type = self.cfg["model"]["default"]
        name = f"Autoencoder ({model_type.upper()})"
        window_size = self.cfg["model"]["window_size"]

        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating: {name}")
            print(f"{'='*60}")

        t0 = time.time()

        # Preprocess
        windower = TimeSeriesWindower(window_size=window_size)
        train_windows = windower.prepare(self._train_data["values"], fit=True)
        split_idx = int(len(train_windows) * 0.8)
        train_loader = create_dataloader(train_windows[:split_idx], batch_size=batch_size, shuffle=True)
        val_loader = create_dataloader(train_windows[split_idx:], batch_size=batch_size, shuffle=False)

        # Train
        model = build_model_from_config(self.cfg)
        trainer = Trainer(model)
        trainer.fit(train_loader, val_loader, epochs=epochs, verbose=verbose)

        # Calibrate scorer
        cal_errors = trainer.compute_reconstruction_errors(val_loader)
        det_cfg = self.cfg["detection"]
        scorer = AnomalyScorer(
            threshold_percentile=det_cfg["threshold_percentile"],
            window_size=det_cfg["history_window"],
        )
        scorer.calibrate(cal_errors)

        # Evaluate on test data
        test_windows = windower.prepare(self._test_data["values"])
        test_loader = create_dataloader(test_windows, batch_size=batch_size, shuffle=False)
        test_errors = trainer.compute_reconstruction_errors(test_loader)
        scored = scorer.score_batch(test_errors)
        predictions = [r.is_anomaly for r in scored]

        elapsed = time.time() - t0

        metrics = compute_metrics(predictions, self._test_data["is_anomaly"], window_size)

        result = ComparisonResult(
            name=name,
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            n_detected=sum(predictions),
            n_total=len(predictions),
            elapsed_seconds=elapsed,
        )
        self.results.append(result)

        if verbose:
            print(f"  -> {result.summary_line()}")

        return result

    # ------------------------------------------------------------------
    # Chronos baseline evaluation
    # ------------------------------------------------------------------

    def evaluate_chronos(
        self,
        model_name: str = "amazon/chronos-t5-tiny",
        context_length: int = 64,
        quantile_low: float = 0.05,
        quantile_high: float = 0.95,
        verbose: bool = True,
    ) -> ComparisonResult:
        """Zero-shot evaluate the Chronos forecasting baseline.

        Requires ``chronos-forecasting`` to be installed.
        """
        from src.models.chronos_baseline import ChronosBaseline, CHRONOS_AVAILABLE

        if not CHRONOS_AVAILABLE:
            raise ImportError(
                "chronos-forecasting is not installed. "
                "Install with: pip install chronos-forecasting"
            )
        if self._test_data is None:
            raise RuntimeError("Call generate_data() first.")

        name = f"Chronos ({model_name.split('/')[-1]})"

        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating: {name}")
            print(f"{'='*60}")

        t0 = time.time()

        detector = ChronosBaseline(
            model_name=model_name,
            context_length=context_length,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
        )

        test_values = self._test_data["values"]
        predictions = detector.get_predictions(test_values, verbose=verbose)

        elapsed = time.time() - t0

        # Align labels: Chronos predictions start at index ``context_length``
        # of the original series, so we slice labels accordingly.
        aligned_labels = self._test_data["is_anomaly"][context_length:]
        n_preds = len(predictions)
        aligned_labels = aligned_labels[:n_preds]

        metrics = compute_metrics(predictions, aligned_labels, window_size=1)

        result = ComparisonResult(
            name=name,
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            n_detected=sum(predictions),
            n_total=len(predictions),
            elapsed_seconds=elapsed,
        )
        self.results.append(result)

        if verbose:
            print(f"  -> {result.summary_line()}")

        return result

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self, verbose: bool = True) -> list[ComparisonResult]:
        """Print a summary table and return all results."""
        if verbose:
            print(f"\n{'='*60}")
            print("COMPARISON SUMMARY")
            print(f"{'='*60}")
            for r in self.results:
                print(f"  {r.summary_line()}")
            print(f"{'='*60}\n")

        return self.results
