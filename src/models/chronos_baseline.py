"""
Chronos foundation model baseline for anomaly detection.

Uses Amazon Chronos (a pre-trained time-series forecasting model) for
zero-shot forecast-based anomaly detection:
  1. Given a context window of historical values, forecast the next step(s).
  2. Compare actual values against the forecast confidence interval.
  3. Flag values outside the interval as anomalies.

This module is optional — it requires ``chronos-forecasting`` and ``torch``
to be installed.  All public functions degrade gracefully when the
dependency is missing.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    from chronos import ChronosPipeline

    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False


@dataclass
class ChronosAnomalyResult:
    """Per-window anomaly result produced by the Chronos baseline."""

    is_anomaly: bool
    actual: float
    forecast_median: float
    forecast_low: float
    forecast_high: float
    deviation: float  # how far actual is from the nearest boundary (0 if inside)


class ChronosBaseline:
    """Zero-shot anomaly detector backed by a Chronos forecasting model.

    Parameters
    ----------
    model_name : str
        HuggingFace model id, e.g. ``"amazon/chronos-t5-tiny"``.
    context_length : int
        Number of historical points fed as context for each forecast.
    prediction_length : int
        How many steps ahead to forecast (usually 1 for point-anomaly detection).
    quantile_low : float
        Lower quantile for the confidence interval (default 0.05 → 5th percentile).
    quantile_high : float
        Upper quantile for the confidence interval (default 0.95 → 95th percentile).
    device : str
        ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-tiny",
        context_length: int = 64,
        prediction_length: int = 1,
        quantile_low: float = 0.05,
        quantile_high: float = 0.95,
        device: str = "cpu",
    ):
        if not CHRONOS_AVAILABLE:
            raise ImportError(
                "chronos-forecasting is required for ChronosBaseline. "
                "Install it with: pip install chronos-forecasting"
            )

        self.model_name = model_name
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.quantile_low = quantile_low
        self.quantile_high = quantile_high
        self.device = device

        logger.info("Loading Chronos model '%s' on %s ...", model_name, device)
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float32,
        )
        logger.info("Chronos model loaded.")

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def forecast(self, context: np.ndarray) -> dict:
        """Produce a probabilistic forecast for the next ``prediction_length`` steps.

        Parameters
        ----------
        context : np.ndarray
            1-D array of recent values (length >= ``context_length``).

        Returns
        -------
        dict with keys ``"median"``, ``"low"``, ``"high"`` — each a 1-D
        numpy array of length ``prediction_length``.
        """
        context_tensor = torch.tensor(context, dtype=torch.float32)
        # ChronosPipeline.predict returns (n_samples, prediction_length) samples
        samples = self.pipeline.predict(
            context_tensor,
            prediction_length=self.prediction_length,
            num_samples=100,
        )  # shape: (100, prediction_length)

        samples_np = samples.numpy()
        median = np.median(samples_np, axis=0)
        low = np.quantile(samples_np, self.quantile_low, axis=0)
        high = np.quantile(samples_np, self.quantile_high, axis=0)

        return {"median": median, "low": low, "high": high}

    def detect_single(self, context: np.ndarray, actual: float) -> ChronosAnomalyResult:
        """Detect whether a single new value is anomalous given its context.

        Parameters
        ----------
        context : np.ndarray
            1-D array of historical values (the window before ``actual``).
        actual : float
            The observed value immediately following the context.

        Returns
        -------
        ChronosAnomalyResult
        """
        fc = self.forecast(context)
        low = float(fc["low"][0])
        high = float(fc["high"][0])
        median = float(fc["median"][0])

        if actual < low:
            deviation = low - actual
        elif actual > high:
            deviation = actual - high
        else:
            deviation = 0.0

        return ChronosAnomalyResult(
            is_anomaly=(actual < low or actual > high),
            actual=actual,
            forecast_median=median,
            forecast_low=low,
            forecast_high=high,
            deviation=deviation,
        )

    # ------------------------------------------------------------------
    # Batch evaluation (compatible with pipeline.compute_metrics)
    # ------------------------------------------------------------------

    def detect_batch(
        self,
        values: np.ndarray,
        verbose: bool = False,
    ) -> list[ChronosAnomalyResult]:
        """Run sliding-window anomaly detection over an entire time series.

        For each position ``i`` (where ``i >= context_length``), the model
        receives ``values[i - context_length : i]`` as context and checks
        whether ``values[i]`` falls inside the forecast confidence interval.

        Parameters
        ----------
        values : np.ndarray
            1-D array of the full time series.
        verbose : bool
            Print progress every 500 steps.

        Returns
        -------
        list[ChronosAnomalyResult] of length ``len(values) - context_length``.
        """
        n = len(values)
        results: list[ChronosAnomalyResult] = []

        for i in range(self.context_length, n):
            context = values[i - self.context_length : i]
            actual = float(values[i])
            result = self.detect_single(context, actual)
            results.append(result)

            if verbose and (i - self.context_length) % 500 == 0:
                done = i - self.context_length
                total = n - self.context_length
                print(f"  Chronos progress: {done}/{total}")

        return results

    def get_predictions(self, values: np.ndarray, verbose: bool = False) -> list[bool]:
        """Convenience wrapper: returns a flat list of boolean predictions.

        Same length as ``detect_batch`` output.  Useful for feeding directly
        into ``compute_metrics``.
        """
        results = self.detect_batch(values, verbose=verbose)
        return [r.is_anomaly for r in results]
