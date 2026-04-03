"""
Anomaly scoring: converts model reconstruction errors into interpretable
anomaly scores and applies threshold-based detection.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class AnomalyResult:
    """Result of anomaly detection for a single window."""
    score: float               # Normalized anomaly score [0, 1]
    is_anomaly: bool           # Whether score exceeds threshold
    raw_error: float           # Raw reconstruction error
    threshold: float           # Current raw threshold value
    threshold_score: float     # Normalized threshold [0, 1] (for display)
    severity: str              # "normal", "warning", "critical"


class AnomalyScorer:
    """
    Converts raw reconstruction errors into normalized anomaly scores
    and applies adaptive thresholding.
    """

    def __init__(self, threshold_percentile: float = 95, window_size: int = 200):
        self.threshold_percentile = threshold_percentile
        self.window_size = window_size
        self.error_history: list[float] = []
        self.baseline_mean: float = 0.0
        self.baseline_std: float = 1.0
        self._calibrated = False

    def calibrate(self, normal_errors: np.ndarray):
        """Calibrate scorer using reconstruction errors from normal data."""
        self.baseline_mean = normal_errors.mean()
        self.baseline_std = normal_errors.std() + 1e-8
        self.error_history = list(normal_errors[-self.window_size:])
        self._calibrated = True

    def _normalize_score(self, raw_error: float) -> float:
        """Normalize raw error to [0, 1] range using calibration stats."""
        z_score = (raw_error - self.baseline_mean) / self.baseline_std
        score = 1 / (1 + np.exp(-z_score + 2))  # Sigmoid centered at 2 std
        return float(np.clip(score, 0, 1))

    def _get_adaptive_threshold(self) -> float:
        """Compute adaptive threshold from recent error history."""
        if len(self.error_history) < 10:
            return self.baseline_mean + 2 * self.baseline_std
        return float(np.percentile(self.error_history, self.threshold_percentile))

    def _classify_severity(self, score: float, threshold_score: float) -> str:
        if score < threshold_score:
            return "normal"
        elif score < threshold_score + 0.15:
            return "warning"
        else:
            return "critical"

    def score(self, raw_error: float) -> AnomalyResult:
        """Score a single reconstruction error."""
        if not self._calibrated:
            raise ValueError("Call calibrate() first with normal data errors")

        normalized = self._normalize_score(raw_error)
        threshold = self._get_adaptive_threshold()
        threshold_score = self._normalize_score(threshold)

        self.error_history.append(raw_error)
        if len(self.error_history) > self.window_size:
            self.error_history.pop(0)

        return AnomalyResult(
            score=normalized,
            is_anomaly=raw_error > threshold,
            raw_error=raw_error,
            threshold=threshold,
            threshold_score=threshold_score,
            severity=self._classify_severity(normalized, threshold_score),
        )

    def score_batch(self, raw_errors: np.ndarray) -> list[AnomalyResult]:
        return [self.score(e) for e in raw_errors]
