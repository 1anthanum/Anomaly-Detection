"""Tests for anomaly scoring."""

import numpy as np

from src.detection.scoring import AnomalyScorer, AnomalyResult


def test_calibrate():
    scorer = AnomalyScorer(threshold_percentile=95, window_size=100)
    errors = np.random.exponential(0.1, size=200)
    scorer.calibrate(errors)
    assert scorer._calibrated
    assert scorer.baseline_mean > 0


def test_score_returns_result():
    scorer = AnomalyScorer()
    errors = np.random.exponential(0.1, size=200)
    scorer.calibrate(errors)

    result = scorer.score(0.05)
    assert isinstance(result, AnomalyResult)
    assert 0 <= result.score <= 1
    assert result.severity in ("normal", "warning", "critical")


def test_high_error_detected():
    scorer = AnomalyScorer(threshold_percentile=90)
    normal_errors = np.random.exponential(0.01, size=300)
    scorer.calibrate(normal_errors)

    # Very high error should be anomalous
    result = scorer.score(10.0)
    assert result.is_anomaly
    assert result.severity in ("warning", "critical")


def test_low_error_normal():
    scorer = AnomalyScorer(threshold_percentile=95)
    normal_errors = np.random.exponential(1.0, size=300)
    scorer.calibrate(normal_errors)

    # Very low error should be normal
    result = scorer.score(0.001)
    assert not result.is_anomaly
    assert result.severity == "normal"


def test_score_batch():
    scorer = AnomalyScorer()
    errors = np.random.exponential(0.1, size=200)
    scorer.calibrate(errors)

    batch_errors = np.random.exponential(0.1, size=10)
    results = scorer.score_batch(batch_errors)
    assert len(results) == 10
    assert all(isinstance(r, AnomalyResult) for r in results)


def test_uncalibrated_raises():
    scorer = AnomalyScorer()
    try:
        scorer.score(0.1)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    test_calibrate()
    test_score_returns_result()
    test_high_error_detected()
    test_low_error_normal()
    test_score_batch()
    test_uncalibrated_raises()
    print("All scoring tests passed!")
