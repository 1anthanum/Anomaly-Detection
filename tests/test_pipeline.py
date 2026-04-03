"""Tests for the training pipeline."""

import numpy as np

from src.training.pipeline import compute_metrics


def test_compute_metrics_perfect():
    """All anomalies detected, no false positives."""
    labels = np.array([False, False, True, True, False, False, False])
    # window_size=1 for simplicity: each window covers exactly one point
    predictions = [False, False, True, True, False, False, False]
    m = compute_metrics(predictions, labels, window_size=1)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_compute_metrics_no_detections():
    """No anomalies detected."""
    labels = np.array([False, False, True, True, False])
    predictions = [False, False, False, False, False]
    m = compute_metrics(predictions, labels, window_size=1)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0


def test_compute_metrics_all_false_positives():
    """All predictions positive, but no actual anomalies."""
    labels = np.array([False, False, False, False, False])
    predictions = [True, True, True, True, True]
    m = compute_metrics(predictions, labels, window_size=1)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0


def test_compute_metrics_window_overlap():
    """Window-level evaluation: a window is positive if any point in it is anomalous."""
    labels = np.array([False, False, True, False, False, False])
    # window_size=3: window 0 covers [0,1,2] (contains anomaly at idx 2)
    predictions = [True, False, False, False]
    m = compute_metrics(predictions, labels, window_size=3)
    assert m["precision"] == 1.0  # detected window 0, which truly has anomaly
    assert m["recall"] > 0


def test_compute_metrics_partial():
    labels = np.array([False, True, False, True, False, False])
    predictions = [False, True, False, False, False, False]
    m = compute_metrics(predictions, labels, window_size=1)
    assert m["precision"] == 1.0
    assert 0 < m["recall"] < 1.0
    assert 0 < m["f1"] < 1.0
