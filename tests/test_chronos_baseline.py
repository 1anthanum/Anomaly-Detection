"""Tests for the Chronos baseline wrapper.

These tests verify the module's structure and graceful degradation
without requiring ``chronos-forecasting`` to be installed.
"""

import importlib
import numpy as np
import pytest

from src.models.chronos_baseline import CHRONOS_AVAILABLE


# ------------------------------------------------------------------
# Import / availability guard
# ------------------------------------------------------------------

def test_chronos_available_flag_is_bool():
    """CHRONOS_AVAILABLE should be a boolean regardless of install state."""
    assert isinstance(CHRONOS_AVAILABLE, bool)


def test_import_without_crash():
    """The module must import cleanly even when chronos is missing."""
    mod = importlib.import_module("src.models.chronos_baseline")
    assert hasattr(mod, "ChronosBaseline")
    assert hasattr(mod, "ChronosAnomalyResult")


@pytest.mark.skipif(CHRONOS_AVAILABLE, reason="Only tests missing-dep path")
def test_init_raises_without_chronos():
    from src.models.chronos_baseline import ChronosBaseline

    with pytest.raises(ImportError, match="chronos-forecasting"):
        ChronosBaseline()


# ------------------------------------------------------------------
# ChronosAnomalyResult dataclass
# ------------------------------------------------------------------

def test_anomaly_result_dataclass():
    from src.models.chronos_baseline import ChronosAnomalyResult

    r = ChronosAnomalyResult(
        is_anomaly=True,
        actual=105.0,
        forecast_median=50.0,
        forecast_low=40.0,
        forecast_high=60.0,
        deviation=45.0,
    )
    assert r.is_anomaly is True
    assert r.deviation == 45.0


# ------------------------------------------------------------------
# Live model tests (skipped when chronos is not installed)
# ------------------------------------------------------------------

@pytest.mark.skipif(not CHRONOS_AVAILABLE, reason="chronos-forecasting not installed")
class TestChronosLive:
    """Integration tests that load an actual Chronos model."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from src.models.chronos_baseline import ChronosBaseline
        self.detector = ChronosBaseline(
            model_name="amazon/chronos-t5-tiny",
            context_length=32,
            prediction_length=1,
        )

    def test_forecast_shape(self):
        context = np.sin(np.linspace(0, 4 * np.pi, 32)).astype(np.float32)
        fc = self.detector.forecast(context)
        assert "median" in fc and "low" in fc and "high" in fc
        assert fc["median"].shape == (1,)

    def test_detect_single(self):
        context = np.ones(32, dtype=np.float32) * 50.0
        result = self.detector.detect_single(context, actual=50.0)
        assert isinstance(result.is_anomaly, bool)
        assert result.deviation >= 0.0

    def test_detect_batch_length(self):
        values = np.random.randn(100).astype(np.float32)
        results = self.detector.detect_batch(values)
        expected_len = len(values) - self.detector.context_length
        assert len(results) == expected_len

    def test_get_predictions_returns_bools(self):
        values = np.random.randn(80).astype(np.float32)
        preds = self.detector.get_predictions(values)
        assert all(isinstance(p, bool) for p in preds)
