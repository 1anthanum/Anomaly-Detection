"""Tests for the evaluation / comparison module."""

import yaml
import pytest

from src.evaluation.compare import ModelComparator, ComparisonResult


@pytest.fixture
def cfg():
    with open("configs/default.yaml") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# ComparisonResult
# ------------------------------------------------------------------

def test_comparison_result_summary():
    r = ComparisonResult(
        name="TestModel",
        precision=0.85,
        recall=0.70,
        f1=0.77,
        n_detected=35,
        n_total=100,
        elapsed_seconds=2.5,
    )
    line = r.summary_line()
    assert "TestModel" in line
    assert "0.85" in line
    assert "0.70" in line


# ------------------------------------------------------------------
# ModelComparator lifecycle
# ------------------------------------------------------------------

def test_comparator_generate_data(cfg):
    comp = ModelComparator(cfg)
    comp.generate_data(n_train=500, n_test=200, verbose=False)
    assert comp._train_data is not None
    assert comp._test_data is not None
    assert len(comp._test_data["values"]) == 200


def test_comparator_evaluate_before_data_raises(cfg):
    comp = ModelComparator(cfg)
    with pytest.raises(RuntimeError, match="generate_data"):
        comp.evaluate_autoencoder(verbose=False)


def test_comparator_report_empty(cfg):
    comp = ModelComparator(cfg)
    results = comp.report(verbose=False)
    assert results == []


def test_comparator_autoencoder_end_to_end(cfg):
    """Full autoencoder evaluation with small data (smoke test)."""
    comp = ModelComparator(cfg)
    comp.generate_data(n_train=500, n_test=200, verbose=False)
    result = comp.evaluate_autoencoder(epochs=2, batch_size=16, verbose=False)

    assert isinstance(result, ComparisonResult)
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0
    assert 0.0 <= result.f1 <= 1.0
    assert result.elapsed_seconds > 0
    assert len(comp.results) == 1
