"""Tests for the data simulator."""

import numpy as np

from src.data.simulator import TimeSeriesSimulator, SimulatorConfig, AnomalyConfig


def test_basic_generation():
    sim = TimeSeriesSimulator()
    point = sim.generate_point()
    assert "value" in point
    assert "is_anomaly" in point
    assert "anomaly_type" in point
    assert 0 <= point["value"] <= 100


def test_batch_generation():
    sim = TimeSeriesSimulator()
    batch = sim.generate_batch(500)
    assert len(batch["values"]) == 500
    assert len(batch["is_anomaly"]) == 500
    assert batch["values"].min() >= 0
    assert batch["values"].max() <= 100


def test_anomaly_injection():
    config = SimulatorConfig(anomaly=AnomalyConfig(point_prob=0.5, contextual_prob=0.0, collective_prob=0.0))
    sim = TimeSeriesSimulator(config)
    batch = sim.generate_batch(1000)
    anomaly_rate = batch["is_anomaly"].mean()
    assert anomaly_rate > 0.1, f"Expected anomalies, got rate {anomaly_rate}"


def test_no_anomalies():
    config = SimulatorConfig(anomaly=AnomalyConfig(point_prob=0.0, contextual_prob=0.0, collective_prob=0.0))
    sim = TimeSeriesSimulator(config)
    batch = sim.generate_batch(500)
    assert batch["is_anomaly"].sum() == 0


def test_streaming():
    sim = TimeSeriesSimulator()
    points = list(sim.stream(n_points=100))
    assert len(points) == 100
    assert all("step" in p for p in points)


if __name__ == "__main__":
    test_basic_generation()
    test_batch_generation()
    test_anomaly_injection()
    test_no_anomalies()
    test_streaming()
    print("All tests passed!")
