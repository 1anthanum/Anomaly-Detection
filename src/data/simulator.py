"""
Streaming time series data simulator with controllable anomaly injection.
Generates realistic server metric patterns (CPU usage) with seasonal trends,
noise, and three types of anomalies.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Generator


@dataclass
class AnomalyConfig:
    """Configuration for anomaly injection."""
    point_prob: float = 0.02        # Probability of point anomaly per step
    contextual_prob: float = 0.005  # Probability of contextual anomaly start
    collective_prob: float = 0.003  # Probability of collective anomaly start
    collective_duration: tuple = (10, 50)  # Min/max duration of collective anomalies
    point_magnitude: float = 3.0    # Std deviations for point anomalies
    collective_shift: float = 1.5   # Magnitude of sustained shift


@dataclass
class SimulatorConfig:
    """Configuration for the data simulator."""
    base_value: float = 50.0        # Base CPU usage %
    daily_amplitude: float = 20.0   # Daily cycle amplitude
    weekly_amplitude: float = 10.0  # Weekly cycle amplitude
    noise_std: float = 2.0          # Gaussian noise std
    sampling_rate: float = 1.0      # Samples per minute
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)


class TimeSeriesSimulator:
    """Generates realistic streaming time series with anomaly injection."""

    def __init__(self, config: SimulatorConfig = None, seed: int | None = 42):
        self.config = config or SimulatorConfig()
        self.step = 0
        self.rng = np.random.default_rng(seed)
        self._collective_remaining = 0
        self._collective_direction = 0

    def _base_pattern(self, t: float) -> float:
        """Generate base seasonal pattern (daily + weekly cycles)."""
        daily = self.config.daily_amplitude * np.sin(2 * np.pi * t / 1440)  # 1440 min/day
        weekly = self.config.weekly_amplitude * np.sin(2 * np.pi * t / 10080)  # 10080 min/week
        return self.config.base_value + daily + weekly

    def _add_noise(self, value: float) -> float:
        """Add Gaussian noise."""
        return value + self.rng.normal(0, self.config.noise_std)

    def _inject_anomaly(self, value: float) -> tuple[float, bool, str]:
        """
        Inject anomalies into the signal.
        Returns: (modified_value, is_anomaly, anomaly_type)
        """
        cfg = self.config.anomaly

        # Ongoing collective anomaly
        if self._collective_remaining > 0:
            self._collective_remaining -= 1
            shifted = value + self._collective_direction * cfg.collective_shift * self.config.noise_std * 5
            return np.clip(shifted, 0, 100), True, "collective"

        # Point anomaly: sudden spike or drop
        if self.rng.random() < cfg.point_prob:
            direction = self.rng.choice([-1, 1])
            spike = value + direction * cfg.point_magnitude * self.config.noise_std * 3
            return np.clip(spike, 0, 100), True, "point"

        # Contextual anomaly: normal value at abnormal time
        if self.rng.random() < cfg.contextual_prob:
            t_minutes = self.step * self.config.sampling_rate
            hour = (t_minutes / 60) % 24
            # Inject high value during typically low period (2-5 AM)
            if 2 <= hour <= 5:
                return np.clip(value + 30, 0, 100), True, "contextual"
            # Inject low value during peak hours (10 AM - 2 PM)
            elif 10 <= hour <= 14:
                return np.clip(value - 30, 0, 100), True, "contextual"

        # Collective anomaly: sustained shift
        if self.rng.random() < cfg.collective_prob:
            duration = self.rng.integers(*cfg.collective_duration)
            self._collective_remaining = duration
            self._collective_direction = self.rng.choice([-1, 1])
            shifted = value + self._collective_direction * cfg.collective_shift * self.config.noise_std * 5
            return np.clip(shifted, 0, 100), True, "collective"

        return value, False, "normal"

    def generate_point(self) -> dict:
        """Generate a single data point."""
        t = self.step * self.config.sampling_rate
        base = self._base_pattern(t)
        noisy = self._add_noise(base)
        value, is_anomaly, anomaly_type = self._inject_anomaly(noisy)

        point = {
            "step": self.step,
            "timestamp_minutes": t,
            "value": round(float(value), 2),
            "is_anomaly": is_anomaly,
            "anomaly_type": anomaly_type,
            "base_value": round(float(base), 2),
        }
        self.step += 1
        return point

    def stream(self, n_points: int = None) -> Generator[dict, None, None]:
        """Generate a stream of data points."""
        count = 0
        while n_points is None or count < n_points:
            yield self.generate_point()
            count += 1

    def generate_batch(self, n_points: int) -> dict:
        """Generate a batch of data points as arrays (useful for training)."""
        points = [self.generate_point() for _ in range(n_points)]
        return {
            "values": np.array([p["value"] for p in points]),
            "is_anomaly": np.array([p["is_anomaly"] for p in points]),
            "anomaly_types": [p["anomaly_type"] for p in points],
            "timestamps": np.array([p["timestamp_minutes"] for p in points]),
        }


class MultiMetricSimulator:
    """Generates correlated multi-variate time series (CPU, memory, network).

    Each metric shares the same anomaly injection timing but has
    independent base patterns and noise, enabling multi-variate
    anomaly detection experiments.
    """

    METRIC_CONFIGS = {
        "cpu": {"base_value": 50.0, "daily_amplitude": 20.0, "weekly_amplitude": 10.0, "noise_std": 2.0},
        "memory": {"base_value": 65.0, "daily_amplitude": 10.0, "weekly_amplitude": 5.0, "noise_std": 1.5},
        "network": {"base_value": 30.0, "daily_amplitude": 25.0, "weekly_amplitude": 15.0, "noise_std": 3.0},
    }

    def __init__(self, metrics: list[str] = None, anomaly: AnomalyConfig = None,
                 sampling_rate: float = 1.0, seed: int | None = 42):
        self.metric_names = metrics or ["cpu", "memory", "network"]
        anomaly_cfg = anomaly or AnomalyConfig()
        self.simulators: dict[str, TimeSeriesSimulator] = {}
        for i, name in enumerate(self.metric_names):
            cfg = self.METRIC_CONFIGS.get(name, self.METRIC_CONFIGS["cpu"])
            self.simulators[name] = TimeSeriesSimulator(
                SimulatorConfig(
                    base_value=cfg["base_value"],
                    daily_amplitude=cfg["daily_amplitude"],
                    weekly_amplitude=cfg["weekly_amplitude"],
                    noise_std=cfg["noise_std"],
                    sampling_rate=sampling_rate,
                    anomaly=anomaly_cfg,
                ),
                seed=seed + i if seed is not None else None,
            )

    @property
    def n_metrics(self) -> int:
        return len(self.metric_names)

    def generate_point(self) -> dict:
        """Generate a single multi-metric data point."""
        points = {name: sim.generate_point() for name, sim in self.simulators.items()}
        first = next(iter(points.values()))
        return {
            "step": first["step"],
            "timestamp_minutes": first["timestamp_minutes"],
            "values": {name: p["value"] for name, p in points.items()},
            "is_anomaly": any(p["is_anomaly"] for p in points.values()),
            "anomaly_types": {name: p["anomaly_type"] for name, p in points.items()},
        }

    def generate_batch(self, n_points: int) -> dict:
        """Generate a batch of multi-metric data points."""
        points = [self.generate_point() for _ in range(n_points)]
        return {
            "values": np.column_stack([
                [p["values"][name] for p in points] for name in self.metric_names
            ]),  # shape: (n_points, n_metrics)
            "is_anomaly": np.array([p["is_anomaly"] for p in points]),
            "metric_names": self.metric_names,
            "timestamps": np.array([p["timestamp_minutes"] for p in points]),
        }


if __name__ == "__main__":
    sim = TimeSeriesSimulator()
    batch = sim.generate_batch(1000)
    n_anomalies = batch["is_anomaly"].sum()
    print(f"Generated 1000 points, {n_anomalies} anomalies ({n_anomalies/10:.1f}%)")
    print(f"Value range: [{batch['values'].min():.1f}, {batch['values'].max():.1f}]")
