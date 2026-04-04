"""
Pluggable data source interface.

Provides a unified ``DataSource`` ABC so the pipeline and dashboard can
consume time-series data from the built-in simulator, CSV files,
Prometheus, or InfluxDB — without changing downstream code.

Each source yields ``DataPoint`` dicts that match the contract used by
``TimeSeriesSimulator.generate_point()``.
"""

import csv
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator

import numpy as np

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Abstract base for all time-series data sources.

    Implementations must provide ``stream()`` (for real-time / dashboard use)
    and ``read_batch()`` (for training / evaluation).
    """

    @abstractmethod
    def stream(self) -> Generator[dict, None, None]:
        """Yield data points one at a time (may block for real-time sources).

        Each dict must contain at least:
            step (int), value (float), is_anomaly (bool), anomaly_type (str)
        """
        ...

    @abstractmethod
    def read_batch(self, n_points: int | None = None) -> dict:
        """Return a batch dict with keys: values, is_anomaly, timestamps.

        ``n_points`` limits how many rows are returned (None = all available).
        """
        ...


# ------------------------------------------------------------------
# Built-in simulator adapter
# ------------------------------------------------------------------


class SimulatorSource(DataSource):
    """Wraps ``TimeSeriesSimulator`` behind the ``DataSource`` interface."""

    def __init__(self, simulator):
        self.simulator = simulator

    def stream(self) -> Generator[dict, None, None]:
        yield from self.simulator.stream()

    def read_batch(self, n_points: int | None = None) -> dict:
        if n_points is None:
            n_points = 10000
        return self.simulator.generate_batch(n_points)


# ------------------------------------------------------------------
# CSV file source
# ------------------------------------------------------------------


class CSVSource(DataSource):
    """Read time-series data from a CSV file.

    Expected CSV columns (header row required):
        value          – numeric metric value (required)
        timestamp      – optional; integer or float
        is_anomaly     – optional; 0/1 or True/False ground-truth label
        anomaly_type   – optional; string label

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.
    value_column : str
        Column name for the metric value.
    label_column : str | None
        Column name for the ground-truth anomaly label.
    timestamp_column : str | None
        Column name for the timestamp.
    """

    def __init__(
        self,
        path: str | Path,
        value_column: str = "value",
        label_column: str | None = "is_anomaly",
        timestamp_column: str | None = "timestamp",
    ):
        self.path = Path(path)
        self.value_col = value_column
        self.label_col = label_column
        self.ts_col = timestamp_column
        self._rows: list[dict] | None = None

    def _load(self):
        if self._rows is not None:
            return
        with open(self.path, newline="") as f:
            reader = csv.DictReader(f)
            self._rows = list(reader)
        logger.info("Loaded %d rows from %s", len(self._rows), self.path)

    def _parse_row(self, idx: int, row: dict) -> dict:
        value = float(row[self.value_col])
        is_anomaly = False
        if self.label_col and self.label_col in row:
            raw = row[self.label_col].strip().lower()
            is_anomaly = raw in ("1", "true", "yes")
        anomaly_type = row.get("anomaly_type", "labeled" if is_anomaly else "normal")
        ts = float(row[self.ts_col]) if (self.ts_col and self.ts_col in row) else float(idx)
        return {
            "step": idx,
            "value": value,
            "is_anomaly": is_anomaly,
            "anomaly_type": anomaly_type,
            "timestamp_minutes": ts,
        }

    def stream(self) -> Generator[dict, None, None]:
        self._load()
        for idx, row in enumerate(self._rows):
            yield self._parse_row(idx, row)

    def read_batch(self, n_points: int | None = None) -> dict:
        self._load()
        rows = self._rows[:n_points] if n_points else self._rows
        parsed = [self._parse_row(i, r) for i, r in enumerate(rows)]
        return {
            "values": np.array([p["value"] for p in parsed]),
            "is_anomaly": np.array([p["is_anomaly"] for p in parsed]),
            "timestamps": np.array([p["timestamp_minutes"] for p in parsed]),
        }


# ------------------------------------------------------------------
# Prometheus source (requires ``requests``)
# ------------------------------------------------------------------


class PrometheusSource(DataSource):
    """Pull metrics from a Prometheus server via the HTTP query API.

    Parameters
    ----------
    url : str
        Prometheus server URL (e.g. ``"http://localhost:9090"``).
    query : str
        PromQL query (e.g. ``"rate(node_cpu_seconds_total[5m])"``).
    step : str
        Query resolution step (e.g. ``"60s"``).
    lookback : str
        How far back to query (e.g. ``"1h"``, ``"6h"``).
    """

    def __init__(
        self,
        url: str = "http://localhost:9090",
        query: str = 'rate(node_cpu_seconds_total{mode="idle"}[5m])',
        step: str = "60s",
        lookback: str = "1h",
    ):
        self.url = url.rstrip("/")
        self.query = query
        self.step = step
        self.lookback = lookback

    def _fetch(self) -> list[tuple[float, float]]:
        """Fetch data from Prometheus range query API."""
        try:
            import requests
        except ImportError:
            raise ImportError("requests is required for PrometheusSource: pip install requests")

        import time as _time

        end = _time.time()
        duration_map = {"s": 1, "m": 60, "h": 3600, "d": 86400}
        unit = self.lookback[-1]
        amount = float(self.lookback[:-1])
        seconds = amount * duration_map.get(unit, 3600)
        start = end - seconds

        resp = requests.get(
            f"{self.url}/api/v1/query_range",
            params={"query": self.query, "start": start, "end": end, "step": self.step},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if data["status"] != "success" or not data["data"]["result"]:
            logger.warning("Prometheus returned no data for query: %s", self.query)
            return []

        # Take first result series
        series = data["data"]["result"][0]["values"]
        return [(float(ts), float(val)) for ts, val in series]

    def stream(self) -> Generator[dict, None, None]:
        pairs = self._fetch()
        for idx, (ts, val) in enumerate(pairs):
            yield {
                "step": idx,
                "value": val,
                "is_anomaly": False,
                "anomaly_type": "normal",
                "timestamp_minutes": ts / 60.0,
            }

    def read_batch(self, n_points: int | None = None) -> dict:
        pairs = self._fetch()
        if n_points:
            pairs = pairs[:n_points]
        return {
            "values": np.array([v for _, v in pairs]),
            "is_anomaly": np.zeros(len(pairs), dtype=bool),
            "timestamps": np.array([t for t, _ in pairs]),
        }


# ------------------------------------------------------------------
# InfluxDB source (requires ``influxdb-client``)
# ------------------------------------------------------------------


class InfluxDBSource(DataSource):
    """Pull metrics from InfluxDB 2.x via the client library.

    Parameters
    ----------
    url : str
        InfluxDB server URL.
    token : str
        API token.
    org : str
        Organization name.
    bucket : str
        Bucket name.
    measurement : str
        Measurement to query.
    field : str
        Field key to extract.
    range_str : str
        Flux range (e.g. ``"-1h"``, ``"-6h"``).
    """

    def __init__(
        self,
        url: str = "http://localhost:8086",
        token: str = "",
        org: str = "",
        bucket: str = "default",
        measurement: str = "cpu",
        field: str = "usage_idle",
        range_str: str = "-1h",
    ):
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.measurement = measurement
        self.field = field
        self.range_str = range_str

    def _fetch(self) -> list[tuple[float, float]]:
        try:
            from influxdb_client import InfluxDBClient
        except ImportError:
            raise ImportError(
                "influxdb-client is required for InfluxDBSource: pip install influxdb-client"
            )

        flux = (
            f'from(bucket: "{self.bucket}")'
            f"  |> range(start: {self.range_str})"
            f'  |> filter(fn: (r) => r._measurement == "{self.measurement}")'
            f'  |> filter(fn: (r) => r._field == "{self.field}")'
        )

        client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
        try:
            tables = client.query_api().query(flux)
        finally:
            client.close()

        pairs: list[tuple[float, float]] = []
        for table in tables:
            for record in table.records:
                ts = record.get_time().timestamp()
                val = float(record.get_value())
                pairs.append((ts, val))

        return pairs

    def stream(self) -> Generator[dict, None, None]:
        pairs = self._fetch()
        for idx, (ts, val) in enumerate(pairs):
            yield {
                "step": idx,
                "value": val,
                "is_anomaly": False,
                "anomaly_type": "normal",
                "timestamp_minutes": ts / 60.0,
            }

    def read_batch(self, n_points: int | None = None) -> dict:
        pairs = self._fetch()
        if n_points:
            pairs = pairs[:n_points]
        return {
            "values": np.array([v for _, v in pairs]),
            "is_anomaly": np.zeros(len(pairs), dtype=bool),
            "timestamps": np.array([t for t, _ in pairs]),
        }
