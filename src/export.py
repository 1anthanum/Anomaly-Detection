"""
Data export utilities for anomaly detection results.
Supports CSV export and webhook notifications.
"""

import csv
import json
import logging
from pathlib import Path
from datetime import datetime
from urllib.request import Request, urlopen
from urllib.error import URLError

from src.detection.alerts import Alert

logger = logging.getLogger(__name__)


def export_results_csv(
    steps: list[int],
    values: list[float],
    scores: list[float],
    thresholds: list[float],
    anomaly_flags: list[bool],
    output_path: str = "results/detection_results.csv",
) -> str:
    """Export detection results to a CSV file.

    Returns the path of the written file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "value", "score", "threshold", "is_anomaly"])
        for row in zip(steps, values, scores, thresholds, anomaly_flags):
            writer.writerow(row)

    logger.info("Exported %d results to %s", len(steps), path)
    return str(path)


def export_alerts_csv(
    alerts: list[Alert],
    output_path: str = "results/alerts.csv",
) -> str:
    """Export alert history to a CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "step", "score", "severity", "value", "message"])
        for a in alerts:
            writer.writerow([a.timestamp, a.step, a.score, a.severity, a.value, a.message])

    logger.info("Exported %d alerts to %s", len(alerts), path)
    return str(path)


def send_webhook(url: str, alert: Alert, timeout: int = 5) -> bool:
    """Send an alert notification via webhook (POST JSON).

    Returns True if the request succeeded.
    """
    payload = json.dumps({
        "timestamp": alert.timestamp,
        "step": alert.step,
        "score": alert.score,
        "severity": alert.severity,
        "value": alert.value,
        "message": alert.message,
        "sent_at": datetime.now().isoformat(),
    }).encode()

    req = Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:
            logger.info("Webhook sent to %s (status %d)", url, resp.status)
            return resp.status < 400
    except (URLError, OSError) as e:
        logger.warning("Webhook to %s failed: %s", url, e)
        return False
