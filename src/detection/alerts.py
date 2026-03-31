"""Alert engine: manages anomaly alerts with deduplication and severity tracking."""

from dataclasses import dataclass, field
from datetime import datetime
from .scoring import AnomalyResult


@dataclass
class Alert:
    timestamp: str
    step: int
    score: float
    severity: str  # "warning" or "critical"
    value: float
    message: str


class AlertEngine:
    """Manages anomaly alerts with cooldown to prevent alert fatigue."""

    def __init__(self, cooldown_steps: int = 10, max_alerts: int = 100):
        self.cooldown_steps = cooldown_steps
        self.max_alerts = max_alerts
        self.alerts: list[Alert] = []
        self._last_alert_step: int = -999

    def check(self, result: AnomalyResult, step: int, value: float) -> Alert | None:
        """Check if an alert should be raised."""
        if not result.is_anomaly:
            return None
        if result.severity == "normal":
            return None
        if step - self._last_alert_step < self.cooldown_steps:
            return None

        self._last_alert_step = step
        alert = Alert(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            step=step,
            score=round(result.score, 3),
            severity=result.severity,
            value=round(value, 2),
            message=f"{'⚠️ WARNING' if result.severity == 'warning' else '🚨 CRITICAL'}: "
                    f"Anomaly detected at step {step} (score: {result.score:.3f}, value: {value:.1f})",
        )
        self.alerts.append(alert)
        if len(self.alerts) > self.max_alerts:
            self.alerts.pop(0)
        return alert

    def get_recent(self, n: int = 10) -> list[Alert]:
        return self.alerts[-n:]

    def get_stats(self) -> dict:
        if not self.alerts:
            return {"total": 0, "warnings": 0, "criticals": 0}
        return {
            "total": len(self.alerts),
            "warnings": sum(1 for a in self.alerts if a.severity == "warning"),
            "criticals": sum(1 for a in self.alerts if a.severity == "critical"),
        }
