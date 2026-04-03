"""Tests for alert engine."""


from src.detection.scoring import AnomalyResult
from src.detection.alerts import AlertEngine, Alert


def _make_result(score=0.9, is_anomaly=True, severity="critical"):
    return AnomalyResult(
        score=score, is_anomaly=is_anomaly,
        raw_error=0.5, threshold=0.3, threshold_score=0.7, severity=severity,
    )


def test_alert_on_anomaly():
    engine = AlertEngine(cooldown_steps=5)
    result = _make_result()
    alert = engine.check(result, step=1, value=85.0)
    assert alert is not None
    assert alert.severity == "critical"


def test_no_alert_on_normal():
    engine = AlertEngine()
    result = _make_result(is_anomaly=False, severity="normal")
    alert = engine.check(result, step=1, value=50.0)
    assert alert is None


def test_cooldown():
    engine = AlertEngine(cooldown_steps=10)
    result = _make_result()

    alert1 = engine.check(result, step=1, value=85.0)
    assert alert1 is not None

    # Within cooldown: should be suppressed
    alert2 = engine.check(result, step=5, value=85.0)
    assert alert2 is None

    # After cooldown: should fire
    alert3 = engine.check(result, step=12, value=85.0)
    assert alert3 is not None


def test_get_recent():
    engine = AlertEngine(cooldown_steps=0)
    result = _make_result()

    for i in range(20):
        engine.check(result, step=i, value=80.0)

    recent = engine.get_recent(5)
    assert len(recent) == 5


def test_get_stats():
    engine = AlertEngine(cooldown_steps=0)

    engine.check(_make_result(severity="warning"), step=0, value=70.0)
    engine.check(_make_result(severity="critical"), step=1, value=90.0)
    engine.check(_make_result(severity="warning"), step=2, value=75.0)

    stats = engine.get_stats()
    assert stats["total"] == 3
    assert stats["warnings"] == 2
    assert stats["criticals"] == 1


def test_max_alerts():
    engine = AlertEngine(cooldown_steps=0, max_alerts=5)
    result = _make_result()

    for i in range(10):
        engine.check(result, step=i, value=80.0)

    assert len(engine.alerts) == 5


if __name__ == "__main__":
    test_alert_on_anomaly()
    test_no_alert_on_normal()
    test_cooldown()
    test_get_recent()
    test_get_stats()
    test_max_alerts()
    print("All alert tests passed!")
