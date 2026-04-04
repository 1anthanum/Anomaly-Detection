"""
Streamlit dashboard components for real-time anomaly visualization.
Provides charts, metrics, and alert panels.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import deque
from dataclasses import dataclass


@dataclass
class DashboardState:
    """Holds the rolling state for dashboard rendering.

    Uses fixed-size deques for O(1) append with automatic eviction
    of oldest entries when max_display is exceeded.
    """
    steps: deque
    values: deque
    scores: deque
    thresholds: deque
    anomaly_flags: deque
    max_display: int = 500

    def append(self, step: int, value: float, score: float,
               threshold: float, is_anomaly: bool):
        self.steps.append(step)
        self.values.append(value)
        self.scores.append(score)
        self.thresholds.append(threshold)
        self.anomaly_flags.append(is_anomaly)

    @classmethod
    def create(cls, max_display: int = 500) -> "DashboardState":
        return cls(
            steps=deque(maxlen=max_display),
            values=deque(maxlen=max_display),
            scores=deque(maxlen=max_display),
            thresholds=deque(maxlen=max_display),
            anomaly_flags=deque(maxlen=max_display),
            max_display=max_display,
        )


def create_timeseries_chart(state: DashboardState) -> go.Figure:
    """Create the main time series chart with anomaly markers."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.08,
        subplot_titles=("CPU Usage (%)", "Anomaly Score"),
    )

    # --- Top panel: Time series with anomaly markers ---
    fig.add_trace(
        go.Scatter(
            x=state.steps, y=state.values,
            mode="lines", name="CPU Usage",
            line=dict(color="#3b82f6", width=1.5),
        ),
        row=1, col=1,
    )

    # Highlight anomaly points
    anom_steps = [s for s, f in zip(state.steps, state.anomaly_flags) if f]
    anom_vals = [v for v, f in zip(state.values, state.anomaly_flags) if f]
    if anom_steps:
        fig.add_trace(
            go.Scatter(
                x=anom_steps, y=anom_vals,
                mode="markers", name="Anomaly",
                marker=dict(color="#ef4444", size=7, symbol="x"),
            ),
            row=1, col=1,
        )

    # --- Bottom panel: Anomaly score vs threshold ---
    fig.add_trace(
        go.Scatter(
            x=state.steps, y=state.scores,
            mode="lines", name="Score",
            line=dict(color="#f59e0b", width=1.5),
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=state.steps, y=state.thresholds,
            mode="lines", name="Threshold",
            line=dict(color="#ef4444", width=1, dash="dash"),
        ),
        row=2, col=1,
    )

    fig.update_layout(
        height=500,
        margin=dict(l=50, r=20, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
    )
    fig.update_yaxes(title_text="CPU %", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=1)
    fig.update_xaxes(title_text="Step", row=2, col=1)

    return fig


def render_metrics(total_points: int, anomaly_count: int,
                   current_score: float, alert_stats: dict) -> dict:
    """Prepare metric values for Streamlit st.metric display."""
    anomaly_rate = (anomaly_count / total_points * 100) if total_points > 0 else 0
    return {
        "total_points": total_points,
        "anomaly_rate": f"{anomaly_rate:.1f}%",
        "current_score": f"{current_score:.3f}",
        "total_alerts": alert_stats.get("total", 0),
        "warnings": alert_stats.get("warnings", 0),
        "criticals": alert_stats.get("criticals", 0),
    }


def format_alert_table(alerts: list) -> list[dict]:
    """Format alerts for display in a Streamlit table."""
    return [
        {
            "Time": a.timestamp,
            "Step": a.step,
            "Severity": a.severity.upper(),
            "Score": f"{a.score:.3f}",
            "Value": f"{a.value:.1f}",
            "Message": a.message,
        }
        for a in reversed(alerts)
    ]
