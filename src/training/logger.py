"""
Training experiment loggers.

Provides a pluggable ``TrainingLogger`` interface with TensorBoard and
MLflow backends.  The ``Trainer`` can accept any logger to record
loss curves, learning rates, and model metadata without coupling to
a specific tracking framework.

Both backends are optional — the module degrades gracefully if the
dependency is not installed.
"""

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class TrainingLogger(ABC):
    """Abstract interface for training experiment loggers."""

    @abstractmethod
    def log_scalar(self, tag: str, value: float, step: int):
        """Record a single scalar metric."""
        ...

    @abstractmethod
    def log_params(self, params: dict):
        """Record hyperparameters / config at the start of a run."""
        ...

    @abstractmethod
    def close(self):
        """Flush and close the logger."""
        ...


# ------------------------------------------------------------------
# TensorBoard
# ------------------------------------------------------------------


class TensorBoardLogger(TrainingLogger):
    """Logs training metrics to TensorBoard via ``torch.utils.tensorboard``.

    Parameters
    ----------
    log_dir : str
        Directory for TensorBoard event files (default ``"runs/"``).
    """

    def __init__(self, log_dir: str = "runs/"):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError(
                "TensorBoard logging requires tensorboard: pip install tensorboard"
            )
        self.writer = SummaryWriter(log_dir=log_dir)
        logger.info("TensorBoard logger initialized → %s", log_dir)

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def log_params(self, params: dict):
        # TensorBoard stores hyperparams as text
        text = "\n".join(f"**{k}**: {v}" for k, v in params.items())
        self.writer.add_text("hyperparameters", text, 0)

    def close(self):
        self.writer.flush()
        self.writer.close()


# ------------------------------------------------------------------
# MLflow
# ------------------------------------------------------------------


class MLflowLogger(TrainingLogger):
    """Logs training metrics to MLflow.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment name.
    run_name : str | None
        Optional run name.
    tracking_uri : str | None
        MLflow tracking server URI (default: local ``./mlruns``).
    """

    def __init__(
        self,
        experiment_name: str = "anomaly-detection",
        run_name: str | None = None,
        tracking_uri: str | None = None,
    ):
        try:
            import mlflow
        except ImportError:
            raise ImportError("MLflow logging requires mlflow: pip install mlflow")

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run = mlflow.start_run(run_name=run_name)
        self._mlflow = mlflow
        logger.info(
            "MLflow logger initialized → experiment=%s, run=%s",
            experiment_name,
            self._run.info.run_id,
        )

    def log_scalar(self, tag: str, value: float, step: int):
        self._mlflow.log_metric(tag, value, step=step)

    def log_params(self, params: dict):
        self._mlflow.log_params(params)

    def close(self):
        self._mlflow.end_run()


# ------------------------------------------------------------------
# Null logger (no-op fallback)
# ------------------------------------------------------------------


class NullLogger(TrainingLogger):
    """No-op logger used when no tracking backend is configured."""

    def log_scalar(self, tag: str, value: float, step: int):
        pass

    def log_params(self, params: dict):
        pass

    def close(self):
        pass
