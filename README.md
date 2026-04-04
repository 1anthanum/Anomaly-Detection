# Real-time Anomaly Detection System

A modular real-time anomaly detection system using LSTM and Transformer autoencoders, with a Streamlit dashboard for live visualization and an optional Chronos foundation-model baseline for comparison.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Quick Start](#quick-start)
3. [Step-by-Step Tutorial](#step-by-step-tutorial)
4. [Project Structure](#project-structure)
5. [Configuration Reference](#configuration-reference)
6. [Model Comparison (Chronos Baseline)](#model-comparison-chronos-baseline)
7. [Data Sources (CSV / Prometheus / InfluxDB)](#data-sources)
8. [Experiment Tracking (TensorBoard / MLflow)](#experiment-tracking)
9. [Anomaly Types](#anomaly-types)
10. [Multi-variate Support](#multi-variate-support)
11. [Tests](#tests)
12. [FAQ / Troubleshooting](#faq--troubleshooting)

---

## How It Works

The system detects anomalies in time-series data (e.g. CPU metrics) using **reconstruction-based** anomaly detection. The core idea is simple: train an autoencoder to reconstruct *normal* patterns; when a new data point is hard to reconstruct (high error), it is likely anomalous.

```
Raw time series
     │
     ▼
┌────────────────┐    Sliding windows of length 50
│ Preprocessor   │──► (Z-score normalized)
└────────────────┘
     │
     ▼
┌────────────────┐    Learns to compress & reconstruct
│ Autoencoder    │──► normal patterns
│ (LSTM or       │
│  Transformer)  │
└────────────────┘
     │  reconstruction error
     ▼
┌────────────────┐    Adaptive threshold (95th percentile
│ AnomalyScorer  │──► of recent errors) + sigmoid scoring
└────────────────┘
     │  score > threshold?
     ▼
┌────────────────┐    Cooldown to prevent alert fatigue
│ AlertEngine    │──► (suppresses repeated alerts)
└────────────────┘
     │
     ▼
┌────────────────┐    Real-time charts, severity colors,
│ Dashboard      │──► alert log
└────────────────┘
```

The pipeline has **six phases**: data generation → windowing → training → calibration → scoring → alerting. Each phase is handled by a dedicated module, so you can swap any component independently.

---

## Quick Start

### Prerequisites

Python 3.10+ and pip.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train a model

```bash
python -m src.training.pipeline
```

This will generate synthetic CPU data, train the default LSTM autoencoder, calibrate the anomaly scorer, and print precision / recall / F1 on a test set. Trained weights are saved to `weights/best_model.pt`.

### 3. Launch the dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. You will see a sidebar with Start / Pause / Reset controls, live time-series charts, anomaly scores, and an alert log.

### 4. Run tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

### Docker (optional)

```bash
docker build -t anomaly-detector .
docker run -p 8501:8501 anomaly-detector
```

---

## Step-by-Step Tutorial

This section walks through everything that happens when you run the system, from data generation to a live alert appearing on the dashboard.

### Phase 1 — Data Generation

The `TimeSeriesSimulator` (in `src/data/simulator.py`) creates realistic CPU-like time series with daily and weekly seasonality plus Gaussian noise. Three types of anomalies are injected at configurable rates (see [Anomaly Types](#anomaly-types)).

You can explore the simulator in a Python shell:

```python
from src.data import TimeSeriesSimulator

sim = TimeSeriesSimulator()
data = sim.generate_batch(1000)
print(data.keys())   # dict_keys(['values', 'timestamps', 'is_anomaly', 'anomaly_types'])
print(data["values"].shape)  # (1000,)
```

`is_anomaly` is a boolean array (ground truth) and `anomaly_types` records which kind of anomaly each point is (or `"normal"`).

### Phase 2 — Windowing & Normalization

The `TimeSeriesWindower` (in `src/data/preprocessor.py`) converts the 1-D time series into overlapping sliding windows of length `window_size` (default 50). During `fit`, it computes the mean and standard deviation from training data for Z-score normalization.

```python
from src.data import TimeSeriesWindower

windower = TimeSeriesWindower(window_size=50)
windows = windower.prepare(data["values"], fit=True)
print(windows.shape)  # (951, 50, 1) — each window is (50 timesteps, 1 feature)
```

### Phase 3 — Training

The `Trainer` (in `src/training/trainer.py`) trains the autoencoder to minimize MSE between its input and reconstruction on *normal* data only. Key features:

- **Early stopping**: training halts when validation loss stops improving (configurable patience).
- **Gradient clipping**: `clip_grad_norm_` prevents exploding gradients.
- **Learning rate scheduling**: `ReduceLROnPlateau` halves the LR when validation loss plateaus.

```python
from src.training.trainer import Trainer
from src.models import build_model_from_config
import yaml

with open("configs/default.yaml") as f:
    cfg = yaml.safe_load(f)

model = build_model_from_config(cfg)
trainer = Trainer(model)
trainer.fit(train_loader, val_loader, epochs=50, verbose=True)
```

After training, the model is saved with `model.save("weights/best_model.pt")`. The checkpoint contains both the state dict and the full model configuration, so it can be loaded without remembering hyperparameters.

### Phase 4 — Calibration

Before scoring, the `AnomalyScorer` needs to learn what "normal" reconstruction errors look like. This is done by running the trained model on a validation set and recording the error distribution:

```python
from src.detection import AnomalyScorer

cal_errors = trainer.compute_reconstruction_errors(val_loader)
scorer = AnomalyScorer(threshold_percentile=95, window_size=200)
scorer.calibrate(cal_errors)
```

After calibration, the scorer knows the baseline mean and standard deviation of errors, and sets its adaptive threshold at the 95th percentile of recent history.

### Phase 5 — Real-time Scoring

For each incoming window, the scorer computes:

1. **Raw error**: MSE between original and reconstructed window.
2. **Normalized score**: sigmoid function centered at 2 standard deviations above the calibration mean, mapped to `[0, 1]`.
3. **Adaptive threshold**: 95th percentile of the last 200 errors (rolls forward as new data arrives).
4. **Severity**: `"normal"` / `"warning"` / `"critical"` based on how far the score exceeds the threshold.

```python
result = scorer.score(raw_error=0.15)
print(result)
# AnomalyResult(score=0.72, is_anomaly=True, raw_error=0.15,
#               threshold=0.12, threshold_score=0.55, severity='warning')
```

### Phase 6 — Alerting & Dashboard

The `AlertEngine` (in `src/detection/alerts.py`) wraps the scorer with a cooldown mechanism: after an alert fires, subsequent anomalies within `alert_cooldown_steps` steps are suppressed. This prevents a single anomaly burst from flooding the log.

The Streamlit dashboard (`app/`) ties everything together. On startup it loads the config, builds (or loads) the model, calibrates the scorer on a short clean run, and then enters a streaming loop that processes batches of simulated data at the configured update interval.

---

## Project Structure

```
├── app.py                          # Streamlit entry point (thin orchestrator)
├── app/
│   ├── __init__.py
│   ├── config.py                   # YAML loading, factory re-exports
│   ├── state.py                    # Session state init, model loading
│   ├── sidebar.py                  # Sidebar controls (Start/Pause/Reset)
│   └── streaming.py                # Streaming loop + chart rendering
├── src/
│   ├── data/
│   │   ├── simulator.py            # TimeSeriesSimulator, MultiMetricSimulator
│   │   ├── preprocessor.py         # TimeSeriesWindower, WindowDataset
│   │   └── sources.py              # DataSource ABC + CSV/Prometheus/InfluxDB
│   ├── models/
│   │   ├── base.py                 # AnomalyDetector ABC (save/load)
│   │   ├── lstm_autoencoder.py     # LSTM-based autoencoder
│   │   ├── transformer_detector.py # Transformer with learned query tokens
│   │   ├── chronos_baseline.py     # Optional: Chronos foundation model wrapper
│   │   └── factory.py              # Shared build_model / build_simulator
│   ├── detection/
│   │   ├── scoring.py              # AnomalyScorer, AnomalyResult
│   │   └── alerts.py               # AlertEngine with cooldown
│   ├── training/
│   │   ├── trainer.py              # Training loop with early stopping
│   │   ├── pipeline.py             # End-to-end train + evaluate pipeline
│   │   └── logger.py               # TensorBoard / MLflow experiment loggers
│   ├── evaluation/
│   │   └── compare.py              # ModelComparator (autoencoder vs Chronos)
│   ├── visualization/
│   │   └── dashboard.py            # DashboardState, Plotly chart helpers
│   └── export.py                   # CSV export & webhook utilities
├── scripts/
│   └── compare_models.py           # CLI for model comparison
├── configs/
│   └── default.yaml                # All configuration parameters
├── tests/                          # Unit tests (pytest)
│   ├── conftest.py                 # Shared fixtures / path setup
│   ├── test_models.py
│   ├── test_scoring.py
│   ├── test_alerts.py
│   ├── test_preprocessor.py
│   ├── test_simulator.py
│   ├── test_trainer.py
│   └── test_pipeline.py
├── weights/                        # Model checkpoints (gitignored)
├── requirements.txt                # Core dependencies
├── requirements-dev.txt            # Dev dependencies (pytest, ruff)
└── Dockerfile
```

---

## Configuration Reference

All settings live in `configs/default.yaml`. Here is what each section controls:

**`simulator`** — Controls the synthetic data generator. `base_value` is the resting CPU level. `daily_amplitude` and `weekly_amplitude` control seasonal fluctuations. The `anomaly` sub-section sets injection rates for each anomaly type.

**`model`** — `default` selects which model to use (`"lstm"` or `"transformer"`). `window_size` is the sliding-window length. The `lstm` and `transformer` sub-sections contain architecture hyperparameters (hidden dimensions, number of layers, dropout, etc.).

**`detection`** — `threshold_percentile` (default 95) controls how aggressive the threshold is. `history_window` (default 200) is how many recent errors the adaptive threshold looks at. `alert_cooldown_steps` (default 10) suppresses repeated alerts.

**`dashboard`** — `update_interval_ms` controls the streaming speed. `chart_history` sets how many points to display. `max_alerts_display` caps the alert log length.

To override settings, either edit the YAML directly or create a second YAML file and modify `app/config.py` to load it.

---

## Model Comparison (Chronos Baseline)

You can compare the trained autoencoder against Amazon's Chronos foundation model for time-series forecasting, used here as a zero-shot anomaly detector.

### Install Chronos

```bash
pip install chronos-forecasting
```

### Run the comparison

```bash
# Autoencoder only:
python -m scripts.compare_models

# Include Chronos baseline:
python -m scripts.compare_models --with-chronos

# Customise:
python -m scripts.compare_models --with-chronos --epochs 30 --n-train 5000 --n-test 1000
```

The script generates shared train/test data, trains the autoencoder, runs Chronos zero-shot on the same test set, and prints a side-by-side comparison of precision, recall, F1, and runtime.

### How the Chronos baseline works

Chronos is a pre-trained forecasting model, not a native anomaly detector. We adapt it for anomaly detection by: feeding a context window of historical values, sampling 100 forecast trajectories, computing the 5th–95th percentile confidence interval, and flagging any actual value that falls outside the interval as anomalous. This is a standard approach for using forecast models as zero-shot anomaly baselines.

In the Streamlit dashboard, you can enable the Chronos overlay via the sidebar checkbox "Enable Chronos overlay". When active, Chronos anomaly markers (purple triangles) appear alongside the autoencoder markers (red crosses) on the time-series chart, giving you a real-time visual comparison.

---

## Data Sources

By default, the system uses a built-in `TimeSeriesSimulator`. The `DataSource` abstraction (in `src/data/sources.py`) lets you swap in real data without changing any downstream code.

### CSV files

```python
from src.data import CSVSource

source = CSVSource("data/cpu_metrics.csv", value_column="cpu_percent", label_column="anomaly")
batch = source.read_batch()
# batch["values"], batch["is_anomaly"] — same format as the simulator
```

Expected CSV format: a header row with at least a `value` column. Optional columns: `is_anomaly` (0/1), `timestamp`, `anomaly_type`.

### Prometheus

```python
from src.data import PrometheusSource

source = PrometheusSource(
    url="http://localhost:9090",
    query='rate(node_cpu_seconds_total{mode="idle"}[5m])',
    lookback="6h",
    step="60s",
)
for point in source.stream():
    print(point["value"])
```

Requires `requests`: `pip install requests`.

### InfluxDB 2.x

```python
from src.data import InfluxDBSource

source = InfluxDBSource(
    url="http://localhost:8086",
    token="my-token",
    org="my-org",
    bucket="telegraf",
    measurement="cpu",
    field="usage_idle",
    range_str="-1h",
)
batch = source.read_batch()
```

Requires `influxdb-client`: `pip install influxdb-client`.

All sources implement `stream()` (for real-time use) and `read_batch()` (for training / evaluation), returning the same dict format as the simulator.

---

## Experiment Tracking

The `Trainer` accepts an optional `experiment_logger` that records loss curves, learning rates, and hyperparameters to TensorBoard or MLflow.

### TensorBoard

```python
from src.training import Trainer, TensorBoardLogger

logger = TensorBoardLogger(log_dir="runs/experiment_01")
trainer = Trainer(model, experiment_logger=logger)
trainer.fit(train_loader, val_loader, epochs=50)
# then: tensorboard --logdir runs/
```

Requires `tensorboard`: `pip install tensorboard`.

### MLflow

```python
from src.training import Trainer, MLflowLogger

logger = MLflowLogger(experiment_name="anomaly-detection", run_name="lstm-v2")
trainer = Trainer(model, experiment_logger=logger)
trainer.fit(train_loader, val_loader, epochs=50)
# then: mlflow ui
```

Requires `mlflow`: `pip install mlflow`.

If no logger is provided, a `NullLogger` is used (zero overhead, no dependencies).

---

## Anomaly Types

The simulator injects three types of anomalies:

**Point anomalies** — Sudden spikes or drops of approximately 3 standard deviations. These are the easiest for reconstruction-based detectors to catch because the sharp deviation creates a large reconstruction error.

**Contextual anomalies** — Values that are normal in magnitude but appear at abnormal times (e.g. high CPU usage at 3 AM). These are harder to detect because the model must learn temporal patterns, not just value ranges.

**Collective anomalies** — Sustained shifts lasting 10–50 time steps, where the mean level shifts by 1–2 standard deviations. These test whether the model can detect slow drifts, not just instantaneous spikes.

---

## Multi-variate Support

The `MultiMetricSimulator` generates correlated CPU, memory, and network metrics for multi-variate anomaly detection experiments. Both the LSTM and Transformer models support `n_features > 1` natively. To use it, set `n_features` accordingly in your config and modify the simulator instantiation to use `MultiMetricSimulator`.

---

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

The test suite covers the simulator, preprocessor, both model architectures, scoring, alert engine, trainer, and full pipeline. Tests use synthetic data and run without GPU.

---

## FAQ / Troubleshooting

**Q: The dashboard shows "Calibrating..." and never starts streaming.**
A: Calibration generates a short clean data run and passes it through the model to establish baseline errors. On CPU, this takes a few seconds. If it hangs, check that `configs/default.yaml` has reasonable values (especially `window_size` should not be larger than a few hundred).

**Q: I trained a model, but the dashboard does not load it.**
A: The dashboard checks for `weights/best_model.pt` on startup. Make sure training completed successfully (`python -m src.training.pipeline`) and the file exists. Also verify that the model type in the config matches what was trained (e.g. you trained LSTM but config says `"transformer"`).

**Q: `weights_only=True` fails when loading a checkpoint.**
A: PyTorch's `weights_only=True` restricts `torch.load` to only deserialize tensor data. If your checkpoint contains non-tensor metadata (config dict, model class name), you may need to either save the config as a separate JSON file alongside the `.pt` file, or set `weights_only=False` in `src/models/base.py` (with the understanding that this is less secure for untrusted checkpoints).

**Q: I want to use real data instead of the simulator.**
A: Replace the simulator call with your own data loading code. The rest of the pipeline expects a 1-D numpy array of float values. Feed it into `TimeSeriesWindower.prepare()` and everything downstream works the same. You will also need to provide ground-truth labels if you want to compute precision/recall/F1.

**Q: Chronos is too slow for my dataset.**
A: Chronos runs a Transformer forward pass per time step, so it is much slower than the autoencoder on long series. Use `--n-test` to reduce the test set size, or use `amazon/chronos-t5-tiny` (the smallest variant at 8M parameters). The comparison script is meant for evaluation, not real-time use.
