# Real-time Anomaly Detection System

A modular real-time anomaly detection system using LSTM and Transformer autoencoders, with a Streamlit dashboard for live visualization.

## Architecture

```
TimeSeriesSimulator → Windower → Model (LSTM/Transformer) → AnomalyScorer → AlertEngine → Dashboard
```

**Core modules:**

| Module | Description |
|--------|-------------|
| `src/data/` | Time series simulator with anomaly injection and sliding-window preprocessor |
| `src/models/` | LSTM Autoencoder and Transformer Autoencoder for reconstruction-based detection |
| `src/detection/` | Adaptive anomaly scoring with calibration and alert engine with cooldown |
| `src/training/` | Training pipeline with early stopping, LR scheduling, and evaluation metrics |
| `src/visualization/` | Plotly charts and metric rendering for the Streamlit dashboard |
| `src/export.py` | CSV export and webhook notification utilities |
| `app/` | Streamlit dashboard with real-time streaming, sidebar controls, and model hot-loading |

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Run Dashboard

```bash
streamlit run app.py
```

### Train Model

```bash
python -m src.training.pipeline
```

This generates training data, trains the model, calibrates the scorer, and evaluates on test data with precision/recall/F1 metrics. Trained weights are saved to `weights/best_model.pt` and automatically loaded by the dashboard on next start.

### Docker

```bash
docker build -t anomaly-detector .
docker run -p 8501:8501 anomaly-detector
```

## Configuration

All settings are in `configs/default.yaml`:

- **Simulator**: base value, seasonality, noise, anomaly probabilities
- **Model**: LSTM or Transformer architecture, hidden sizes, dropout
- **Detection**: threshold percentile, history window, alert cooldown
- **Dashboard**: update speed, chart history length

## Anomaly Types

| Type | Description |
|------|-------------|
| Point | Sudden spike or drop (3σ magnitude) |
| Contextual | Normal value at abnormal time (e.g., high CPU at 3 AM) |
| Collective | Sustained shift lasting 10-50 time steps |

## Multi-variate Support

The `MultiMetricSimulator` generates correlated CPU, memory, and network metrics for multi-variate anomaly detection experiments. Models support `n_features > 1` natively.

## Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
├── app.py                  # Streamlit entry point
├── app/                    # Dashboard modules
├── src/
│   ├── data/               # Simulator + preprocessor
│   ├── models/             # LSTM & Transformer autoencoders
│   ├── detection/          # Scoring + alerts
│   ├── training/           # Trainer + pipeline
│   ├── visualization/      # Chart rendering
│   └── export.py           # CSV & webhook export
├── configs/default.yaml    # Configuration
├── tests/                  # Unit tests
├── weights/                # Model checkpoints
└── Dockerfile
```
