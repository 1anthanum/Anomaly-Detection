#!/usr/bin/env python3
"""
Compare autoencoder-based anomaly detection with the Chronos foundation model baseline.

Usage
-----
    # Autoencoder only (no extra dependencies):
    python -m scripts.compare_models

    # Include Chronos baseline (requires chronos-forecasting):
    python -m scripts.compare_models --with-chronos

    # Customise run:
    python -m scripts.compare_models --with-chronos --epochs 30 --n-train 5000 --n-test 1000
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is importable
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import yaml  # noqa: E402
from src.evaluation import ModelComparator  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Compare anomaly detection models.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML.")
    parser.add_argument("--n-train", type=int, default=10000, help="Training data points.")
    parser.add_argument("--n-test", type=int, default=2000, help="Test data points.")
    parser.add_argument("--epochs", type=int, default=50, help="Autoencoder training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--with-chronos", action="store_true", help="Include Chronos baseline.")
    parser.add_argument(
        "--chronos-model",
        default="amazon/chronos-t5-tiny",
        help="HuggingFace model id for Chronos.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    verbose = not args.quiet
    comparator = ModelComparator(cfg)

    # Step 1: shared data
    comparator.generate_data(n_train=args.n_train, n_test=args.n_test, verbose=verbose)

    # Step 2: autoencoder
    comparator.evaluate_autoencoder(epochs=args.epochs, batch_size=args.batch_size, verbose=verbose)

    # Step 3: Chronos (optional)
    if args.with_chronos:
        try:
            comparator.evaluate_chronos(model_name=args.chronos_model, verbose=verbose)
        except ImportError as e:
            print(f"\nSkipping Chronos baseline: {e}")
            logging.warning("Chronos baseline skipped: %s", e)

    # Step 4: report
    comparator.report(verbose=verbose)


if __name__ == "__main__":
    main()
