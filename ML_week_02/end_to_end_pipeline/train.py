#!/usr/bin/env python
"""
train.py — CLI Entry Point for Training
========================================
Usage:
    python train.py                       # Train all models in config
    python train.py --model xgboost       # Train a single model
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import (
    fit_and_save_preprocessor,
    load_and_split,
    transform_validation,
)
from src.models.evaluator import generate_report, plot_confusion_matrix
from src.models.trainer import train_all_models
from src.utils.config_loader import load_config, setup_logging

logger = logging.getLogger("titanic_mlops.train")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Titanic classification models")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: config/config.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Train only this model (e.g. 'xgboost'). If omitted, trains all.",
    )
    args = parser.parse_args()

    # Setup
    setup_logging()
    cfg = load_config(args.config)

    logger.info("=" * 60)
    logger.info("Titanic MLOps Pipeline — Training")
    logger.info("=" * 60)

    # 1. Load & preprocess
    X_train_raw, X_val_raw, y_train, y_val = load_and_split(cfg)
    X_train, preprocessor = fit_and_save_preprocessor(X_train_raw, y_train, cfg)
    X_val = transform_validation(X_val_raw, preprocessor, cfg)

    # 2. Filter models if --model flag provided
    if args.model:
        available = list(cfg["training"]["models"].keys())
        if args.model not in available:
            logger.error("Model '%s' not in config. Available: %s", args.model, available)
            sys.exit(1)
        cfg["training"]["models"] = {
            args.model: cfg["training"]["models"][args.model]
        }

    # 3. Train
    results = train_all_models(X_train, y_train.values, X_val, y_val.values, cfg)

    # 4. Generate reports for best model
    best_name = max(results, key=lambda k: results[k]["metrics"]["accuracy"])
    best = results[best_name]
    logger.info("Best model: %s (accuracy=%.4f)", best_name, best["metrics"]["accuracy"])

    report = generate_report(
        best["model"], X_val, y_val.values,
        output_dir=cfg["paths"]["models_dir"],
    )
    plot_confusion_matrix(
        best["model"], X_val, y_val.values,
        output_dir=cfg["paths"]["models_dir"],
    )

    logger.info("\n%s", report)
    logger.info("Training complete ✓")


if __name__ == "__main__":
    main()
