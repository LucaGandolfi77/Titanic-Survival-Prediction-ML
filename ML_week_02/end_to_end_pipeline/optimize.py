#!/usr/bin/env python
"""
optimize.py — CLI Entry Point for Hyperparameter Optimization
==============================================================
Usage:
    python optimize.py                        # Use defaults from config
    python optimize.py --n-trials 100         # Override trial count
    python optimize.py --timeout 600          # Override timeout (seconds)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import (
    fit_and_save_preprocessor,
    load_and_split,
    load_pickle,
    transform_validation,
)
from src.models.evaluator import generate_report, plot_confusion_matrix
from src.optimization.optuna_optimizer import run_optimization
from src.utils.config_loader import load_config, setup_logging

logger = logging.getLogger("titanic_mlops.optimize")


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna HPO for Titanic models")
    parser.add_argument("--config", type=Path, default=None, help="Config path")
    parser.add_argument("--n-trials", type=int, default=None, help="Override n_trials")
    parser.add_argument("--timeout", type=int, default=None, help="Override timeout (s)")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    if args.n_trials is not None:
        cfg["optuna"]["n_trials"] = args.n_trials
    if args.timeout is not None:
        cfg["optuna"]["timeout"] = args.timeout

    logger.info("=" * 60)
    logger.info("Titanic MLOps Pipeline — Hyperparameter Optimization")
    logger.info("n_trials=%d  timeout=%ss",
                cfg["optuna"]["n_trials"], cfg["optuna"].get("timeout"))
    logger.info("=" * 60)

    # 1. Load & preprocess (reuse existing artefacts if available)
    processed_dir: Path = cfg["paths"]["processed_dir"]
    preprocessor_path = processed_dir / "preprocessor.pkl"

    if preprocessor_path.exists():
        logger.info("Reusing existing preprocessor from %s", preprocessor_path)
        X_train = load_pickle(processed_dir / "X_train.pkl")
        X_val = load_pickle(processed_dir / "X_test.pkl")
        y_train = load_pickle(processed_dir / "y_train.pkl")

        # We still need y_val — re-split to get it
        _, X_val_raw, _, y_val = load_and_split(cfg)
        preprocessor = load_pickle(preprocessor_path)
        X_val = preprocessor.transform(X_val_raw)
    else:
        X_train_raw, X_val_raw, y_train_series, y_val = load_and_split(cfg)
        X_train, preprocessor = fit_and_save_preprocessor(
            X_train_raw, y_train_series, cfg
        )
        X_val = transform_validation(X_val_raw, preprocessor, cfg)
        y_train = y_train_series.values

    y_val_arr = y_val.values if hasattr(y_val, "values") else y_val

    # 2. Optimise
    result = run_optimization(X_train, y_train, X_val, y_val_arr, cfg)

    logger.info("Best params: %s", result["best_params"])
    logger.info("Best CV accuracy: %.4f", result["best_score"])
    logger.info("Validation metrics: %s", result["metrics"])

    # 3. Report
    report = generate_report(
        result["model"], X_val, y_val_arr,
        output_dir=cfg["paths"]["models_dir"],
    )
    plot_confusion_matrix(
        result["model"], X_val, y_val_arr,
        output_dir=cfg["paths"]["models_dir"],
    )
    logger.info("\n%s", report)
    logger.info("Optimization complete ✓")


if __name__ == "__main__":
    main()
