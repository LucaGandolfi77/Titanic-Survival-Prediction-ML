"""
Experiment 6 — Accuracy vs Number of Estimators
==================================================
Convergence comparison: how quickly each method reaches its plateau.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import CFG, ensure_dirs
from data.loaders import DatasetBundle
from experiments.utils import evaluate_method

logger = logging.getLogger(__name__)

ESTIMATOR_METHODS = ("bagging", "random_forest", "adaboost", "gradient_boosting")


def run_n_estimators_experiment(
    datasets: List[DatasetBundle],
    n_estimators_list: List[int] | None = None,
    seeds: List[int] | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    n_estimators_list = n_estimators_list if n_estimators_list is not None else CFG.N_ESTIMATORS_SWEEP
    seeds = seeds if seeds is not None else list(CFG.RANDOM_SEEDS)
    ensure_dirs()
    rows: List[Dict] = []

    for X, y, ds_name in datasets:
        logger.info("Exp-NEstimators | dataset=%s", ds_name)
        for seed in seeds:
            rng = np.random.default_rng(seed)
            rs = int(rng.integers(0, 2**31))
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=rs, stratify=y,
            )
            for n_est in n_estimators_list:
                for method in ESTIMATOR_METHODS:
                    try:
                        metrics = evaluate_method(
                            method, X_train, y_train, X_test, y_test,
                            n_estimators=n_est, random_state=rs,
                        )
                        rows.append({
                            "dataset": ds_name, "seed": seed,
                            "n_estimators_used": n_est, **metrics,
                        })
                    except Exception as exc:
                        logger.warning("Skip %s/%s/n=%d: %s", ds_name, method, n_est, exc)

    df = pd.DataFrame(rows)
    out = Path(CFG.RESULTS_DIR) / "exp_n_estimators.csv"
    df.to_csv(out, index=False)
    logger.info("Exp-NEstimators saved → %s (%d rows)", out, len(df))
    return df
