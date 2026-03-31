"""
Experiment 4 — Outlier Fraction Sweep
=======================================
Evaluate ensemble robustness to synthetic feature-space outliers.
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
from data.outlier_injector import inject_outliers
from experiments.utils import evaluate_method

logger = logging.getLogger(__name__)


def run_outlier_experiment(
    datasets: List[DatasetBundle],
    fractions: List[float] | None = None,
    seeds: List[int] | None = None,
    methods: tuple[str, ...] | None = None,
    n_estimators: int = CFG.DEFAULT_N_ESTIMATORS,
    dry_run: bool = False,
) -> pd.DataFrame:
    fractions = fractions if fractions is not None else CFG.OUTLIER_FRACTIONS
    seeds = seeds if seeds is not None else list(CFG.RANDOM_SEEDS)
    methods = methods or CFG.METHOD_NAMES
    ensure_dirs()
    rows: List[Dict] = []

    for X, y, ds_name in datasets:
        logger.info("Exp-Outliers | dataset=%s", ds_name)
        for frac in fractions:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                rs = int(rng.integers(0, 2**31))
                X_out, _, _ = inject_outliers(X, frac, rng)
                X_train, X_test, y_train, y_test = train_test_split(
                    X_out, y, test_size=0.3, random_state=rs, stratify=y,
                )
                for method in methods:
                    try:
                        metrics = evaluate_method(
                            method, X_train, y_train, X_test, y_test,
                            n_estimators=n_estimators, random_state=rs,
                        )
                        rows.append({
                            "dataset": ds_name, "seed": seed,
                            "outlier_fraction": frac, **metrics,
                        })
                    except Exception as exc:
                        logger.warning("Skip %s/%s/frac=%.2f: %s", ds_name, method, frac, exc)

    df = pd.DataFrame(rows)
    out = Path(CFG.RESULTS_DIR) / "exp_outliers.csv"
    df.to_csv(out, index=False)
    logger.info("Exp-Outliers saved → %s (%d rows)", out, len(df))
    return df
