"""
Experiment 5 — Feature Noise Sweep
====================================
Inject Gaussian noise into features at various sigma-factor levels and
measure how each pruning strategy copes with noisy inputs.
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
from data.noise_injector import inject_feature_noise
from experiments.exp_pruning import _select_and_build
from trees.tree_metrics import compute_metrics, timed_fit

logger = logging.getLogger(__name__)


def run_noise_feature_experiment(
    datasets: List[DatasetBundle],
    sigmas: List[float] | None = None,
    seeds: List[int] | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Sweep feature-noise sigma × pruning strategies."""
    sigmas = sigmas if sigmas is not None else CFG.FEATURE_NOISE_SIGMAS
    seeds = seeds if seeds is not None else list(CFG.RANDOM_SEEDS)
    strategies = CFG.PRUNING_STRATEGIES
    ensure_dirs()
    rows: List[Dict] = []

    for X, y, ds_name in datasets:
        logger.info("Exp-FeatureNoise | dataset=%s", ds_name)
        for sigma in sigmas:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                rs = int(rng.integers(0, 2**31))

                X_noisy, meta = inject_feature_noise(X, sigma_factor=sigma, rng=rng)

                X_train, X_temp, y_train, y_temp = train_test_split(
                    X_noisy, y, test_size=0.4, random_state=rs, stratify=y,
                )
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=0.5, random_state=rs, stratify=y_temp,
                )

                for strat in strategies:
                    tree = _select_and_build(strat, X_train, y_train, rs)
                    fit_ms = timed_fit(tree, X_train, y_train)
                    metrics = compute_metrics(
                        tree, X_train, y_train, X_val, y_val, X_test, y_test,
                    )
                    rows.append({
                        "dataset": ds_name,
                        "sigma_factor": sigma,
                        "strategy": strat,
                        "seed": seed,
                        "fit_time_ms": fit_ms,
                        **metrics,
                    })

    df = pd.DataFrame(rows)
    out = Path(CFG.RESULTS_DIR) / "exp_noise_feature.csv"
    df.to_csv(out, index=False)
    logger.info("Exp-FeatureNoise results saved → %s  (%d rows)", out, len(df))
    return df
