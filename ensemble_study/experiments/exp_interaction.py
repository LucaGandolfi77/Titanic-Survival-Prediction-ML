"""
Experiment 7 — Interaction Effects
====================================
2-D interaction sweeps:
  A) noise × dataset_size → accuracy
  B) imbalance × outlier_fraction → F1
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import CFG, ensure_dirs
from data.dataset_sampler import subsample
from data.imbalance_generator import generate_imbalanced
from data.loaders import DatasetBundle
from data.noise_injector import inject_label_noise
from data.outlier_injector import inject_outliers
from experiments.utils import evaluate_method

logger = logging.getLogger(__name__)


def run_interaction_noise_size(
    datasets: List[DatasetBundle],
    noise_rates: List[float] | None = None,
    sizes: List[int] | None = None,
    seeds: List[int] | None = None,
    methods: tuple[str, ...] | None = None,
    n_estimators: int = CFG.DEFAULT_N_ESTIMATORS,
    dry_run: bool = False,
) -> pd.DataFrame:
    noise_rates = noise_rates if noise_rates is not None else CFG.LABEL_NOISE_RATES
    sizes = sizes if sizes is not None else CFG.DATASET_SIZES
    seeds = seeds if seeds is not None else list(CFG.RANDOM_SEEDS)
    methods = methods or CFG.METHOD_NAMES
    ensure_dirs()
    rows: List[Dict] = []

    for X, y, ds_name in datasets:
        logger.info("Exp-Interaction(noise×size) | dataset=%s", ds_name)
        for noise_rate in noise_rates:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                rs = int(rng.integers(0, 2**31))
                y_noisy, _, _ = inject_label_noise(y, noise_rate, rng)
                X_pool, X_test, y_pool, y_test = train_test_split(
                    X, y_noisy, test_size=0.2, random_state=rs, stratify=y_noisy,
                )
                for size in sizes:
                    if size > len(y_pool):
                        continue
                    try:
                        X_sub, y_sub = subsample(X_pool, y_pool, size, rng=rng)
                    except ValueError:
                        continue
                    for method in methods:
                        try:
                            metrics = evaluate_method(
                                method, X_sub, y_sub, X_test, y_test,
                                n_estimators=n_estimators, random_state=rs,
                            )
                            rows.append({
                                "dataset": ds_name, "seed": seed,
                                "noise_rate": noise_rate,
                                "subsample_size": size, **metrics,
                            })
                        except Exception as exc:
                            logger.warning("Skip: %s", exc)

    df = pd.DataFrame(rows)
    out = Path(CFG.RESULTS_DIR) / "exp_interaction_noise_size.csv"
    df.to_csv(out, index=False)
    logger.info("Interaction(noise×size) saved → %s", out)
    return df


def run_interaction_imbalance_outliers(
    datasets: List[DatasetBundle],
    ratios: List[str] | None = None,
    fractions: List[float] | None = None,
    seeds: List[int] | None = None,
    methods: tuple[str, ...] | None = None,
    n_estimators: int = CFG.DEFAULT_N_ESTIMATORS,
    dry_run: bool = False,
) -> pd.DataFrame:
    ratios = ratios if ratios is not None else CFG.IMBALANCE_RATIOS
    fractions = fractions if fractions is not None else CFG.OUTLIER_FRACTIONS
    seeds = seeds if seeds is not None else list(CFG.RANDOM_SEEDS)
    methods = methods or CFG.METHOD_NAMES
    ensure_dirs()
    rows: List[Dict] = []

    for X, y, ds_name in datasets:
        logger.info("Exp-Interaction(imb×out) | dataset=%s", ds_name)
        for ratio in ratios:
            for frac in fractions:
                for seed in seeds:
                    rng = np.random.default_rng(seed)
                    rs = int(rng.integers(0, 2**31))
                    try:
                        X_imb, y_imb, _ = generate_imbalanced(X, y, ratio, rng)
                        X_out, _, _ = inject_outliers(X_imb, frac, rng)
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_out, y_imb, test_size=0.3, random_state=rs, stratify=y_imb,
                        )
                    except Exception:
                        continue

                    for method in methods:
                        try:
                            metrics = evaluate_method(
                                method, X_train, y_train, X_test, y_test,
                                n_estimators=n_estimators, random_state=rs,
                            )
                            rows.append({
                                "dataset": ds_name, "seed": seed,
                                "imbalance_ratio": ratio,
                                "outlier_fraction": frac, **metrics,
                            })
                        except Exception as exc:
                            logger.warning("Skip: %s", exc)

    df = pd.DataFrame(rows)
    out = Path(CFG.RESULTS_DIR) / "exp_interaction_imb_out.csv"
    df.to_csv(out, index=False)
    logger.info("Interaction(imb×out) saved → %s", out)
    return df
