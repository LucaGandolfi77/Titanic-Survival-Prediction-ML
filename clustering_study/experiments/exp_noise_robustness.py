"""
Experiment 6 — Noise Robustness
==================================
How does each algorithm degrade as feature noise or outlier
contamination increases?
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import CFG, ensure_dirs
from data.synthetic import make_blobs_dataset
from data.noise_injector import inject_noise, inject_outliers
from data.preprocessor import scale_data
from experiments.utils import evaluate_clustering

logger = logging.getLogger(__name__)


def run_noise_robustness(
    noise_levels: list | None = None,
    outlier_fractions: list | None = None,
    seeds: list | None = None,
    methods: list | None = None,
    n_clusters: int = 5,
) -> pd.DataFrame:
    ensure_dirs()
    noise_levels = noise_levels or CFG.NOISE_LEVELS
    outlier_fractions = outlier_fractions or CFG.OUTLIER_FRACTIONS
    seeds = seeds or list(CFG.RANDOM_SEEDS)
    methods = methods or list(CFG.METHOD_NAMES)

    rows = []

    # Phase 1: feature noise sweep
    for nl in noise_levels:
        for seed in seeds:
            X, y_true, _ = make_blobs_dataset(
                n_samples=500, centers=n_clusters, random_state=seed,
            )
            rng = np.random.default_rng(seed)
            X_noisy, _ = inject_noise(X, nl, rng)
            X_s, _ = scale_data(X_noisy)

            for method in methods:
                res = evaluate_clustering(
                    method, X_s, y_true=y_true,
                    n_clusters=n_clusters, random_state=seed,
                )
                res["noise_level"] = nl
                res["outlier_fraction"] = 0.0
                res["perturbation"] = "noise"
                res["seed"] = seed
                rows.append(res)

        logger.info(f"  Exp-Noise noise_level={nl:.2f} done")

    # Phase 2: outlier sweep
    for of in outlier_fractions:
        for seed in seeds:
            X, y_true, _ = make_blobs_dataset(
                n_samples=500, centers=n_clusters, random_state=seed,
            )
            rng = np.random.default_rng(seed)
            X_out, _, _ = inject_outliers(X, of, rng)
            X_s, _ = scale_data(X_out)

            for method in methods:
                res = evaluate_clustering(
                    method, X_s, y_true=y_true,
                    n_clusters=n_clusters, random_state=seed,
                )
                res["noise_level"] = 0.0
                res["outlier_fraction"] = of
                res["perturbation"] = "outliers"
                res["seed"] = seed
                rows.append(res)

        logger.info(f"  Exp-Noise outlier_fraction={of:.2f} done")

    df = pd.DataFrame(rows)
    out = Path(CFG.RESULTS_DIR) / "exp_noise_robustness.csv"
    df.to_csv(out, index=False)
    logger.info(f"Exp-NoiseRobustness saved → {out} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
                        datefmt="%H:%M:%S")
    run_noise_robustness(noise_levels=[0.0, 0.1], outlier_fractions=[0.0, 0.1], seeds=[42])
