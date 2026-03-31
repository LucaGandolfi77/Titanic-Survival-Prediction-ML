"""
Experiment 6 — Interaction Effects
====================================
2-D interaction sweeps:
  • noise_rate × max_depth   (label noise interacting with tree complexity)
  • noise_rate × dataset_size (noise interacting with sample budget)
Results are stored as tidy DataFrames suitable for heatmap plotting.
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
from data.loaders import DatasetBundle
from data.noise_injector import inject_label_noise
from trees.tree_factory import build_tree
from trees.tree_metrics import compute_metrics, timed_fit

logger = logging.getLogger(__name__)


# ── Interaction A: noise × depth ──────────────────────────────────────


def run_interaction_noise_depth(
    datasets: List[DatasetBundle],
    noise_rates: List[float] | None = None,
    depths: List[int | None] | None = None,
    seeds: List[int] | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    noise_rates = noise_rates if noise_rates is not None else CFG.LABEL_NOISE_RATES
    depths = depths if depths is not None else CFG.DEPTHS
    seeds = seeds if seeds is not None else list(CFG.RANDOM_SEEDS)
    ensure_dirs()
    rows: List[Dict] = []

    for X, y, ds_name in datasets:
        logger.info("Exp-Interaction(noise×depth) | dataset=%s", ds_name)
        for noise_rate in noise_rates:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                rs = int(rng.integers(0, 2**31))
                y_noisy, _, _ = inject_label_noise(y, noise_rate, rng=rng)

                X_train, X_temp, y_train, y_temp = train_test_split(
                    X, y_noisy, test_size=0.4, random_state=rs, stratify=y_noisy,
                )
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=0.5, random_state=rs, stratify=y_temp,
                )

                for depth in depths:
                    tree = build_tree("none", max_depth=depth, random_state=rs)
                    fit_ms = timed_fit(tree, X_train, y_train)
                    metrics = compute_metrics(
                        tree, X_train, y_train, X_val, y_val, X_test, y_test,
                    )
                    rows.append({
                        "dataset": ds_name,
                        "noise_rate": noise_rate,
                        "max_depth": depth if depth is not None else -1,
                        "seed": seed,
                        "fit_time_ms": fit_ms,
                        **metrics,
                    })

    df = pd.DataFrame(rows)
    out = Path(CFG.RESULTS_DIR) / "exp_interaction_noise_depth.csv"
    df.to_csv(out, index=False)
    logger.info("Exp-Interaction(noise×depth) saved → %s", out)
    return df


# ── Interaction B: noise × dataset size ───────────────────────────────


def run_interaction_noise_size(
    datasets: List[DatasetBundle],
    noise_rates: List[float] | None = None,
    sizes: List[int] | None = None,
    seeds: List[int] | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    noise_rates = noise_rates if noise_rates is not None else CFG.LABEL_NOISE_RATES
    sizes = sizes if sizes is not None else CFG.DATASET_SIZES
    seeds = seeds if seeds is not None else list(CFG.RANDOM_SEEDS)
    ensure_dirs()
    rows: List[Dict] = []

    for X, y, ds_name in datasets:
        logger.info("Exp-Interaction(noise×size) | dataset=%s", ds_name)
        for noise_rate in noise_rates:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                rs = int(rng.integers(0, 2**31))
                y_noisy, _, _ = inject_label_noise(y, noise_rate, rng=rng)

                # Shared test set
                X_pool, X_test, y_pool, y_test = train_test_split(
                    X, y_noisy, test_size=0.2, random_state=rs, stratify=y_noisy,
                )

                for size in sizes:
                    if size > len(y_pool):
                        continue
                    try:
                        X_sub, y_sub = subsample(X_pool, y_pool, size, rng=rng)
                        X_train, X_val, y_train, y_val = train_test_split(
                            X_sub, y_sub, test_size=0.2, random_state=rs, stratify=y_sub,
                        )
                    except ValueError:
                        continue

                    tree = build_tree("none", random_state=rs)
                    fit_ms = timed_fit(tree, X_train, y_train)
                    metrics = compute_metrics(
                        tree, X_train, y_train, X_val, y_val, X_test, y_test,
                    )
                    rows.append({
                        "dataset": ds_name,
                        "noise_rate": noise_rate,
                        "subsample_size": size,
                        "seed": seed,
                        "fit_time_ms": fit_ms,
                        **metrics,
                    })

    df = pd.DataFrame(rows)
    out = Path(CFG.RESULTS_DIR) / "exp_interaction_noise_size.csv"
    df.to_csv(out, index=False)
    logger.info("Exp-Interaction(noise×size) saved → %s", out)
    return df
