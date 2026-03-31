"""
Experiment 1 — Accuracy vs Max Depth
=====================================
For each dataset, vary max_depth from the configured grid and measure
train / val / test accuracy (averaged over seeds).  This exposes the
classic bias-variance trade-off.
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
from trees.tree_factory import build_tree
from trees.tree_metrics import compute_metrics, timed_fit

logger = logging.getLogger(__name__)


def run_depth_experiment(
    datasets: List[DatasetBundle],
    depths: List[int | None] | None = None,
    seeds: List[int] | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Run the depth sweep experiment and return a tidy DataFrame."""
    depths = depths if depths is not None else CFG.DEPTHS
    seeds = seeds if seeds is not None else list(CFG.RANDOM_SEEDS)

    ensure_dirs()
    rows: List[Dict] = []

    for X, y, ds_name in datasets:
        logger.info("Exp-Depth | dataset=%s", ds_name)
        for seed in seeds:
            rng = np.random.default_rng(seed)
            rs = int(rng.integers(0, 2**31))
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.4, random_state=rs, stratify=y,
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=rs, stratify=y_temp,
            )
            for depth in depths:
                tree = build_tree("none", max_depth=depth, random_state=rs)
                fit_ms = timed_fit(tree, X_train, y_train)
                metrics = compute_metrics(tree, X_train, y_train, X_val, y_val, X_test, y_test)
                row = {
                    "dataset": ds_name,
                    "seed": seed,
                    "max_depth": depth if depth is not None else -1,
                    "fit_time_ms": fit_ms,
                    **metrics,
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    out = Path(CFG.RESULTS_DIR) / "exp_depth.csv"
    df.to_csv(out, index=False)
    logger.info("Exp-Depth results saved → %s  (%d rows)", out, len(df))
    return df
