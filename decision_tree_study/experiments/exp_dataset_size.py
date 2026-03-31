"""
Experiment 3 — Learning Curves (Accuracy vs Dataset Size)
==========================================================
For each pruning strategy, subsample the training set at various sizes
and measure generalisation performance.  This answers Q3: "At what
dataset sizes does pruning become most beneficial?"
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
from experiments.exp_pruning import _select_and_build
from trees.tree_metrics import compute_metrics, timed_fit

logger = logging.getLogger(__name__)


def run_dataset_size_experiment(
    datasets: List[DatasetBundle],
    sizes: List[int] | None = None,
    seeds: List[int] | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Run the dataset-size sweep and return tidy results."""
    sizes = sizes if sizes is not None else CFG.DATASET_SIZES
    seeds = seeds if seeds is not None else list(CFG.RANDOM_SEEDS)
    strategies = CFG.PRUNING_STRATEGIES
    ensure_dirs()
    rows: List[Dict] = []

    for X, y, ds_name in datasets:
        logger.info("Exp-DatasetSize | dataset=%s", ds_name)
        n_total = len(y)

        for seed in seeds:
            rng = np.random.default_rng(seed)
            rs = int(rng.integers(0, 2**31))

            # Fixed test set: 20 % of original data
            X_pool, X_test, y_pool, y_test = train_test_split(
                X, y, test_size=0.2, random_state=rs, stratify=y,
            )

            for size in sizes:
                if size > len(y_pool):
                    continue  # skip sizes larger than available pool

                X_sub, y_sub = subsample(X_pool, y_pool, size, rng=rng)

                # 80/20 train/val from the subsample
                try:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_sub, y_sub, test_size=0.2, random_state=rs, stratify=y_sub,
                    )
                except ValueError:
                    # too few samples per class for stratification
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_sub, y_sub, test_size=0.2, random_state=rs,
                    )

                for strat in strategies:
                    try:
                        tree = _select_and_build(strat, X_train, y_train, rs)
                        fit_ms = timed_fit(tree, X_train, y_train)
                        metrics = compute_metrics(
                            tree, X_train, y_train, X_val, y_val, X_test, y_test,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Skipping %s/%s/n=%d/seed=%d: %s",
                            ds_name, strat, size, seed, exc,
                        )
                        continue

                    rows.append({
                        "dataset": ds_name,
                        "dataset_total": n_total,
                        "subsample_size": size,
                        "strategy": strat,
                        "seed": seed,
                        "fit_time_ms": fit_ms,
                        **metrics,
                    })

    df = pd.DataFrame(rows)
    out = Path(CFG.RESULTS_DIR) / "exp_dataset_size.csv"
    df.to_csv(out, index=False)
    logger.info("Exp-DatasetSize results saved → %s  (%d rows)", out, len(df))
    return df
