"""
Experiment 1 — Learning Curves: Accuracy vs Dataset Size
=========================================================
For each method, subsample the training set at various sizes and
measure generalisation performance averaged over multiple seeds.
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
from experiments.utils import evaluate_method

logger = logging.getLogger(__name__)


def run_dataset_size_experiment(
    datasets: List[DatasetBundle],
    sizes: List[int] | None = None,
    seeds: List[int] | None = None,
    methods: tuple[str, ...] | None = None,
    n_estimators: int = CFG.DEFAULT_N_ESTIMATORS,
    dry_run: bool = False,
) -> pd.DataFrame:
    sizes = sizes if sizes is not None else CFG.DATASET_SIZES
    seeds = seeds if seeds is not None else list(CFG.RANDOM_SEEDS)
    methods = methods or CFG.METHOD_NAMES
    ensure_dirs()
    rows: List[Dict] = []

    for X, y, ds_name in datasets:
        logger.info("Exp-DatasetSize | dataset=%s", ds_name)
        for seed in seeds:
            rng = np.random.default_rng(seed)
            rs = int(rng.integers(0, 2**31))
            X_pool, X_test, y_pool, y_test = train_test_split(
                X, y, test_size=0.2, random_state=rs, stratify=y,
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
                            "subsample_size": size, **metrics,
                        })
                    except Exception as exc:
                        logger.warning("Skip %s/%s/n=%d: %s", ds_name, method, size, exc)

    df = pd.DataFrame(rows)
    out = Path(CFG.RESULTS_DIR) / "exp_dataset_size.csv"
    df.to_csv(out, index=False)
    logger.info("Exp-DatasetSize saved → %s (%d rows)", out, len(df))
    return df
