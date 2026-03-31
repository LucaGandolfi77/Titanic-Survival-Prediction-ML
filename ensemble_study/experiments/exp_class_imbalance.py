"""
Experiment 2 — Class Imbalance
================================
Sweep imbalance ratios and measure F1-macro, balanced accuracy, AUC
per ensemble method.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import CFG, ensure_dirs
from data.imbalance_generator import generate_imbalanced
from data.loaders import DatasetBundle
from experiments.utils import evaluate_method

logger = logging.getLogger(__name__)


def run_class_imbalance_experiment(
    datasets: List[DatasetBundle],
    ratios: List[str] | None = None,
    seeds: List[int] | None = None,
    methods: tuple[str, ...] | None = None,
    n_estimators: int = CFG.DEFAULT_N_ESTIMATORS,
    dry_run: bool = False,
) -> pd.DataFrame:
    ratios = ratios if ratios is not None else CFG.IMBALANCE_RATIOS
    seeds = seeds if seeds is not None else list(CFG.RANDOM_SEEDS)
    methods = methods or CFG.METHOD_NAMES
    ensure_dirs()
    rows: List[Dict] = []

    for X, y, ds_name in datasets:
        logger.info("Exp-Imbalance | dataset=%s", ds_name)
        for ratio in ratios:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                rs = int(rng.integers(0, 2**31))
                try:
                    X_imb, y_imb, _ = generate_imbalanced(X, y, ratio, rng)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_imb, y_imb, test_size=0.3, random_state=rs, stratify=y_imb,
                    )
                except (ValueError, Exception) as exc:
                    logger.warning("Skip %s/ratio=%s/seed=%d: %s", ds_name, ratio, seed, exc)
                    continue

                for method in methods:
                    try:
                        metrics = evaluate_method(
                            method, X_train, y_train, X_test, y_test,
                            n_estimators=n_estimators, random_state=rs,
                        )
                        rows.append({
                            "dataset": ds_name, "seed": seed,
                            "imbalance_ratio": ratio, **metrics,
                        })
                    except Exception as exc:
                        logger.warning("Skip %s/%s/ratio=%s: %s", ds_name, method, ratio, exc)

    df = pd.DataFrame(rows)
    out = Path(CFG.RESULTS_DIR) / "exp_class_imbalance.csv"
    df.to_csv(out, index=False)
    logger.info("Exp-Imbalance saved → %s (%d rows)", out, len(df))
    return df
