"""
Experiment 2 — Pruning Strategy Comparison
============================================
Compare all five pruning strategies across multiple noise levels.
For strategies that need CV tuning (pre_depth, pre_samples, ccp,
combined), the best hyper-parameter is selected via inner CV before
the final evaluation on the held-out test set.
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
from data.noise_injector import inject_label_noise
from trees.pruning_strategies import (
    best_ccp_alpha_cv,
    best_combined_cv,
    best_depth_cv,
    best_samples_cv,
)
from trees.tree_factory import build_tree
from trees.tree_metrics import compute_metrics, timed_fit

logger = logging.getLogger(__name__)


def _select_and_build(
    strategy: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    rs: int,
) -> "DecisionTreeClassifier":  # noqa: F821
    """Select hyper-params via inner CV, then build & fit final tree."""
    from sklearn.tree import DecisionTreeClassifier  # type: ignore

    if strategy == "none":
        tree = build_tree("none", random_state=rs)
    elif strategy == "pre_depth":
        best_d, _ = best_depth_cv(X_train, y_train, random_state=rs)
        tree = build_tree("pre_depth", max_depth=best_d, random_state=rs)
    elif strategy == "pre_samples":
        best_leaf, best_split, _ = best_samples_cv(X_train, y_train, random_state=rs)
        tree = build_tree(
            "pre_samples",
            min_samples_leaf=best_leaf,
            min_samples_split=best_split,
            random_state=rs,
        )
    elif strategy == "ccp":
        best_alpha, _, _, _ = best_ccp_alpha_cv(X_train, y_train, random_state=rs)
        tree = build_tree("ccp", ccp_alpha=best_alpha, random_state=rs)
    elif strategy == "combined":
        info = best_combined_cv(X_train, y_train, random_state=rs)
        tree = build_tree(
            "combined",
            max_depth=info["max_depth"],
            ccp_alpha=info["ccp_alpha"],
            random_state=rs,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return tree


def run_pruning_experiment(
    datasets: List[DatasetBundle],
    noise_rates: List[float] | None = None,
    seeds: List[int] | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Compare pruning strategies under varying label noise."""
    noise_rates = noise_rates if noise_rates is not None else CFG.LABEL_NOISE_RATES
    seeds = seeds if seeds is not None else list(CFG.RANDOM_SEEDS)
    strategies = CFG.PRUNING_STRATEGIES
    ensure_dirs()
    rows: List[Dict] = []

    for X, y, ds_name in datasets:
        logger.info("Exp-Pruning | dataset=%s", ds_name)
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

                for strat in strategies:
                    tree = _select_and_build(strat, X_train, y_train, rs)
                    fit_ms = timed_fit(tree, X_train, y_train)
                    metrics = compute_metrics(
                        tree, X_train, y_train, X_val, y_val, X_test, y_test,
                    )
                    rows.append({
                        "dataset": ds_name,
                        "noise_rate": noise_rate,
                        "strategy": strat,
                        "seed": seed,
                        "fit_time_ms": fit_ms,
                        **metrics,
                    })

    df = pd.DataFrame(rows)
    out = Path(CFG.RESULTS_DIR) / "exp_pruning.csv"
    df.to_csv(out, index=False)
    logger.info("Exp-Pruning results saved → %s  (%d rows)", out, len(df))
    return df
