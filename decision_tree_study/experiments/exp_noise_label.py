"""
Experiment 4 — Label Noise Sweep
==================================
Sweep label‐noise rate from 0 % to 30 % and measure how each pruning
strategy degrades.  Both symmetric and asymmetric noise modes are tested.
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
from experiments.exp_pruning import _select_and_build
from trees.tree_metrics import compute_metrics, timed_fit

logger = logging.getLogger(__name__)


def run_noise_label_experiment(
    datasets: List[DatasetBundle],
    noise_rates: List[float] | None = None,
    seeds: List[int] | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Sweep label noise rates × pruning strategies."""
    noise_rates = noise_rates if noise_rates is not None else CFG.LABEL_NOISE_RATES
    seeds = seeds if seeds is not None else list(CFG.RANDOM_SEEDS)
    strategies = CFG.PRUNING_STRATEGIES
    ensure_dirs()
    rows: List[Dict] = []

    for X, y, ds_name in datasets:
        logger.info("Exp-LabelNoise | dataset=%s", ds_name)
        for noise_mode in ("symmetric", "asymmetric"):
            for noise_rate in noise_rates:
                for seed in seeds:
                    rng = np.random.default_rng(seed)
                    rs = int(rng.integers(0, 2**31))

                    y_noisy, flipped, meta = inject_label_noise(
                        y, noise_rate, mode=noise_mode, rng=rng,
                    )

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
                            "noise_mode": noise_mode,
                            "noise_rate": noise_rate,
                            "actual_flip_rate": meta["n_affected"] / max(meta["n_total"], 1),
                            "strategy": strat,
                            "seed": seed,
                            "fit_time_ms": fit_ms,
                            **metrics,
                        })

    df = pd.DataFrame(rows)
    out = Path(CFG.RESULTS_DIR) / "exp_noise_label.csv"
    df.to_csv(out, index=False)
    logger.info("Exp-LabelNoise results saved → %s  (%d rows)", out, len(df))
    return df
