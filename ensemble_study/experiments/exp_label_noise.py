"""
Experiment 3 — Label Noise Sweep
==================================
Measure how each ensemble method degrades under increasing label noise.
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
from experiments.utils import evaluate_method

logger = logging.getLogger(__name__)


def run_label_noise_experiment(
    datasets: List[DatasetBundle],
    noise_rates: List[float] | None = None,
    seeds: List[int] | None = None,
    methods: tuple[str, ...] | None = None,
    n_estimators: int = CFG.DEFAULT_N_ESTIMATORS,
    dry_run: bool = False,
) -> pd.DataFrame:
    noise_rates = noise_rates if noise_rates is not None else CFG.LABEL_NOISE_RATES
    seeds = seeds if seeds is not None else list(CFG.RANDOM_SEEDS)
    methods = methods or CFG.METHOD_NAMES
    ensure_dirs()
    rows: List[Dict] = []

    for X, y, ds_name in datasets:
        logger.info("Exp-LabelNoise | dataset=%s", ds_name)
        for noise_rate in noise_rates:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                rs = int(rng.integers(0, 2**31))
                y_noisy, _, _ = inject_label_noise(y, noise_rate, rng)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_noisy, test_size=0.3, random_state=rs, stratify=y_noisy,
                )
                for method in methods:
                    try:
                        metrics = evaluate_method(
                            method, X_train, y_train, X_test, y_test,
                            n_estimators=n_estimators, random_state=rs,
                        )
                        rows.append({
                            "dataset": ds_name, "seed": seed,
                            "noise_rate": noise_rate, **metrics,
                        })
                    except Exception as exc:
                        logger.warning("Skip %s/%s/noise=%.2f: %s", ds_name, method, noise_rate, exc)

    df = pd.DataFrame(rows)
    out = Path(CFG.RESULTS_DIR) / "exp_label_noise.csv"
    df.to_csv(out, index=False)
    logger.info("Exp-LabelNoise saved → %s (%d rows)", out, len(df))
    return df
