"""
Experiment 5 — Diversity vs Accuracy Trade-off (THESIS CORE)
==============================================================
The central experiment: compare homogeneous vs heterogeneous ensembles
on the diversity-accuracy plane.  For each method and condition, compute
both diversity metrics and test accuracy, then analyse correlations.
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
from ensembles.diversity_metrics import (
    ambiguity_decomposition,
    compute_all_diversity,
    extract_base_predictions,
)
from ensembles.ensemble_factory import build_method
from experiments.utils import evaluate_method

logger = logging.getLogger(__name__)

ENSEMBLE_METHODS = ("bagging", "random_forest", "adaboost", "gradient_boosting",
                    "hard_voting", "soft_voting")


def run_diversity_accuracy_experiment(
    datasets: List[DatasetBundle],
    noise_rates: List[float] | None = None,
    seeds: List[int] | None = None,
    n_estimators: int = CFG.DEFAULT_N_ESTIMATORS,
    dry_run: bool = False,
) -> pd.DataFrame:
    noise_rates = noise_rates if noise_rates is not None else [0.0, 0.1, 0.2]
    seeds = seeds if seeds is not None else list(CFG.RANDOM_SEEDS)
    ensure_dirs()
    rows: List[Dict] = []

    for X, y, ds_name in datasets:
        logger.info("Exp-DivAccuracy | dataset=%s", ds_name)
        for noise_rate in noise_rates:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                rs = int(rng.integers(0, 2**31))
                y_noisy, _, _ = inject_label_noise(y, noise_rate, rng)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_noisy, test_size=0.3, random_state=rs, stratify=y_noisy,
                )

                for method in ENSEMBLE_METHODS:
                    try:
                        metrics = evaluate_method(
                            method, X_train, y_train, X_test, y_test,
                            n_estimators=n_estimators, random_state=rs,
                        )

                        clf = build_method(method, n_estimators=n_estimators, random_state=rs)
                        clf.fit(X_train, y_train)
                        base_preds = extract_base_predictions(clf, X_test)

                        div = compute_all_diversity(base_preds, y_test)
                        amb = ambiguity_decomposition(base_preds, y_test)

                        kind = "heterogeneous" if method in ("hard_voting", "soft_voting") else "homogeneous"

                        rows.append({
                            "dataset": ds_name, "seed": seed,
                            "noise_rate": noise_rate,
                            "ensemble_type": kind,
                            **metrics,
                            **{f"div_{k}": v for k, v in div.items()},
                            **{f"amb_{k}": v for k, v in amb.items()},
                        })
                    except Exception as exc:
                        logger.warning("Skip %s/%s: %s", ds_name, method, exc)

    df = pd.DataFrame(rows)
    out = Path(CFG.RESULTS_DIR) / "exp_diversity_accuracy.csv"
    df.to_csv(out, index=False)
    logger.info("Exp-DivAccuracy saved → %s (%d rows)", out, len(df))
    return df
