"""
Experiment 7 — CCP Alpha Path Analysis
========================================
For each dataset and noise level, compute the full cost-complexity
pruning path and record test accuracy at each alpha.  This visualises
the entire regularisation trajectory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from config import CFG, ensure_dirs
from data.loaders import DatasetBundle
from data.noise_injector import inject_label_noise
from trees.tree_metrics import compute_metrics

logger = logging.getLogger(__name__)


def run_ccp_alpha_experiment(
    datasets: List[DatasetBundle],
    noise_rates: List[float] | None = None,
    seeds: List[int] | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Trace the CCP-alpha path for each dataset × noise combination."""
    noise_rates = noise_rates if noise_rates is not None else [0.0, 0.1, 0.2]
    seeds = seeds if seeds is not None else list(CFG.RANDOM_SEEDS)
    ensure_dirs()
    rows: List[Dict] = []

    for X, y, ds_name in datasets:
        logger.info("Exp-CCP | dataset=%s", ds_name)
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

                # Full tree for pruning path
                full_tree = DecisionTreeClassifier(random_state=rs)
                full_tree.fit(X_train, y_train)
                path = full_tree.cost_complexity_pruning_path(X_train, y_train)
                alphas = path.ccp_alphas
                impurities = path.impurities

                for alpha, imp in zip(alphas, impurities):
                    pruned = DecisionTreeClassifier(ccp_alpha=alpha, random_state=rs)
                    pruned.fit(X_train, y_train)
                    metrics = compute_metrics(
                        pruned, X_train, y_train, X_val, y_val, X_test, y_test,
                    )
                    rows.append({
                        "dataset": ds_name,
                        "noise_rate": noise_rate,
                        "ccp_alpha": alpha,
                        "total_impurity": imp,
                        "seed": seed,
                        **metrics,
                    })

    df = pd.DataFrame(rows)
    out = Path(CFG.RESULTS_DIR) / "exp_ccp_alpha.csv"
    df.to_csv(out, index=False)
    logger.info("Exp-CCP results saved → %s  (%d rows)", out, len(df))
    return df
