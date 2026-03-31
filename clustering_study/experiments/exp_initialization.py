"""
Experiment 3 — Initialisation Sensitivity
============================================
Measure how much each algorithm's result varies across different
random seeds (initialisation instability).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from config import CFG, ensure_dirs
from data.synthetic import ALL_SYNTHETIC
from data.preprocessor import scale_data
from algorithms.algorithm_factory import fit_predict_algorithm

logger = logging.getLogger(__name__)


def run_initialization(
    datasets: dict | None = None,
    seeds: list | None = None,
    methods: list | None = None,
) -> pd.DataFrame:
    ensure_dirs()
    datasets = datasets or ALL_SYNTHETIC
    seeds = seeds or list(CFG.RANDOM_SEEDS)
    methods = methods or list(CFG.METHOD_NAMES)

    rows = []
    for ds_name, ds_fn in datasets.items():
        X, y_true, _ = ds_fn()
        X_s, _ = scale_data(X)
        true_k = len(np.unique(y_true))

        for method in methods:
            labels_all = []
            for seed in seeds:
                labels, _ = fit_predict_algorithm(
                    method, X_s, n_clusters=true_k, random_state=seed,
                )
                labels_all.append(labels)

            # pairwise ARI across seeds
            ari_pairs = []
            for i in range(len(seeds)):
                for j in range(i + 1, len(seeds)):
                    ari = adjusted_rand_score(labels_all[i], labels_all[j])
                    ari_pairs.append(ari)

            rows.append({
                "dataset": ds_name,
                "method": method,
                "true_k": true_k,
                "n_seeds": len(seeds),
                "mean_pairwise_ari": float(np.mean(ari_pairs)),
                "std_pairwise_ari": float(np.std(ari_pairs)),
                "min_pairwise_ari": float(np.min(ari_pairs)),
                "max_pairwise_ari": float(np.max(ari_pairs)),
            })

        logger.info(f"  Exp-Init [{ds_name}] done")

    df = pd.DataFrame(rows)
    out = Path(CFG.RESULTS_DIR) / "exp_initialization.csv"
    df.to_csv(out, index=False)
    logger.info(f"Exp-Init saved → {out} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
                        datefmt="%H:%M:%S")
    run_initialization(seeds=[42, 7, 13])
