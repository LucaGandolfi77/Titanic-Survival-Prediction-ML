"""
Experiment 7 — High-Dimensional Behaviour
============================================
How does increasing dimensionality affect clustering quality?
(Curse of dimensionality, concentration of distances.)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from config import CFG, ensure_dirs
from data.preprocessor import scale_data
from experiments.utils import evaluate_clustering

logger = logging.getLogger(__name__)


def run_high_dimensional(
    feature_counts: list | None = None,
    seeds: list | None = None,
    methods: list | None = None,
    n_clusters: int = 5,
    n_samples: int = 500,
) -> pd.DataFrame:
    ensure_dirs()
    feature_counts = feature_counts or CFG.HIGH_DIM_FEATURES
    seeds = seeds or list(CFG.RANDOM_SEEDS)
    methods = methods or list(CFG.METHOD_NAMES)

    rows = []
    for d in feature_counts:
        for seed in seeds:
            X, y_true = make_blobs(
                n_samples=n_samples, n_features=d,
                centers=n_clusters, random_state=seed,
            )
            X_s, _ = scale_data(X)

            for method in methods:
                res = evaluate_clustering(
                    method, X_s, y_true=y_true,
                    n_clusters=n_clusters, random_state=seed,
                )
                res["n_features"] = d
                res["seed"] = seed
                rows.append(res)

        logger.info(f"  Exp-HighDim d={d} done")

    df = pd.DataFrame(rows)
    out = Path(CFG.RESULTS_DIR) / "exp_high_dimensional.csv"
    df.to_csv(out, index=False)
    logger.info(f"Exp-HighDim saved → {out} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
                        datefmt="%H:%M:%S")
    run_high_dimensional(feature_counts=[2, 10, 50], seeds=[42])
