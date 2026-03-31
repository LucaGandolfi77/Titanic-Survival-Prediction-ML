"""
Experiment 5 — Adaptive Framework Evaluation
===============================================
Compare the adaptive split/merge framework against fixed-k algorithms:
does it find the right k? is the partition quality competitive?
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import CFG, ensure_dirs
from data.synthetic import ALL_SYNTHETIC
from data.real_datasets import ALL_REAL
from data.preprocessor import scale_data
from experiments.utils import evaluate_clustering
from algorithms.isodata import ISODATA
from algorithms.adaptive_clustering import AdaptiveClustering
from algorithms.algorithm_factory import fit_predict_algorithm

logger = logging.getLogger(__name__)


def run_adaptive(
    datasets: dict | None = None,
    seeds: list | None = None,
) -> pd.DataFrame:
    ensure_dirs()
    datasets = datasets or {**ALL_SYNTHETIC, **ALL_REAL}
    seeds = seeds or list(CFG.RANDOM_SEEDS)
    adaptive_methods = ["isodata", "adaptive"]
    fixed_methods = ["kmeans", "bisecting_kmeans", "gmm"]

    rows = []
    for ds_name, ds_fn in datasets.items():
        X, y_true, _ = ds_fn()
        X_s, _ = scale_data(X)
        true_k = len(np.unique(y_true))

        for seed in seeds:
            # adaptive methods: no ground-truth k given
            for method in adaptive_methods:
                res = evaluate_clustering(
                    method, X_s, y_true=y_true,
                    n_clusters=true_k * 2,  # over-estimated start
                    random_state=seed,
                )
                res["dataset"] = ds_name
                res["true_k"] = true_k
                res["seed"] = seed
                res["k_type"] = "adaptive"
                rows.append(res)

            # fixed methods with correct k (oracle)
            for method in fixed_methods:
                res = evaluate_clustering(
                    method, X_s, y_true=y_true,
                    n_clusters=true_k,
                    random_state=seed,
                )
                res["dataset"] = ds_name
                res["true_k"] = true_k
                res["seed"] = seed
                res["k_type"] = "oracle"
                rows.append(res)

        logger.info(f"  Exp-Adaptive [{ds_name}] done")

    df = pd.DataFrame(rows)
    out = Path(CFG.RESULTS_DIR) / "exp_adaptive.csv"
    df.to_csv(out, index=False)
    logger.info(f"Exp-Adaptive saved → {out} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
                        datefmt="%H:%M:%S")
    run_adaptive(seeds=[42])
