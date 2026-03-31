"""
Experiment 4 — Scalability
============================
Measure fit time vs dataset size for each algorithm.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from config import CFG, ensure_dirs
from data.preprocessor import scale_data
from algorithms.algorithm_factory import fit_predict_algorithm

logger = logging.getLogger(__name__)


def run_scalability(
    sizes: list | None = None,
    seeds: list | None = None,
    methods: list | None = None,
    n_clusters: int = 5,
    n_features: int = 10,
) -> pd.DataFrame:
    ensure_dirs()
    sizes = sizes or CFG.SYNTH_N_SAMPLES
    seeds = seeds or list(CFG.RANDOM_SEEDS[:5])
    methods = methods or list(CFG.METHOD_NAMES)

    rows = []
    for n in sizes:
        for seed in seeds:
            X, _ = make_blobs(n_samples=n, n_features=n_features,
                              centers=n_clusters, random_state=seed)
            X_s, _ = scale_data(X)

            for method in methods:
                t0 = time.perf_counter()
                labels, _ = fit_predict_algorithm(
                    method, X_s, n_clusters=n_clusters, random_state=seed,
                )
                elapsed = (time.perf_counter() - t0) * 1000
                rows.append({
                    "method": method,
                    "n_samples": n,
                    "n_features": n_features,
                    "seed": seed,
                    "fit_time_ms": elapsed,
                    "k_found": len(np.unique(labels)),
                })

        logger.info(f"  Exp-Scalability n={n} done")

    df = pd.DataFrame(rows)
    out = Path(CFG.RESULTS_DIR) / "exp_scalability.csv"
    df.to_csv(out, index=False)
    logger.info(f"Exp-Scalability saved → {out} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
                        datefmt="%H:%M:%S")
    run_scalability(sizes=[200, 500, 1000], seeds=[42])
