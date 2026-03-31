"""
Experiment 2 — Data Geometry Impact
=====================================
How does the shape of the clusters (isotropic blobs, moons, circles,
anisotropic, varied variance) affect each algorithm?
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import CFG, ensure_dirs
from data.synthetic import ALL_SYNTHETIC
from data.preprocessor import scale_data
from experiments.utils import evaluate_clustering

logger = logging.getLogger(__name__)


def run_geometry(
    seeds: list | None = None,
    methods: list | None = None,
    n_clusters: int = CFG.DEFAULT_K,
) -> pd.DataFrame:
    ensure_dirs()
    seeds = seeds or list(CFG.RANDOM_SEEDS)
    methods = methods or list(CFG.METHOD_NAMES)

    rows = []
    for ds_name, ds_fn in ALL_SYNTHETIC.items():
        X, y_true, _ = ds_fn()
        X_s, _ = scale_data(X)
        true_k = len(set(y_true))

        for seed in seeds:
            for method in methods:
                res = evaluate_clustering(
                    method, X_s, y_true=y_true,
                    n_clusters=true_k, random_state=seed,
                )
                res["dataset"] = ds_name
                res["geometry"] = ds_name
                res["seed"] = seed
                rows.append(res)

        logger.info(f"  Exp-Geometry [{ds_name}] done")

    df = pd.DataFrame(rows)
    out = Path(CFG.RESULTS_DIR) / "exp_geometry.csv"
    df.to_csv(out, index=False)
    logger.info(f"Exp-Geometry saved → {out} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
                        datefmt="%H:%M:%S")
    run_geometry(seeds=[42])
