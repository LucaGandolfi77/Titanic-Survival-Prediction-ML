"""
Experiment 1 — Optimal K Selection
=====================================
Compare elbow, silhouette, and gap statistic on every dataset.
Also compare how each algorithm behaves when given the "wrong" k.
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
from validation.k_selection import select_k_elbow, select_k_silhouette, select_k_gap

logger = logging.getLogger(__name__)


def run_k_selection(
    datasets: dict | None = None,
    k_range: list | None = None,
    seeds: list | None = None,
    methods: list | None = None,
) -> pd.DataFrame:
    """Evaluate clustering at each k, collect internal + external indices."""
    ensure_dirs()
    datasets = datasets or {**ALL_SYNTHETIC, **ALL_REAL}
    k_range = k_range or list(CFG.K_RANGE)
    seeds = seeds or list(CFG.RANDOM_SEEDS)
    methods = methods or list(CFG.METHOD_NAMES)

    rows = []
    for ds_name, ds_fn in datasets.items():
        X, y_true, _ = ds_fn()
        X_s, _ = scale_data(X)
        true_k = len(np.unique(y_true))

        for k in k_range:
            for seed in seeds:
                for method in methods:
                    res = evaluate_clustering(
                        method, X_s, y_true=y_true,
                        n_clusters=k, random_state=seed,
                    )
                    res["dataset"] = ds_name
                    res["k"] = k
                    res["true_k"] = true_k
                    res["seed"] = seed
                    rows.append(res)

        logger.info(f"  Exp-KSelection [{ds_name}] done  "
                    f"({len(k_range)} k × {len(seeds)} seeds × {len(methods)} methods)")

    df = pd.DataFrame(rows)
    out = Path(CFG.RESULTS_DIR) / "exp_k_selection.csv"
    df.to_csv(out, index=False)
    logger.info(f"Exp-KSelection saved → {out} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
                        datefmt="%H:%M:%S")
    run_k_selection(k_range=[2, 3, 4, 5, 6], seeds=[42])
