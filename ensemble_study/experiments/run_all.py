"""
run_all.py — Orchestrator
==========================
Runs all seven experiments sequentially with timing.

Usage::

    cd ensemble_study
    python -m experiments.run_all            # full run
    python -m experiments.run_all --dry-run  # quick verification (<60s)

Dry-run: Exp 1 + Exp 5 on breast_cancer only,
         n_samples=[50,200], noise=[0,0.1], 3 seeds, n_estimators=10.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import CFG, ensure_dirs
from data.loaders import load_all_datasets, load_real_datasets

from experiments.exp_class_imbalance import run_class_imbalance_experiment
from experiments.exp_dataset_size import run_dataset_size_experiment
from experiments.exp_diversity_accuracy import run_diversity_accuracy_experiment
from experiments.exp_interaction import (
    run_interaction_imbalance_outliers,
    run_interaction_noise_size,
)
from experiments.exp_label_noise import run_label_noise_experiment
from experiments.exp_n_estimators import run_n_estimators_experiment
from experiments.exp_outliers import run_outlier_experiment

logger = logging.getLogger(__name__)


def _dry_datasets():
    return [ds for ds in load_real_datasets() if ds[2] == "breast_cancer"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Ensemble Study experiments")
    parser.add_argument("--dry-run", action="store_true", help="Quick smoke test")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    ensure_dirs()

    if args.dry_run:
        datasets = _dry_datasets()
        seeds = [42, 7, 13]
        sizes = [50, 200]
        noise_rates = [0.0, 0.1]
        n_est = 10
        fractions = [0.0, 0.05]
        ratios = ["1:1", "1:5"]
        methods = CFG.METHOD_NAMES
    else:
        datasets = load_all_datasets()
        seeds = list(CFG.RANDOM_SEEDS)
        sizes = CFG.DATASET_SIZES
        noise_rates = CFG.LABEL_NOISE_RATES
        n_est = CFG.DEFAULT_N_ESTIMATORS
        fractions = CFG.OUTLIER_FRACTIONS
        ratios = CFG.IMBALANCE_RATIOS
        methods = CFG.METHOD_NAMES

    t_total = time.perf_counter()

    def _banner(n: int, name: str):
        logger.info("=" * 60)
        logger.info("Experiment %d / 7 — %s", n, name)
        logger.info("=" * 60)

    # ── Exp 1 ─────────────────────────────────────────────────────
    _banner(1, "Dataset Size Learning Curves")
    t0 = time.perf_counter()
    run_dataset_size_experiment(
        datasets, sizes=sizes, seeds=seeds, methods=methods,
        n_estimators=n_est, dry_run=args.dry_run,
    )
    logger.info("  → %.1f s", time.perf_counter() - t0)

    # ── Exp 2 ─────────────────────────────────────────────────────
    if not args.dry_run:
        _banner(2, "Class Imbalance")
        t0 = time.perf_counter()
        run_class_imbalance_experiment(
            datasets, ratios=ratios, seeds=seeds, methods=methods,
            n_estimators=n_est,
        )
        logger.info("  → %.1f s", time.perf_counter() - t0)

    # ── Exp 3 ─────────────────────────────────────────────────────
    if not args.dry_run:
        _banner(3, "Label Noise")
        t0 = time.perf_counter()
        run_label_noise_experiment(
            datasets, noise_rates=noise_rates, seeds=seeds, methods=methods,
            n_estimators=n_est,
        )
        logger.info("  → %.1f s", time.perf_counter() - t0)

    # ── Exp 4 ─────────────────────────────────────────────────────
    if not args.dry_run:
        _banner(4, "Outliers")
        t0 = time.perf_counter()
        run_outlier_experiment(
            datasets, fractions=fractions, seeds=seeds, methods=methods,
            n_estimators=n_est,
        )
        logger.info("  → %.1f s", time.perf_counter() - t0)

    # ── Exp 5 ─────────────────────────────────────────────────────
    _banner(5, "Diversity vs Accuracy (THESIS CORE)")
    t0 = time.perf_counter()
    run_diversity_accuracy_experiment(
        datasets, noise_rates=noise_rates, seeds=seeds,
        n_estimators=n_est, dry_run=args.dry_run,
    )
    logger.info("  → %.1f s", time.perf_counter() - t0)

    # ── Exp 6 ─────────────────────────────────────────────────────
    if not args.dry_run:
        _banner(6, "N Estimators")
        t0 = time.perf_counter()
        run_n_estimators_experiment(
            datasets, seeds=seeds,
        )
        logger.info("  → %.1f s", time.perf_counter() - t0)

    # ── Exp 7 ─────────────────────────────────────────────────────
    if not args.dry_run:
        _banner(7, "Interaction Effects")
        t0 = time.perf_counter()
        run_interaction_noise_size(
            datasets, noise_rates=noise_rates, sizes=sizes, seeds=seeds,
            methods=methods, n_estimators=n_est,
        )
        run_interaction_imbalance_outliers(
            datasets, ratios=ratios, fractions=fractions, seeds=seeds,
            methods=methods, n_estimators=n_est,
        )
        logger.info("  → %.1f s", time.perf_counter() - t0)

    elapsed = time.perf_counter() - t_total
    logger.info("All experiments finished in %.1f s", elapsed)
    logger.info("Results directory: %s", CFG.RESULTS_DIR)


if __name__ == "__main__":
    main()
