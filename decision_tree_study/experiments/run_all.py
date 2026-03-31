"""
run_all.py — Orchestrator
==========================
Runs all seven experiments in sequence.  Supports ``--dry-run`` mode
that uses only iris, a small depth grid, and 2 seeds to finish in
under 30 seconds for CI/smoke‐testing.

Usage::

    cd decision_tree_study
    python -m experiments.run_all            # full run
    python -m experiments.run_all --dry-run  # quick verification
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ── make project root importable ──────────────────────────────────────
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import CFG, ensure_dirs
from data.loaders import load_all_datasets, load_real_datasets

from experiments.exp_ccp_alpha import run_ccp_alpha_experiment
from experiments.exp_dataset_size import run_dataset_size_experiment
from experiments.exp_depth import run_depth_experiment
from experiments.exp_interaction import (
    run_interaction_noise_depth,
    run_interaction_noise_size,
)
from experiments.exp_noise_feature import run_noise_feature_experiment
from experiments.exp_noise_label import run_noise_label_experiment
from experiments.exp_pruning import run_pruning_experiment

logger = logging.getLogger(__name__)


def _dry_datasets():
    """Return only iris for dry runs."""
    datasets = load_real_datasets()
    return [ds for ds in datasets if ds[2] == "iris"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Decision-Tree study experiments")
    parser.add_argument("--dry-run", action="store_true", help="Quick smoke test")
    args = parser.parse_args()

    level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    ensure_dirs()

    if args.dry_run:
        datasets = _dry_datasets()
        depths = [1, 3, 5]
        seeds = [42, 7]
        noise_rates = [0.0, 0.1]
        sigmas = [0.0, 0.3]
        sizes = [50, 100]
    else:
        datasets = load_all_datasets()
        depths = CFG.DEPTHS
        seeds = list(CFG.RANDOM_SEEDS)
        noise_rates = CFG.LABEL_NOISE_RATES
        sigmas = CFG.FEATURE_NOISE_SIGMAS
        sizes = CFG.DATASET_SIZES

    t0 = time.perf_counter()

    logger.info("=" * 60)
    logger.info("Experiment 1 / 7 — Depth sweep")
    logger.info("=" * 60)
    run_depth_experiment(datasets, depths=depths, seeds=seeds, dry_run=args.dry_run)

    logger.info("=" * 60)
    logger.info("Experiment 2 / 7 — Pruning comparison")
    logger.info("=" * 60)
    run_pruning_experiment(datasets, noise_rates=noise_rates, seeds=seeds, dry_run=args.dry_run)

    logger.info("=" * 60)
    logger.info("Experiment 3 / 7 — Dataset‐size learning curves")
    logger.info("=" * 60)
    run_dataset_size_experiment(datasets, sizes=sizes, seeds=seeds, dry_run=args.dry_run)

    logger.info("=" * 60)
    logger.info("Experiment 4 / 7 — Label noise sweep")
    logger.info("=" * 60)
    run_noise_label_experiment(datasets, noise_rates=noise_rates, seeds=seeds, dry_run=args.dry_run)

    logger.info("=" * 60)
    logger.info("Experiment 5 / 7 — Feature noise sweep")
    logger.info("=" * 60)
    run_noise_feature_experiment(datasets, sigmas=sigmas, seeds=seeds, dry_run=args.dry_run)

    logger.info("=" * 60)
    logger.info("Experiment 6 / 7 — Interaction effects")
    logger.info("=" * 60)
    run_interaction_noise_depth(
        datasets, noise_rates=noise_rates, depths=depths, seeds=seeds, dry_run=args.dry_run,
    )
    run_interaction_noise_size(
        datasets, noise_rates=noise_rates, sizes=sizes, seeds=seeds, dry_run=args.dry_run,
    )

    logger.info("=" * 60)
    logger.info("Experiment 7 / 7 — CCP alpha path")
    logger.info("=" * 60)
    run_ccp_alpha_experiment(datasets, noise_rates=noise_rates, seeds=seeds, dry_run=args.dry_run)

    elapsed = time.perf_counter() - t0
    logger.info("All experiments finished in %.1f s", elapsed)
    logger.info("Results directory: %s", CFG.RESULTS_DIR)


if __name__ == "__main__":
    main()
