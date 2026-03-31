"""
Run All Experiments
====================
Orchestrator with --dry-run support.

Dry-run: Exp 1 (k_selection) on blobs only, 3 seeds, k=[2,3,4,5]
       + Exp 5 (adaptive) on blobs only, 3 seeds.
Full:   All 7 experiments with full parameter grids.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from config import CFG, ensure_dirs

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clustering Study — Run All Experiments")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick smoke-test (Exp1 + Exp5, blobs, 3 seeds)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    ensure_dirs()
    t_start = time.perf_counter()

    if args.dry_run:
        _run_dry()
    else:
        _run_full()

    elapsed = time.perf_counter() - t_start
    logger.info(f"All experiments finished in {elapsed:.1f} s")
    logger.info(f"Results directory: {CFG.RESULTS_DIR}")


def _run_dry() -> None:
    from data.synthetic import make_blobs_dataset

    dry_datasets = {"blobs": make_blobs_dataset}
    dry_seeds = [42, 7, 13]
    dry_methods = list(CFG.METHOD_NAMES)

    # Exp 1: K selection (small grid)
    logger.info("DRY-RUN: Exp 1 — K Selection")
    t0 = time.perf_counter()
    from experiments.exp_k_selection import run_k_selection
    run_k_selection(
        datasets=dry_datasets,
        k_range=[2, 3, 4, 5],
        seeds=dry_seeds,
        methods=dry_methods,
    )
    logger.info(f"  → {time.perf_counter() - t0:.1f} s")

    # Exp 5: Adaptive
    logger.info("DRY-RUN: Exp 5 — Adaptive Framework")
    t0 = time.perf_counter()
    from experiments.exp_adaptive import run_adaptive
    run_adaptive(datasets=dry_datasets, seeds=dry_seeds)
    logger.info(f"  → {time.perf_counter() - t0:.1f} s")


def _run_full() -> None:
    experiments = [
        ("Exp 1 — K Selection", _exp1),
        ("Exp 2 — Geometry", _exp2),
        ("Exp 3 — Initialisation", _exp3),
        ("Exp 4 — Scalability", _exp4),
        ("Exp 5 — Adaptive", _exp5),
        ("Exp 6 — Noise Robustness", _exp6),
        ("Exp 7 — High Dimensional", _exp7),
    ]
    for name, fn in experiments:
        logger.info(f"Running {name}")
        t0 = time.perf_counter()
        fn()
        logger.info(f"  → {time.perf_counter() - t0:.1f} s")


def _exp1():
    from experiments.exp_k_selection import run_k_selection
    run_k_selection()


def _exp2():
    from experiments.exp_geometry import run_geometry
    run_geometry()


def _exp3():
    from experiments.exp_initialization import run_initialization
    run_initialization()


def _exp4():
    from experiments.exp_scalability import run_scalability
    run_scalability()


def _exp5():
    from experiments.exp_adaptive import run_adaptive
    run_adaptive()


def _exp6():
    from experiments.exp_noise_robustness import run_noise_robustness
    run_noise_robustness()


def _exp7():
    from experiments.exp_high_dimensional import run_high_dimensional
    run_high_dimensional()


if __name__ == "__main__":
    main()
