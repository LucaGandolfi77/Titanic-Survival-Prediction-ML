#!/usr/bin/env python3
"""visualize.py — Regenerate plots from saved NAS logs.

Usage
-----
    python scripts/visualize.py outputs/
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TMPDIR", "/tmp")

import argparse
import json
import sys
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from loguru import logger

from src.genome import Genome
from src.visualization import (
    plot_diversity,
    plot_evolution_tree,
    plot_fitness_curve,
    plotly_fitness_curve,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regenerate NAS visualisations")
    p.add_argument("output_dir", type=str, help="Root output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"

    # Fitness curve
    fh_path = logs / "fitness_history.json"
    if fh_path.exists():
        history = json.loads(fh_path.read_text())
        plot_fitness_curve(history, plots / "fitness_curve.png")
        plotly_fitness_curve(history, plots / "fitness_curve.html")
        logger.info("Fitness curve plotted.")
    else:
        logger.warning(f"Not found: {fh_path}")

    # Evolution tree
    ag_path = logs / "all_genomes.json"
    if ag_path.exists():
        data = json.loads(ag_path.read_text())
        genomes = [Genome.from_dict(d) for d in data]
        plot_evolution_tree(genomes, plots / "evolution_tree.png")
        logger.info("Evolution tree plotted.")
    else:
        logger.warning(f"Not found: {ag_path}")

    logger.info(f"Done.  Plots saved in {plots}")


if __name__ == "__main__":
    main()
