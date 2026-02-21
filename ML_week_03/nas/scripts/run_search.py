#!/usr/bin/env python3
"""run_search.py — Main NAS entry-point.

Usage
-----
    python scripts/run_search.py                        # default config
    python scripts/run_search.py --config configs/fast.yaml
    python scripts/run_search.py --population 30 --generations 100
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TMPDIR", "/tmp")

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on sys.path so ``src`` is importable
_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from loguru import logger

from src.builder import count_params, build_model
from src.evolution import crossover, mutate, select_parents
from src.fitness import evaluate_population
from src.genome import Genome
from src.predictor import PredictorTrainer
from src.search_space import SearchSpace
from src.trainer import get_cifar10_loaders, get_device
from src.utils import ensure_dirs, load_config, set_seed
from src.visualization import (
    plot_diversity,
    plot_evolution_tree,
    plot_fitness_curve,
    plotly_fitness_curve,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evolutionary NAS for CIFAR-10")
    p.add_argument("--config", type=str, default=None, help="YAML config path")
    p.add_argument("--population", type=int, default=None)
    p.add_argument("--generations", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def _override_cfg(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    """Apply CLI overrides into *cfg*."""
    if args.population is not None:
        cfg["evolution"]["population"] = args.population
    if args.generations is not None:
        cfg["evolution"]["generations"] = args.generations
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device is not None:
        cfg["device"] = args.device


# ── main loop ────────────────────────────────────────────────────────────────

def run_search(cfg: Dict[str, Any]) -> Genome:
    """Execute the full evolutionary NAS loop.

    Returns the best genome found.
    """
    set_seed(cfg.get("seed", 42))
    dirs = ensure_dirs(cfg)
    device = get_device(cfg.get("device", "auto"))
    logger.info(f"Device: {device}")

    space = SearchSpace(cfg["search_space"])
    ecfg = cfg["evolution"]
    pop_size = ecfg["population"]
    n_gens = ecfg["generations"]
    elitism = ecfg.get("elitism", 2)
    crossover_rate = ecfg.get("crossover_rate", 0.8)
    mutation_cfg = ecfg.get("mutation", {})
    tournament_k = ecfg.get("tournament_size", 5)

    # Predictor (PNAS)
    pcfg = cfg.get("predictor", {})
    use_predictor = pcfg.get("enabled", False)
    predictor = PredictorTrainer(hidden_size=pcfg.get("hidden_size", 128)) if use_predictor else None
    pred_start_gen = pcfg.get("train_after_gen", 5)
    pred_top_k = pcfg.get("top_k", 10)

    # Weight bank for inheritance
    wi_enabled = cfg.get("weight_inheritance", {}).get("enabled", False)
    weight_bank: Dict[str, Any] = {} if wi_enabled else None  # type: ignore[assignment]

    # Data loaders (shared across sequential evaluations)
    dcfg = cfg.get("data", {})
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=cfg["training"]["batch_size"],
        val_split=dcfg.get("val_split", 0.1),
        num_workers=cfg.get("num_workers", 2),
        data_dir=str(_PROJ / "data"),
        augment=dcfg.get("augmentation", {}).get("random_flip", True),
    )

    # ── 1. Initialise population ─────────────────────────────────────────
    logger.info(f"Creating initial population of {pop_size} random architectures …")
    population: List[Genome] = [space.random_genome(generation=0) for _ in range(pop_size)]

    all_genomes: List[Genome] = []       # full lineage for visualisation
    fitness_history: List[Dict[str, Any]] = []
    diversity_snapshots: Dict[int, List[Dict[str, Any]]] = {}
    best_ever: Genome = population[0]

    # ── 2. Generation loop ───────────────────────────────────────────────
    for gen in range(n_gens):
        t0 = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"Generation {gen + 1}/{n_gens}  (pop={len(population)})")
        logger.info(f"{'='*60}")

        # Evaluate
        results = evaluate_population(
            population, cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            weight_bank=weight_bank,
        )

        # Record diversity snapshot
        snap = []
        for g, r in zip(population, results):
            try:
                n_params = count_params(build_model(g))
            except Exception:
                n_params = 0
            snap.append({"depth": g.depth, "params": n_params, "fitness": g.fitness or 0.0})
        diversity_snapshots[gen] = snap

        # Record all genomes
        all_genomes.extend(population)

        # Fitness stats
        fitnesses = [g.fitness or 0.0 for g in population]
        gen_best = max(fitnesses)
        gen_mean = sum(fitnesses) / len(fitnesses)
        gen_worst = min(fitnesses)
        fitness_history.append({
            "generation": gen,
            "best": gen_best,
            "mean": gen_mean,
            "worst": gen_worst,
        })

        # Track overall best
        for g in population:
            if (g.fitness or 0.0) > (best_ever.fitness or 0.0):
                best_ever = g

        elapsed = time.time() - t0
        logger.info(
            f"Gen {gen + 1} stats: best={gen_best:.4f}  mean={gen_mean:.4f}  "
            f"worst={gen_worst:.4f}  time={elapsed:.1f}s"
        )
        logger.info(f"Best ever: {best_ever.summary()}")

        # Save best genome per generation
        best_ever.save(dirs["architectures"] / f"gen_{gen:03d}_best.json")

        # ── predictor training ───────────────────────────────────────────
        if predictor is not None:
            predictor.add_observations(population)
            if gen >= pred_start_gen:
                predictor.fit()

        # ── 3. Selection + reproduction ──────────────────────────────────
        if gen == n_gens - 1:
            break  # last gen — no need to produce offspring

        # Elitism: carry the best N unchanged
        sorted_pop = sorted(population, key=lambda g: g.fitness or 0.0, reverse=True)
        next_gen: List[Genome] = [g.clone() for g in sorted_pop[:elitism]]
        for g in next_gen:
            g.fitness = sorted_pop[next_gen.index(g)].fitness  # keep fitness for reference
            g.generation = gen + 1

        # Fill the rest via crossover + mutation
        while len(next_gen) < pop_size:
            p1, p2 = select_parents(population, 2, tournament_k)

            import random as _rng
            if _rng.random() < crossover_rate:
                c1, c2 = crossover(p1, p2, generation=gen + 1, space=space)
            else:
                c1, c2 = p1.clone(), p2.clone()
                c1.generation = c2.generation = gen + 1

            c1 = mutate(c1, space, mutation_cfg)
            c2 = mutate(c2, space, mutation_cfg)

            # Predictor pre-screening
            if predictor is not None and gen >= pred_start_gen:
                candidates = [c1, c2]
                candidates = predictor.rank_and_filter(candidates, top_k=min(pred_top_k, 2))
                next_gen.extend(candidates)
            else:
                next_gen.extend([c1, c2])

        population = next_gen[:pop_size]

    # ── 4. Final outputs ─────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"Search complete!  Best genome: {best_ever.summary()}")
    logger.info(f"{'='*60}")

    # Save best genome
    best_path = dirs["root"] / "best_genome.json"
    best_ever.save(best_path)
    logger.info(f"Best genome saved to {best_path}")

    # Save full history
    history_path = dirs["logs"] / "fitness_history.json"
    history_path.write_text(json.dumps(fitness_history, indent=2))

    lineage_path = dirs["logs"] / "all_genomes.json"
    lineage_path.write_text(json.dumps([g.to_dict() for g in all_genomes], indent=2))

    # Visualisations
    plot_fitness_curve(fitness_history, dirs["plots"] / "fitness_curve.png")
    plotly_fitness_curve(fitness_history, dirs["plots"] / "fitness_curve.html")
    plot_evolution_tree(all_genomes, dirs["plots"] / "evolution_tree.png")
    plot_diversity(diversity_snapshots, dirs["plots"] / "diversity.png")

    logger.info("All outputs saved to " + str(dirs["root"]))
    return best_ever


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    _override_cfg(cfg, args)
    run_search(cfg)


if __name__ == "__main__":
    main()
