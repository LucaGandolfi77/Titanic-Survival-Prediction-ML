"""
config.py — Central configuration for the co-evolutionary predator-prey simulation.

Biological analogy:
    This module defines the "genome-level" parameters that shape the evolutionary
    landscape: population sizes, metabolic costs, sensory capabilities (observation
    radius), and reproductive rates (GP crossover / mutation).  Named presets model
    different ecological scenarios analogous to varying habitat visibility (e.g.
    dense forest vs. open savanna).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict


# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """Immutable-by-convention bag of every tuneable knob in the simulation.

    Attributes are grouped by subsystem for readability.  Every random
    operation in the project receives ``seed`` and derives its own
    ``numpy.random.Generator``, guaranteeing full reproducibility.
    """

    # -- World ---------------------------------------------------------------
    grid_size: int = 20
    """Side length N of the toroidal NxN grid."""

    max_steps: int = 100
    """Number of simulation ticks per evaluation episode."""

    initial_food_density: float = 0.15
    """Fraction of cells that start with food."""

    food_respawn_prob: float = 0.02
    """Per-cell probability that an empty cell spawns food each tick."""

    # -- Agents (shared) -----------------------------------------------------
    num_prey: int = 10
    """Number of prey agents instantiated per episode."""

    num_predators: int = 5
    """Number of predator agents instantiated per episode."""

    prey_start_energy: float = 50.0
    """Initial energy of each prey at episode start."""

    predator_start_energy: float = 60.0
    """Initial energy of each predator at episode start."""

    prey_max_energy: float = 100.0
    """Energy cap for prey (food collected beyond this is wasted)."""

    predator_max_energy: float = 100.0
    """Energy cap for predators."""

    prey_metabolic_cost: float = 1.0
    """Energy lost by prey per time step (metabolic tax)."""

    predator_metabolic_cost: float = 1.5
    """Energy lost by predator per time step."""

    prey_food_energy: float = 10.0
    """Energy gained by prey when it eats a food cell."""

    predator_catch_energy: float = 30.0
    """Energy gained by predator when catching a prey."""

    # -- Observation ---------------------------------------------------------
    prey_obs_radius: int = 3
    """Manhattan-distance observation radius for prey."""

    predator_obs_radius: int = 2
    """Manhattan-distance observation radius for predators."""

    # -- GP ------------------------------------------------------------------
    pop_size: int = 200
    """GP population size per species."""

    max_tree_depth: int = 6
    """Maximum depth of GP trees (initial and after genetic operators)."""

    crossover_prob: float = 0.7
    """Probability of crossover (Pc)."""

    mutation_prob: float = 0.2
    """Probability of sub-tree mutation (Pm)."""

    tournament_size: int = 5
    """Tournament selection size."""

    generations: int = 100
    """Number of co-evolutionary generations."""

    episodes_per_eval: int = 5
    """Independent episodes averaged for each fitness evaluation."""

    opponent_sample_k: int = 5
    """Number of opponents sampled for each individual's evaluation."""

    # -- Reproducibility -----------------------------------------------------
    seed: int = 42
    """Master random seed."""

    # -- Logging / Output ----------------------------------------------------
    output_dir: str = "output"
    """Directory for CSV logs, plots, and serialised GP trees."""

    log_tree_every: int = 10
    """Save best GP trees as expressions every N generations."""

    # -- Behaviour detection thresholds --------------------------------------
    flock_distance_threshold: float = 3.0
    """Average pairwise distance below which prey are considered flocking."""

    pack_angle_diversity_threshold: float = 45.0
    """Minimum angle spread (degrees) among predators targeting same prey."""

    ambush_stationary_steps: int = 5
    """Consecutive stationary steps near food to qualify as ambushing."""

    decoy_distance_delta: float = 2.0
    """Signed distance decrease toward predator to detect decoying."""

    herd_spread_threshold: float = 8.0
    """Minimum average pairwise distance among predators to detect herding."""

    # -- Misc ----------------------------------------------------------------
    energy_low_fraction: float = 0.30
    """Threshold (fraction of max) below which ENERGY_LOW is True."""


# ---------------------------------------------------------------------------
# Named presets
# ---------------------------------------------------------------------------

def _base() -> SimConfig:
    """Return a fresh default config."""
    return SimConfig()


PRESETS: Dict[str, SimConfig] = {}


def _register(name: str, **overrides: object) -> None:
    cfg = _base()
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"Unknown config key: {k}")
        setattr(cfg, k, v)
    PRESETS[name] = cfg


_register("DEFAULT")
_register("BLIND_PREY", prey_obs_radius=1, predator_obs_radius=2)
_register("BLIND_PREDATOR", prey_obs_radius=3, predator_obs_radius=1)
_register("SYMMETRIC", prey_obs_radius=2, predator_obs_radius=2)
_register("ASYMMETRIC", prey_obs_radius=4, predator_obs_radius=1)


def get_preset(name: str) -> SimConfig:
    """Return a deep copy of a named preset config.

    Args:
        name: One of DEFAULT, BLIND_PREY, BLIND_PREDATOR, SYMMETRIC, ASYMMETRIC.

    Returns:
        A fresh ``SimConfig`` instance.

    Raises:
        KeyError: If the preset name is unknown.
    """
    return copy.deepcopy(PRESETS[name.upper()])
