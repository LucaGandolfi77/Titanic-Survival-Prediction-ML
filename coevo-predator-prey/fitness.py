"""
fitness.py — Episode runner and fitness evaluation for both species.

Biological analogy:
    Each call to ``run_episode`` simulates one "lifetime" in the habitat.
    Agents are born, forage or hunt, and eventually die or outlast the season
    (``MAX_STEPS``).  Fitness is the reproductive success proxy: how much
    food a prey collected (body condition), how many kills a predator scored,
    and how long each survived.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from config import SimConfig
from world import Grid, Direction
from agents import Agent, Prey, Predator


# ---------------------------------------------------------------------------
# Data structures for trajectory logging
# ---------------------------------------------------------------------------

@dataclass
class AgentSnapshot:
    """Position and status of a single agent at one time step."""
    y: int
    x: int
    energy: float
    alive: bool


@dataclass
class StepRecord:
    """Full state of all agents at one simulation tick."""
    step: int
    prey_snapshots: List[AgentSnapshot]
    predator_snapshots: List[AgentSnapshot]
    food_count: int


@dataclass
class EpisodeResult:
    """Complete trajectory and summary of a single evaluation episode.

    Used by ``behavior_analysis`` and ``visualization`` downstream.
    """
    steps: List[StepRecord] = field(default_factory=list)
    total_prey_food: int = 0
    total_predator_kills: int = 0
    prey_survival_times: List[int] = field(default_factory=list)
    predator_survival_times: List[int] = field(default_factory=list)
    prey_times_caught: List[int] = field(default_factory=list)
    duration: int = 0
    config: Optional[SimConfig] = None


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    prey_funcs: List[Callable[..., float]],
    predator_funcs: List[Callable[..., float]],
    config: SimConfig,
    rng: np.random.Generator,
) -> EpisodeResult:
    """Simulate a single predator-prey episode and return the full trajectory.

    Each prey / predator is assigned its own GP-compiled callable.  If there
    are fewer callables than configured agents, they wrap around.

    Args:
        prey_funcs:     Compiled GP functions for prey (one per prey agent).
        predator_funcs: Compiled GP functions for predator agents.
        config:         Simulation parameters.
        rng:            Seeded random generator (not shared with GP).

    Returns:
        An ``EpisodeResult`` with step-by-step trajectory data.
    """
    grid = Grid(config, rng)

    # -- Instantiate agents ---------------------------------------------------
    prey_agents: List[Prey] = []
    for i in range(config.num_prey):
        p = Prey(
            y=int(rng.integers(0, config.grid_size)),
            x=int(rng.integers(0, config.grid_size)),
            energy=config.prey_start_energy,
            max_energy=config.prey_max_energy,
            metabolic_cost=config.prey_metabolic_cost,
            obs_radius=config.prey_obs_radius,
        )
        func_idx = i % max(len(prey_funcs), 1)
        p.gp_func = prey_funcs[func_idx] if prey_funcs else None
        prey_agents.append(p)

    pred_agents: List[Predator] = []
    for i in range(config.num_predators):
        p = Predator(
            y=int(rng.integers(0, config.grid_size)),
            x=int(rng.integers(0, config.grid_size)),
            energy=config.predator_start_energy,
            max_energy=config.predator_max_energy,
            metabolic_cost=config.predator_metabolic_cost,
            obs_radius=config.predator_obs_radius,
        )
        func_idx = i % max(len(predator_funcs), 1)
        p.gp_func = predator_funcs[func_idx] if predator_funcs else None
        pred_agents.append(p)

    result = EpisodeResult(config=config)

    # -- Main simulation loop ------------------------------------------------
    for step in range(config.max_steps):
        # Record snapshot *before* actions.
        snap = StepRecord(
            step=step,
            prey_snapshots=[
                AgentSnapshot(y=a.y, x=a.x, energy=a.energy, alive=a.alive)
                for a in prey_agents
            ],
            predator_snapshots=[
                AgentSnapshot(y=a.y, x=a.x, energy=a.energy, alive=a.alive)
                for a in pred_agents
            ],
            food_count=grid.count_food(),
        )
        result.steps.append(snap)

        # Early termination: all prey dead or all predators dead.
        if not any(p.alive for p in prey_agents):
            break
        if not any(p.alive for p in pred_agents):
            # Let prey keep foraging if predators die (still counts).
            pass

        # --- Prey act --------------------------------------------------------
        for prey in prey_agents:
            if not prey.alive:
                continue
            obs = prey.observe(grid, pred_agents, prey_agents, config)
            action = prey.decide(obs)
            prey.y, prey.x = grid.move(prey.y, prey.x, action)

            # Eat food.
            if grid.consume_food(prey.y, prey.x):
                prey.energy += config.prey_food_energy
                prey.clamp_energy()
                prey.food_collected += 1

            prey.tick_energy()

        # --- Predators act ---------------------------------------------------
        for pred in pred_agents:
            if not pred.alive:
                continue
            obs = pred.observe(grid, pred_agents, prey_agents, config)
            action = pred.decide(obs)
            pred.y, pred.x = grid.move(pred.y, pred.x, action)

            # Catch prey: any alive prey on the same cell.
            for prey in prey_agents:
                if prey.alive and prey.y == pred.y and prey.x == pred.x:
                    prey.alive = False
                    prey.energy = 0.0
                    prey.times_caught += 1
                    pred.energy += config.predator_catch_energy
                    pred.clamp_energy()
                    pred.kills += 1

            pred.tick_energy()

        # --- Food respawn ----------------------------------------------------
        grid.spawn_food()

    # -- Collect summary statistics ------------------------------------------
    result.duration = len(result.steps)
    result.total_prey_food = sum(p.food_collected for p in prey_agents)
    result.total_predator_kills = sum(p.kills for p in pred_agents)
    result.prey_survival_times = [p.age for p in prey_agents]
    result.predator_survival_times = [p.age for p in pred_agents]
    result.prey_times_caught = [p.times_caught for p in prey_agents]

    return result


# ---------------------------------------------------------------------------
# Fitness evaluation wrappers
# ---------------------------------------------------------------------------

def evaluate_prey(
    individual: Any,
    predator_sample: List[Any],
    config: SimConfig,
    prey_compile: Callable[..., Callable[..., float]],
    pred_compile: Callable[..., Callable[..., float]],
    rng: np.random.Generator,
) -> Tuple[float]:
    """Evaluate a single prey GP individual against a sample of predators.

    Runs ``config.episodes_per_eval`` independent episodes and averages
    the fitness.

    Fitness = (total food) + (survival_time / MAX_STEPS) * 10 − (caught) * 5

    Args:
        individual:       DEAP GP tree (prey).
        predator_sample:  List of predator GP trees to compete against.
        config:           Simulation config.
        prey_compile:     ``toolbox.compile`` for prey.
        pred_compile:     ``toolbox.compile`` for predators.
        rng:              Random generator.

    Returns:
        1-tuple of float fitness (for DEAP).
    """
    prey_func = _safe_compile(prey_compile, individual)
    pred_funcs = [_safe_compile(pred_compile, ind) for ind in predator_sample]

    total_fitness = 0.0
    for _ in range(config.episodes_per_eval):
        ep_rng = np.random.Generator(np.random.PCG64(int(rng.integers(0, 2**31))))
        episode = run_episode(
            prey_funcs=[prey_func] * config.num_prey,
            predator_funcs=pred_funcs,
            config=config,
            rng=ep_rng,
        )
        # Average over prey agents.
        food_sum = episode.total_prey_food / max(config.num_prey, 1)
        surv_sum = sum(episode.prey_survival_times) / max(config.num_prey, 1)
        caught_sum = sum(episode.prey_times_caught) / max(config.num_prey, 1)

        fitness = (
            food_sum
            + (surv_sum / config.max_steps) * 10.0
            - caught_sum * 5.0
        )
        total_fitness += fitness

    avg = total_fitness / max(config.episodes_per_eval, 1)
    return (avg,)


def evaluate_predator(
    individual: Any,
    prey_sample: List[Any],
    config: SimConfig,
    prey_compile: Callable[..., Callable[..., float]],
    pred_compile: Callable[..., Callable[..., float]],
    rng: np.random.Generator,
) -> Tuple[float]:
    """Evaluate a single predator GP individual against a sample of prey.

    Fitness = (total kills) + (energy / max_energy) * 3 − (starvation) * 5

    Args:
        individual:   DEAP GP tree (predator).
        prey_sample:  List of prey GP trees.
        config:       Simulation config.
        prey_compile: ``toolbox.compile`` for prey.
        pred_compile: ``toolbox.compile`` for predators.
        rng:          Random generator.

    Returns:
        1-tuple of float fitness.
    """
    pred_func = _safe_compile(pred_compile, individual)
    prey_funcs = [_safe_compile(prey_compile, ind) for ind in prey_sample]

    total_fitness = 0.0
    for _ in range(config.episodes_per_eval):
        ep_rng = np.random.Generator(np.random.PCG64(int(rng.integers(0, 2**31))))
        episode = run_episode(
            prey_funcs=prey_funcs,
            predator_funcs=[pred_func] * config.num_predators,
            config=config,
            rng=ep_rng,
        )
        kills = episode.total_predator_kills / max(config.num_predators, 1)
        avg_surv = sum(episode.predator_survival_times) / max(config.num_predators, 1)
        died_early = 1.0 if avg_surv < config.max_steps * 0.5 else 0.0

        # Approximate final energy (average of predators).
        # We don't store final energy in EpisodeResult directly, use survival
        # as proxy: longer survival ≈ more energy remaining.
        energy_proxy = avg_surv / config.max_steps

        fitness = (
            kills
            + energy_proxy * 3.0
            - died_early * 5.0
        )
        total_fitness += fitness

    avg = total_fitness / max(config.episodes_per_eval, 1)
    return (avg,)


def _safe_compile(
    compile_fn: Callable[..., Callable[..., float]],
    individual: Any,
) -> Callable[..., float]:
    """Compile a GP individual, returning a no-op function on failure.

    Args:
        compile_fn: DEAP ``toolbox.compile``.
        individual: GP tree.

    Returns:
        Callable that takes observation floats and returns a float.
    """
    try:
        return compile_fn(expr=individual)
    except Exception:
        return lambda *args: 0.0
