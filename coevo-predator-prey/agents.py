"""
agents.py — Prey and Predator agent classes.

Biological analogy:
    Each agent is an individual organism with a position in the habitat, an
    energy reserve (analogous to body condition / fat stores), and a *brain*
    implemented as a GP expression tree.  The ``observe`` method models
    sensory perception limited by the species-specific observation radius —
    think of a prey animal scanning for predators in tall grass, or a raptor
    searching for rodents from above.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from config import SimConfig
from world import Cell, Direction, Grid, DIRECTION_DELTA


# ---------------------------------------------------------------------------
# Action decoding
# ---------------------------------------------------------------------------

def decode_action(value: float) -> Direction:
    """Map a continuous GP tree output to a discrete ``Direction``.

    Encoding bands (from the specification):
        < -0.6  → NORTH
        -0.6 … -0.2 → WEST
        -0.2 … 0.2  → STAY
         0.2 …  0.6 → EAST
        > 0.6  → SOUTH

    Args:
        value: Raw float output of the GP tree.

    Returns:
        A ``Direction`` enum member.
    """
    if value < -0.6:
        return Direction.NORTH
    if value < -0.2:
        return Direction.WEST
    if value < 0.2:
        return Direction.STAY
    if value < 0.6:
        return Direction.EAST
    return Direction.SOUTH


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

@dataclass
class Agent:
    """Base class for all agents in the simulation.

    Not instantiated directly — use ``Prey`` or ``Predator``.

    Attributes:
        y, x:       Current position on the grid.
        energy:     Current energy level (dies at 0).
        max_energy: Cap on energy accumulation.
        metabolic_cost: Energy drained per tick.
        obs_radius: Observation (sensory) radius.
        alive:      Whether the agent is still in play.
        age:        Number of ticks survived so far.
        gp_tree:    Reference to a DEAP GP individual (expression tree).
        gp_func:    Compiled callable of ``gp_tree``.
    """
    y: int = 0
    x: int = 0
    energy: float = 50.0
    max_energy: float = 100.0
    metabolic_cost: float = 1.0
    obs_radius: int = 3
    alive: bool = True
    age: int = 0
    gp_tree: Any = None
    gp_func: Optional[Callable[..., float]] = None

    # Stats accumulated during an episode.
    food_collected: int = 0
    kills: int = 0
    times_caught: int = 0

    def tick_energy(self) -> None:
        """Deduct metabolic cost for one time step.

        If energy drops to zero or below, the agent dies.
        """
        self.energy -= self.metabolic_cost
        if self.energy <= 0.0:
            self.energy = 0.0
            self.alive = False
        self.age += 1

    def clamp_energy(self) -> None:
        """Ensure energy does not exceed the cap."""
        if self.energy > self.max_energy:
            self.energy = self.max_energy


# ---------------------------------------------------------------------------
# Prey
# ---------------------------------------------------------------------------

class Prey(Agent):
    """A prey organism that forages for food and evades predators.

    Biological analogy:
        A herbivore in an open ecosystem.  Its GP program encodes a
        reactive foraging / evasion policy learned through evolution.
    """

    def observe(
        self,
        grid: Grid,
        predators: List[Agent],
        prey_list: List[Agent],
        config: SimConfig,
    ) -> List[float]:
        """Build the observation vector fed into the GP tree.

        Terminal order:
            FOOD_N, FOOD_S, FOOD_E, FOOD_W,
            PRED_N, PRED_S, PRED_E, PRED_W,
            PREY_N, PREY_S, PREY_E, PREY_W,
            ENERGY_LOW

        Each boolean is encoded as 1.0 (true) or 0.0 (false).

        Args:
            grid:      The world grid.
            predators: List of all predator agents (alive or dead).
            prey_list: List of all prey agents (alive or dead, including self).
            config:    Simulation config.

        Returns:
            A list of 13 floats.
        """
        r = self.obs_radius
        obs: List[float] = []

        # FOOD in four directions.
        for d in (Direction.NORTH, Direction.SOUTH,
                  Direction.EAST, Direction.WEST):
            obs.append(1.0 if grid.food_in_direction(self.y, self.x, d, r) else 0.0)

        # PREDATOR in four directions.
        for d in (Direction.NORTH, Direction.SOUTH,
                  Direction.EAST, Direction.WEST):
            obs.append(1.0 if _entity_in_direction(
                self.y, self.x, d, r, predators, grid) else 0.0)

        # PREY (other) in four directions.
        others = [p for p in prey_list if p is not self and p.alive]
        for d in (Direction.NORTH, Direction.SOUTH,
                  Direction.EAST, Direction.WEST):
            obs.append(1.0 if _entity_in_direction(
                self.y, self.x, d, r, others, grid) else 0.0)

        # ENERGY_LOW.
        low_thresh = config.energy_low_fraction * self.max_energy
        obs.append(1.0 if self.energy < low_thresh else 0.0)

        return obs

    def decide(self, observation: List[float]) -> Direction:
        """Run the GP program on *observation* and decode the action.

        Args:
            observation: Output of ``observe()``.

        Returns:
            A ``Direction``.
        """
        if self.gp_func is None:
            return Direction.STAY
        try:
            raw = float(self.gp_func(*observation))
        except Exception:
            raw = 0.0
        # Clamp to avoid extreme floats.
        raw = max(-1e6, min(1e6, raw))
        return decode_action(raw)


# ---------------------------------------------------------------------------
# Predator
# ---------------------------------------------------------------------------

class Predator(Agent):
    """A predator organism that hunts prey for energy.

    Biological analogy:
        A carnivore whose GP program encodes hunting and pursuit tactics.
    """

    def observe(
        self,
        grid: Grid,
        predators: List[Agent],
        prey_list: List[Agent],
        config: SimConfig,
    ) -> List[float]:
        """Build the observation vector fed into the GP tree.

        Terminal order:
            PREY_N, PREY_S, PREY_E, PREY_W,
            PRED_N, PRED_S, PRED_E, PRED_W,
            CLOSEST_PREY_DIR,
            ENERGY_LOW

        Args:
            grid:      The world grid.
            predators: List of all predator agents.
            prey_list: List of all prey agents.
            config:    Simulation config.

        Returns:
            A list of 10 floats.
        """
        r = self.obs_radius
        obs: List[float] = []

        alive_prey = [p for p in prey_list if p.alive]

        # PREY in four directions.
        for d in (Direction.NORTH, Direction.SOUTH,
                  Direction.EAST, Direction.WEST):
            obs.append(1.0 if _entity_in_direction(
                self.y, self.x, d, r, alive_prey, grid) else 0.0)

        # PRED (others) in four directions.
        others = [p for p in predators if p is not self and p.alive]
        for d in (Direction.NORTH, Direction.SOUTH,
                  Direction.EAST, Direction.WEST):
            obs.append(1.0 if _entity_in_direction(
                self.y, self.x, d, r, others, grid) else 0.0)

        # CLOSEST_PREY_DIR.
        closest_dir = _closest_entity_direction(
            self.y, self.x, r, alive_prey, grid)
        obs.append(float(closest_dir))

        # ENERGY_LOW.
        low_thresh = config.energy_low_fraction * self.max_energy
        obs.append(1.0 if self.energy < low_thresh else 0.0)

        return obs

    def decide(self, observation: List[float]) -> Direction:
        """Run the GP program on *observation* and decode the action.

        Args:
            observation: Output of ``observe()``.

        Returns:
            A ``Direction``.
        """
        if self.gp_func is None:
            return Direction.STAY
        try:
            raw = float(self.gp_func(*observation))
        except Exception:
            raw = 0.0
        raw = max(-1e6, min(1e6, raw))
        return decode_action(raw)


# ---------------------------------------------------------------------------
# Helpers (private)
# ---------------------------------------------------------------------------

def _entity_in_direction(
    y: int, x: int,
    direction: Direction,
    radius: int,
    entities: List[Agent],
    grid: Grid,
) -> bool:
    """Check whether any alive entity exists in *direction* within *radius*.

    A simple ray-march: check cells along the cardinal direction.

    Args:
        y, x:      Observer position.
        direction: Direction to scan.
        radius:    Scan distance.
        entities:  List of agents to look for.
        grid:      Grid (for wrapping).

    Returns:
        True if at least one alive entity is found.
    """
    if direction == Direction.STAY:
        return False
    dy, dx = DIRECTION_DELTA[direction]
    for step in range(1, radius + 1):
        cy, cx = grid.wrap(y + dy * step, x + dx * step)
        for ent in entities:
            if ent.alive and ent.y == cy and ent.x == cx:
                return True
    return False


def _closest_entity_direction(
    y: int, x: int,
    radius: int,
    entities: List[Agent],
    grid: Grid,
) -> int:
    """Direction toward the closest entity within *radius*.

    Returns:
        Integer direction code (0–4).  4 means no entity visible.
    """
    best_dist = radius + 1
    best_dir: Direction = Direction.STAY
    found = False
    for ent in entities:
        if not ent.alive:
            continue
        dist = grid.manhattan(y, x, ent.y, ent.x)
        if dist <= radius and dist < best_dist:
            best_dist = dist
            best_dir = grid.direction_of(y, x, ent.y, ent.x)
            found = True
    return int(best_dir) if found else 4
