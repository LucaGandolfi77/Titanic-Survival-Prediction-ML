"""
world.py — Toroidal grid world for the predator-prey co-evolutionary simulation.

Biological analogy:
    The ``Grid`` represents a bounded yet wrap-around habitat (think of an island
    chain or a continuous savanna).  Cells host static *food* resources that
    regenerate stochastically, modelling seasonal or patchy vegetation.  Agents
    perceive only a local neighbourhood, analogous to a real animal's sensory
    range limited by terrain, foliage, or physiology.
"""

from __future__ import annotations

from enum import IntEnum, unique
from typing import List, Optional, Tuple

import numpy as np

from config import SimConfig


# ---------------------------------------------------------------------------
# Cell types
# ---------------------------------------------------------------------------

@unique
class Cell(IntEnum):
    """Possible contents of a single grid cell.

    EMPTY and FOOD are "terrain"; PREY and PREDATOR layers are tracked
    separately via agent lists, but the enum is handy for rendering and
    neighbourhood queries.
    """
    EMPTY = 0
    FOOD = 1
    PREY = 2
    PREDATOR = 3


# ---------------------------------------------------------------------------
# Direction helpers
# ---------------------------------------------------------------------------

@unique
class Direction(IntEnum):
    """Cardinal movement directions plus *stay*.

    Numerical encoding matches the GP output-decoding table in the spec.
    """
    NORTH = 0
    WEST = 1
    STAY = 2
    EAST = 3
    SOUTH = 4


# (dy, dx) deltas — row 0 is topmost, so NORTH is dy=-1.
DIRECTION_DELTA: dict[Direction, Tuple[int, int]] = {
    Direction.NORTH: (-1, 0),
    Direction.SOUTH: (1, 0),
    Direction.EAST: (0, 1),
    Direction.WEST: (0, -1),
    Direction.STAY: (0, 0),
}


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

class Grid:
    """NxN toroidal grid that hosts food and provides neighbourhood queries.

    The grid stores **food** as a boolean matrix.  Agent positions are tracked
    in the ``agents.py`` layer; the grid is queried only for terrain (food)
    and for computing local observations.

    Args:
        config: Simulation configuration.
        rng:    Seeded numpy random generator.
    """

    def __init__(self, config: SimConfig, rng: np.random.Generator) -> None:
        self.n: int = config.grid_size
        self.config: SimConfig = config
        self._rng: np.random.Generator = rng

        # Boolean food layer.
        self.food: np.ndarray = rng.random((self.n, self.n)) < config.initial_food_density

    # -- Coordinate helpers --------------------------------------------------

    def wrap(self, y: int, x: int) -> Tuple[int, int]:
        """Wrap coordinates on the torus.

        Args:
            y: Row index (may be negative or >= N).
            x: Column index.

        Returns:
            (y % N, x % N) with Python's modulo semantics.
        """
        return y % self.n, x % self.n

    def manhattan(self, y1: int, x1: int, y2: int, x2: int) -> int:
        """Toroidal Manhattan distance between two cells.

        Args:
            y1, x1: First cell.
            y2, x2: Second cell.

        Returns:
            Minimum Manhattan distance on the torus.
        """
        dy = abs(y1 - y2)
        dx = abs(x1 - x2)
        dy = min(dy, self.n - dy)
        dx = min(dx, self.n - dx)
        return dy + dx

    def direction_of(self, y_from: int, x_from: int,
                     y_to: int, x_to: int) -> Direction:
        """Dominant cardinal direction from one cell toward another.

        Ties broken: vertical preferred over horizontal, North over South.

        Args:
            y_from, x_from: Source cell.
            y_to, x_to:     Target cell.

        Returns:
            The ``Direction`` that moves closer to the target on the torus.
        """
        dy = (y_to - y_from) % self.n
        if dy > self.n // 2:
            dy -= self.n
        dx = (x_to - x_from) % self.n
        if dx > self.n // 2:
            dx -= self.n

        if dy == 0 and dx == 0:
            return Direction.STAY

        if abs(dy) >= abs(dx):
            return Direction.NORTH if dy < 0 else Direction.SOUTH
        return Direction.WEST if dx < 0 else Direction.EAST

    # -- Neighbourhood -------------------------------------------------------

    def get_neighborhood_coords(self, y: int, x: int,
                                radius: int) -> List[Tuple[int, int]]:
        """All cells within Manhattan distance *radius* of (y, x).

        Args:
            y, x:   Centre cell.
            radius: Observation radius (Manhattan).

        Returns:
            List of (row, col) tuples (wrapped).
        """
        coords: List[Tuple[int, int]] = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if abs(dy) + abs(dx) <= radius:
                    coords.append(self.wrap(y + dy, x + dx))
        return coords

    def has_food(self, y: int, x: int) -> bool:
        """Check whether a cell contains food.

        Args:
            y, x: Cell coordinates.

        Returns:
            True if food is present.
        """
        return bool(self.food[y % self.n, x % self.n])

    def consume_food(self, y: int, x: int) -> bool:
        """Remove food from a cell (prey eats it).

        Args:
            y, x: Cell coordinates.

        Returns:
            True if food was present and consumed; False otherwise.
        """
        wy, wx = self.wrap(y, x)
        if self.food[wy, wx]:
            self.food[wy, wx] = False
            return True
        return False

    def spawn_food(self) -> None:
        """Stochastic food respawn across all empty (no-food) cells.

        Each cell without food independently gains food with probability
        ``config.food_respawn_prob``.
        """
        mask = ~self.food
        spawn = self._rng.random((self.n, self.n)) < self.config.food_respawn_prob
        self.food |= (mask & spawn)

    # -- Movement helpers ----------------------------------------------------

    def move(self, y: int, x: int, direction: Direction) -> Tuple[int, int]:
        """Compute new coordinates after moving in *direction* on the torus.

        Args:
            y, x:      Current position.
            direction: Movement direction.

        Returns:
            (new_y, new_x) after toroidal wrapping.
        """
        dy, dx = DIRECTION_DELTA[direction]
        return self.wrap(y + dy, x + dx)

    # -- Directional food / entity queries -----------------------------------

    def food_in_direction(self, y: int, x: int, direction: Direction,
                          radius: int) -> bool:
        """Is there food in the given cardinal direction within *radius*?

        Scans cells along a cone in *direction*.

        Args:
            y, x:      Observer position.
            direction: Cardinal direction to scan.
            radius:    Observation radius.

        Returns:
            True if at least one food cell is found.
        """
        if direction == Direction.STAY:
            return self.has_food(y, x)
        dy, dx = DIRECTION_DELTA[direction]
        for step in range(1, radius + 1):
            ny, nx = self.wrap(y + dy * step, x + dx * step)
            if self.has_food(ny, nx):
                return True
        return False

    def count_food(self) -> int:
        """Total number of food cells on the grid.

        Returns:
            Integer count.
        """
        return int(np.sum(self.food))
