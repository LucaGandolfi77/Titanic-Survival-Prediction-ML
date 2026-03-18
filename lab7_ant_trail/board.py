"""board.py — NxN grid board for the Ant Trail Problem.

The board is an NxN grid where N cells are randomly initialised to +1 (food)
and every other cell starts at 0 (empty).  The ant's current cell is marked
with -1 upon placement.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np


class Board:
    """NxN grid environment for the Ant Trail game.

    Attributes:
        n: board dimension.
        grid: 2-D numpy int array of shape (n, n).
        seed: random seed used to place food.
    """

    def __init__(self, n: int, seed: int = 42) -> None:
        """Create a new board with *n* food cells placed randomly.

        Args:
            n: board size (and also the number of food cells).
            seed: controls food placement for reproducibility.
        """
        self.n: int = n
        self.seed: int = seed
        self.grid: np.ndarray = np.zeros((n, n), dtype=int)
        self._place_food(seed)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _place_food(self, seed: int) -> None:
        """Randomly place exactly *n* food cells (value +1)."""
        rng = np.random.RandomState(seed)
        total_cells = self.n * self.n
        positions = rng.choice(total_cells, size=self.n, replace=False)
        for pos in positions:
            r, c = divmod(pos, self.n)
            self.grid[r, c] = 1

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the board to its initial state (optionally with a new seed).

        Args:
            seed: if given, overrides the stored seed for food placement.
        """
        if seed is not None:
            self.seed = seed
        self.grid = np.zeros((self.n, self.n), dtype=int)
        self._place_food(self.seed)

    def copy(self) -> "Board":
        """Return a deep copy of the board."""
        new = Board.__new__(Board)
        new.n = self.n
        new.seed = self.seed
        new.grid = self.grid.copy()
        return new

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def in_bounds(self, r: int, c: int) -> bool:
        """Return True if (r, c) is inside the grid."""
        return 0 <= r < self.n and 0 <= c < self.n

    def get_value(self, r: int, c: int) -> int:
        """Return the cell value, or -(n+2) for out-of-bounds (virtual wall)."""
        if self.in_bounds(r, c):
            return int(self.grid[r, c])
        return -(self.n + 2)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def show_board(self, ant_pos: Optional[tuple[int, int]] = None) -> str:
        """Render the board as a human-readable string.

        Args:
            ant_pos: optional (row, col) of the ant — shown as 'A'.

        Returns:
            Multi-line string representation of the board.
        """
        lines: list[str] = []
        header = "   " + " ".join(f"{c:>3}" for c in range(self.n))
        lines.append(header)
        lines.append("   " + "----" * self.n)
        for r in range(self.n):
            row_str = f"{r:>2}|"
            for c in range(self.n):
                if ant_pos is not None and (r, c) == tuple(ant_pos):
                    row_str += "  A"
                else:
                    row_str += f"{self.grid[r, c]:>3}"
            lines.append(row_str)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Board(n={self.n}, seed={self.seed})"
