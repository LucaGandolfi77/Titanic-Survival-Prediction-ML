"""ant.py — Ant agent for the Ant Trail Problem.

The ant lives on a Board, tracks its position, score, and alive status,
and can move in four cardinal directions.  Moving outside the grid kills
the ant and subtracts (N+2) from its score.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from board import Board

# Canonical direction names and their (row, col) deltas.
DIRECTIONS: dict[str, tuple[int, int]] = {
    "up":    (-1,  0),
    "down":  ( 1,  0),
    "left":  ( 0, -1),
    "right": ( 0,  1),
}

# Reverse direction mapping (used by the fallback rule).
OPPOSITE: dict[str, str] = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}


class Ant:
    """Agent that navigates the board collecting food.

    Attributes:
        board: reference to the Board instance.
        row, col: current grid position.
        score: accumulated score.
        alive: False once the ant exits the grid.
        last_dir: the most recent direction taken (or None at start).
        moves_taken: number of moves performed so far.
        max_moves: maximum allowed moves (2*N).
        m: neighbourhood half-width used for observation.
    """

    def __init__(self, board: Board, start_row: int, start_col: int, m: int = 1) -> None:
        """Place the ant on the board at (start_row, start_col).

        The starting cell's value is set to -1 immediately.

        Args:
            board: the game board.
            start_row: starting row index.
            start_col: starting column index.
            m: neighbourhood half-width.
        """
        self.board: Board = board
        self.n: int = board.n
        self.row: int = start_row
        self.col: int = start_col
        self.score: int = 0
        self.alive: bool = True
        self.last_dir: Optional[str] = None
        self.moves_taken: int = 0
        self.max_moves: int = 2 * self.n
        self.m: int = m

        # Mark starting cell
        self.board.grid[self.row, self.col] = -1

    # ------------------------------------------------------------------
    # Core actions
    # ------------------------------------------------------------------

    def move(self, direction: str) -> int:
        """Move the ant one step in the given direction.

        - If the destination is outside the grid the ant dies and loses
          (N+2) points.
        - Otherwise the ant earns the destination cell's value, then the
          cell is decremented by 1.

        Args:
            direction: one of 'up', 'down', 'left', 'right'.

        Returns:
            The reward collected on this step.
        """
        direction = direction.lower()
        if direction not in DIRECTIONS:
            raise ValueError(f"Invalid direction '{direction}'. Use: {list(DIRECTIONS)}")

        if not self.alive:
            return 0

        dr, dc = DIRECTIONS[direction]
        new_r = self.row + dr
        new_c = self.col + dc

        if not self.board.in_bounds(new_r, new_c):
            # Ant exits the grid
            self.alive = False
            penalty = -(self.n + 2)
            self.score += penalty
            self.last_dir = direction
            self.moves_taken += 1
            return penalty

        # Collect the cell value, then decrement
        reward = int(self.board.grid[new_r, new_c])
        self.score += reward
        self.board.grid[new_r, new_c] -= 1

        self.row = new_r
        self.col = new_c
        self.last_dir = direction
        self.moves_taken += 1
        return reward

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def get_neighborhood(self, m: Optional[int] = None) -> np.ndarray:
        """Return the (2m+1)x(2m+1) neighbourhood centered on the ant.

        Out-of-bounds cells are filled with -(N+2).

        Args:
            m: half-width override (defaults to self.m).

        Returns:
            Flattened int array of length (2m+1)^2.
        """
        if m is None:
            m = self.m
        size = 2 * m + 1
        nb = np.full((size, size), -(self.n + 2), dtype=int)
        for dr in range(-m, m + 1):
            for dc in range(-m, m + 1):
                r, c = self.row + dr, self.col + dc
                if self.board.in_bounds(r, c):
                    nb[dr + m, dc + m] = self.board.grid[r, c]
        return nb.flatten()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def is_alive(self) -> bool:
        """Return True if the ant is still on the board."""
        return self.alive

    def has_moves_left(self) -> bool:
        """Return True if the ant can still take a move."""
        return self.alive and self.moves_taken < self.max_moves

    def __repr__(self) -> str:
        return (f"Ant(pos=({self.row},{self.col}), score={self.score}, "
                f"alive={self.alive}, moves={self.moves_taken}/{self.max_moves})")
