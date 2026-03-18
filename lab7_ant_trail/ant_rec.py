"""ant_rec.py — Interactive recording of ant games.

Provides two modes:
  1. **Interactive** (curses-based): the user steers the ant with w/a/s/d keys.
  2. **Automated**: a simple heuristic plays the game (for batch data collection
     without a terminal).

Each step writes a CSV row:  f0, f1, …, fk, direction
where f0…fk are the flattened (2m+1)² neighbourhood values.
"""

from __future__ import annotations

import csv
import os
import random
import sys
from typing import Optional

import numpy as np

from board import Board
from ant import Ant, DIRECTIONS, OPPOSITE


# -----------------------------------------------------------------------
# Heuristic auto-player (greedy + random fallback)
# -----------------------------------------------------------------------

def _heuristic_direction(ant: Ant) -> str:
    """Pick a greedy direction: highest neighbour value, with random
    tie-breaking.  If no positive neighbour exists use the fallback rule
    (random excluding reverse).

    Args:
        ant: the current Ant instance.

    Returns:
        Direction string.
    """
    best_val = -999
    best_dirs: list[str] = []

    for d, (dr, dc) in DIRECTIONS.items():
        nr, nc = ant.row + dr, ant.col + dc
        val = ant.board.get_value(nr, nc)
        if val > best_val:
            best_val = val
            best_dirs = [d]
        elif val == best_val:
            best_dirs.append(d)

    # If best reachable value is <= 0 → fallback (random, exclude reverse)
    if best_val <= 0:
        choices = [d for d in DIRECTIONS if d != OPPOSITE.get(ant.last_dir, "")]
        if not choices:
            choices = list(DIRECTIONS)
        return random.choice(choices)

    # Among best, prefer directions that stay in-bounds
    safe = [d for d in best_dirs
            if ant.board.in_bounds(ant.row + DIRECTIONS[d][0],
                                   ant.col + DIRECTIONS[d][1])]
    if safe:
        return random.choice(safe)
    return random.choice(best_dirs)


# -----------------------------------------------------------------------
# Core recording function
# -----------------------------------------------------------------------

def ant_rec(
    n: int,
    m: int,
    filename: str,
    seed: int = 42,
    interactive: bool = False,
    append: bool = True,
) -> int:
    """Play one game and record state-action pairs to a CSV file.

    Args:
        n: board size.
        m: neighbourhood half-width.
        filename: destination CSV path.
        seed: board seed.
        interactive: if True use curses for keyboard input; otherwise
                     use the greedy heuristic for automated data collection.
        append: if True, append rows to an existing file.

    Returns:
        The ant's total score at the end of the game.
    """
    board = Board(n, seed=seed)

    # Place ant on a random empty cell
    rng = np.random.RandomState(seed + 1000)
    empty_cells = [(r, c) for r in range(n) for c in range(n)
                   if board.grid[r, c] == 0]
    if not empty_cells:
        # Fallback: pick any cell
        empty_cells = [(r, c) for r in range(n) for c in range(n)]
    start = empty_cells[rng.randint(len(empty_cells))]
    ant = Ant(board, start[0], start[1], m=m)

    random.seed(seed)

    mode = "a" if append else "w"
    file_existed = os.path.exists(filename) and append

    with open(filename, mode, newline="") as f:
        writer = csv.writer(f)

        if interactive:
            _interactive_loop(ant, writer)
        else:
            _auto_loop(ant, writer)

    return ant.score


# -----------------------------------------------------------------------
# Automated loop (heuristic player)
# -----------------------------------------------------------------------

def _auto_loop(ant: Ant, writer: csv.writer) -> None:
    """Play the game with the heuristic and record every step."""
    while ant.has_moves_left():
        nb = ant.get_neighborhood()
        direction = _heuristic_direction(ant)
        row = list(nb) + [direction]
        writer.writerow(row)
        ant.move(direction)


# -----------------------------------------------------------------------
# Interactive (curses) loop
# -----------------------------------------------------------------------

def _interactive_loop(ant: Ant, writer: csv.writer) -> None:
    """Play the game interactively using the curses library."""
    try:
        import curses
    except ImportError:
        print("curses not available — falling back to automated mode.")
        _auto_loop(ant, writer)
        return

    def _run(stdscr: "curses.window") -> None:
        curses.curs_set(0)
        key_map = {
            ord("w"): "up", curses.KEY_UP: "up",
            ord("s"): "down", curses.KEY_DOWN: "down",
            ord("a"): "left", curses.KEY_LEFT: "left",
            ord("d"): "right", curses.KEY_RIGHT: "right",
        }

        while ant.has_moves_left():
            stdscr.clear()
            stdscr.addstr(0, 0, f"Score: {ant.score}  "
                          f"Moves: {ant.moves_taken}/{ant.max_moves}  "
                          f"Alive: {ant.alive}")
            board_str = ant.board.show_board(ant_pos=(ant.row, ant.col))
            for i, line in enumerate(board_str.split("\n")):
                stdscr.addstr(i + 2, 0, line)
            stdscr.addstr(ant.board.n + 4, 0,
                          "Move: w/a/s/d  |  q = quit")
            stdscr.refresh()

            key = stdscr.getch()
            if key == ord("q"):
                break
            direction = key_map.get(key)
            if direction is None:
                continue

            nb = ant.get_neighborhood()
            row = list(nb) + [direction]
            writer.writerow(row)
            ant.move(direction)

    curses.wrapper(_run)


# -----------------------------------------------------------------------
# Convenience: collect multiple games
# -----------------------------------------------------------------------

def collect_games(
    n: int,
    m: int,
    num_games: int,
    filename: str,
    base_seed: int = 42,
    interactive: bool = False,
) -> list[int]:
    """Play *num_games* games and aggregate recordings into one CSV.

    Args:
        n: board size.
        m: neighbourhood half-width.
        num_games: number of games to play.
        filename: destination CSV path.
        base_seed: first game uses this seed; subsequent games increment.
        interactive: if True, use curses for each game.

    Returns:
        List of scores (one per game).
    """
    scores: list[int] = []
    for i in range(num_games):
        score = ant_rec(n, m, filename, seed=base_seed + i,
                        interactive=interactive, append=(i > 0))
        scores.append(score)
    return scores


# -----------------------------------------------------------------------
# Main (standalone usage)
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Record ant games")
    parser.add_argument("-n", type=int, default=10, help="Board size")
    parser.add_argument("-m", type=int, default=1, help="Neighbourhood half-width")
    parser.add_argument("--games", type=int, default=1, help="Number of games")
    parser.add_argument("--file", type=str, default="data/demo.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.file) or ".", exist_ok=True)
    scores = collect_games(args.n, args.m, args.games, args.file,
                           base_seed=args.seed, interactive=args.interactive)
    print(f"Scores: {scores}  |  Mean: {np.mean(scores):.2f}")
