"""ant_move.py — Use a trained model to choose the ant's next move.

Works with all three model types (dt, mlp, gp).  Implements the
fallback rule: when no food is visible in the neighbourhood, pick a
random direction excluding the reverse of the last move.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from ant import Ant, DIRECTIONS, OPPOSITE


# Canonical direction order used by sklearn LabelEncoder (alphabetical)
_DIR_ORDER = ["down", "left", "right", "up"]


def ant_move(ant: Ant, model: Any, model_type: str) -> str:
    """Choose the next direction for *ant* using *model*.

    Args:
        ant: the Ant instance.
        model: trained model (sklearn classifier or GP callable).
        model_type: 'dt', 'mlp', or 'gp'.

    Returns:
        Direction string ('up', 'down', 'left', 'right').
    """
    nb = ant.get_neighborhood()

    # --- Fallback rule ---
    if np.all(nb <= 0):
        choices = [d for d in DIRECTIONS if d != OPPOSITE.get(ant.last_dir, "")]
        if not choices:
            choices = list(DIRECTIONS)
        return random.choice(choices)

    # --- Model prediction ---
    if model_type == "gp":
        return _gp_predict(model, nb)

    # sklearn models (dt / mlp)
    X = nb.reshape(1, -1).astype(float)
    pred = model.predict(X)[0]

    # Decode integer label back to direction string
    if hasattr(model, "_label_encoder"):
        direction = model._label_encoder.inverse_transform([pred])[0]
    else:
        direction = _DIR_ORDER[int(pred)] if isinstance(pred, (int, np.integer)) else str(pred)

    return direction


# -----------------------------------------------------------------------
# GP prediction helper
# -----------------------------------------------------------------------

_K = 1.0  # Threshold constant for mapping GP output → direction

def _gp_predict(gp_func: Any, nb: np.ndarray) -> str:
    """Map a GP individual's output to a direction.

    Thresholds:
        output <= -K  → up
        -K < output <= 0  → down
         0 < output <= K  → right
        output > K  → left

    Args:
        gp_func: compiled GP individual (callable taking *args of floats).
        nb: flattened neighbourhood array.

    Returns:
        Direction string.
    """
    args = tuple(float(x) for x in nb)
    try:
        out = float(gp_func(*args))
    except Exception:
        return random.choice(list(DIRECTIONS))

    if out <= -_K:
        return "up"
    elif out <= 0:
        return "down"
    elif out <= _K:
        return "right"
    else:
        return "left"


# -----------------------------------------------------------------------
# Play a full game with a model
# -----------------------------------------------------------------------

def play_game(
    n: int,
    m: int,
    model: Any,
    model_type: str,
    seed: int = 42,
) -> int:
    """Play one game using a trained model and return the score.

    Args:
        n: board size.
        m: neighbourhood half-width.
        model: trained model.
        model_type: 'dt', 'mlp', or 'gp'.
        seed: board seed.

    Returns:
        Final score.
    """
    from board import Board

    board = Board(n, seed=seed)
    rng = np.random.RandomState(seed + 1000)
    empty = [(r, c) for r in range(n) for c in range(n) if board.grid[r, c] == 0]
    if not empty:
        empty = [(r, c) for r in range(n) for c in range(n)]
    start = empty[rng.randint(len(empty))]
    ant = Ant(board, start[0], start[1], m=m)

    random.seed(seed + 2000)

    while ant.has_moves_left():
        direction = ant_move(ant, model, model_type)
        ant.move(direction)

    return ant.score
