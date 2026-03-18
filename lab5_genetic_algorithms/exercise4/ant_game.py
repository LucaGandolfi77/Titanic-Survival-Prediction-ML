"""Exercise 4 — Ant Game: Board and Environment Logic.

Implements a grid-based ant game where an ant navigates a toroidal NxN board
collecting food. The ant starts at (0,0) facing right and executes a sequence of moves.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")


# Move constants
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
MOVE_NAMES = {UP: "UP", DOWN: "DOWN", LEFT: "LEFT", RIGHT: "RIGHT"}

# Direction vectors: (row_delta, col_delta)
MOVE_DELTAS = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}


def generate_board(n=10, food_ratio=0.3, seed=42):
    """Generate a random NxN board with food cells.

    Args:
        n: board size (NxN grid).
        food_ratio: fraction of cells containing food (0.0 to 1.0).
        seed: random seed for reproducibility.

    Returns:
        2D numpy array of shape (n, n) with 1=food, 0=empty.
    """
    rng = np.random.RandomState(seed)
    board = np.zeros((n, n), dtype=int)
    n_food = max(1, int(n * n * food_ratio))
    food_positions = rng.choice(n * n, size=n_food, replace=False)
    for pos in food_positions:
        r, c = divmod(pos, n)
        board[r, c] = 1
    # Ensure start position (0,0) is not food to make it slightly harder
    board[0, 0] = 0
    return board


def run_game(board, move_sequence):
    """Execute a sequence of moves on the board and return the score.

    The ant starts at (0,0). The grid wraps around (toroidal).
    Score = number of distinct food cells visited.

    Args:
        board: 2D numpy array (NxN) with 1=food, 0=empty.
        move_sequence: list of move integers (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT).

    Returns:
        Tuple of (score, path) where path is a list of (row, col) positions.
    """
    n = board.shape[0]
    row, col = 0, 0
    path = [(row, col)]
    eaten = set()

    # Check starting position
    if board[row, col] == 1:
        eaten.add((row, col))

    for move in move_sequence:
        # Clamp move to valid range
        move = int(move) % 4
        dr, dc = MOVE_DELTAS[move]
        row = (row + dr) % n
        col = (col + dc) % n
        path.append((row, col))

        if board[row, col] == 1:
            eaten.add((row, col))

    return len(eaten), path


def decode_moves(bitstring):
    """Decode a bitstring into a sequence of moves.

    Each pair of bits encodes one move: 00=UP, 01=DOWN, 10=LEFT, 11=RIGHT.

    Args:
        bitstring: list of bits (length must be even).

    Returns:
        List of move integers.
    """
    moves = []
    for i in range(0, len(bitstring), 2):
        move = bitstring[i] * 2 + bitstring[i + 1]
        moves.append(move)
    return moves


def visualize_game(board, move_sequence, title="Ant Game", filename=None):
    """Visualize the ant game board with the ant's path.

    Args:
        board: 2D numpy array (NxN).
        move_sequence: list of move integers.
        title: plot title.
        filename: path to save the PNG.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from utils.plotting import plot_ant_game

    score, path = run_game(board, move_sequence)
    full_title = f"{title} (score={score}/{int(board.sum())} food)"
    plot_ant_game(board, path, full_title, filename)
    return score, path


if __name__ == "__main__":
    board = generate_board(n=10, food_ratio=0.3, seed=42)
    print(f"Board (10x10, food cells={board.sum()}):")
    print(board)

    # Random moves for demo
    import random
    random.seed(42)
    moves = [random.randint(0, 3) for _ in range(20)]
    score, path = run_game(board, moves)
    print(f"\nRandom moves: {[MOVE_NAMES[m] for m in moves]}")
    print(f"Score: {score}/{board.sum()}")
    print(f"Path: {path}")
