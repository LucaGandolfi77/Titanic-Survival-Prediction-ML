"""Shared plotting utilities for all exercises using matplotlib and DEAP logbooks."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(logbooks, labels, title, filename, figsize=(10, 6)):
    """Plot best and average fitness vs generation for one or more logbooks.

    Args:
        logbooks: list of DEAP Logbook objects (or list of dicts with 'gen', 'min'/'max', 'avg').
        labels: list of string labels for each logbook.
        title: plot title.
        filename: path to save the PNG.
        figsize: figure size tuple.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for logbook, label in zip(logbooks, labels):
        gen = logbook.select("gen")
        # Use 'min' for minimization, 'max' for maximization — include both
        if "min" in logbook[0]:
            ax1.plot(gen, logbook.select("min"), label=f"{label} (best/min)")
        if "max" in logbook[0]:
            ax1.plot(gen, logbook.select("max"), label=f"{label} (best/max)", linestyle="--")
        avg = logbook.select("avg")
        ax2.plot(gen, avg, label=label)

    ax1.set_title(f"{title} — Best Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.legend()
    ax1.grid(True)

    ax2.set_title(f"{title} — Average Fitness")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Fitness")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  [plot] Saved: {filename}")


def plot_convergence_single(logbook, title, filename, best_key="min", figsize=(8, 5)):
    """Plot best and average fitness for a single logbook.

    Args:
        logbook: a DEAP Logbook.
        title: plot title.
        filename: path to save the PNG.
        best_key: 'min' for minimization, 'max' for maximization.
        figsize: figure size tuple.
    """
    fig, ax = plt.subplots(figsize=figsize)
    gen = logbook.select("gen")
    best = logbook.select(best_key)
    avg = logbook.select("avg")

    ax.plot(gen, best, label=f"Best ({best_key})", linewidth=2)
    ax.plot(gen, avg, label="Average", linewidth=1, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  [plot] Saved: {filename}")


def plot_scaling(sizes, avg_nf, title, filename, figsize=(8, 5)):
    """Plot average Nf vs pattern size.

    Args:
        sizes: list of pattern sizes (number of bits).
        avg_nf: list of average Nf values.
        title: plot title.
        filename: path to save the PNG.
        figsize: figure size tuple.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(sizes, avg_nf, "o-", linewidth=2, markersize=8)
    ax.set_title(title)
    ax.set_xlabel("Pattern Size (bits)")
    ax.set_ylabel("Average Nf (fitness evaluations)")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  [plot] Saved: {filename}")


def plot_heatmap(pop_sizes, gen_counts, nf_matrix, title, filename, figsize=(8, 6)):
    """Plot a heatmap of average Nf for (pop_size, n_gen) combinations.

    Args:
        pop_sizes: list of population sizes (columns).
        gen_counts: list of generation counts (rows).
        nf_matrix: 2D numpy array of shape (len(gen_counts), len(pop_sizes)).
        title: plot title.
        filename: path to save the PNG.
        figsize: figure size tuple.
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(nf_matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pop_sizes)))
    ax.set_xticklabels(pop_sizes)
    ax.set_yticks(range(len(gen_counts)))
    ax.set_yticklabels(gen_counts)
    ax.set_xlabel("Population Size")
    ax.set_ylabel("Generations")
    ax.set_title(title)

    # Annotate cells
    for i in range(len(gen_counts)):
        for j in range(len(pop_sizes)):
            val = nf_matrix[i, j]
            text = f"{val:.0f}" if not np.isinf(val) else "∞"
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                    color="white" if val > nf_matrix[np.isfinite(nf_matrix)].mean() else "black")

    fig.colorbar(im, ax=ax, label="Avg Nf")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  [plot] Saved: {filename}")


def plot_chessboard(solution, n, title, filename, figsize=None):
    """Visualize an N-Queens solution as a chessboard.

    Args:
        solution: list/array of length N where solution[i] = column of queen in row i.
        n: board size.
        title: plot title.
        filename: path to save the PNG.
        figsize: figure size tuple.
    """
    if figsize is None:
        figsize = (max(4, n * 0.5), max(4, n * 0.5))
    fig, ax = plt.subplots(figsize=figsize)

    # Draw checkerboard
    for i in range(n):
        for j in range(n):
            color = "#F0D9B5" if (i + j) % 2 == 0 else "#B58863"
            ax.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1, facecolor=color))

    # Place queens
    for i, col in enumerate(solution):
        ax.text(col + 0.5, n - 1 - i + 0.5, "♛", fontsize=max(6, 300 // n),
                ha="center", va="center", color="red")

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  [plot] Saved: {filename}")


def plot_grid_pattern(individual, rows, cols, title, filename=None, figsize=(6, 5)):
    """Visualize a binary individual as a rows×cols grid.

    Args:
        individual: flat array/list of 0/1 values.
        rows: number of rows.
        cols: number of columns.
        title: plot title.
        filename: path to save the PNG (None to just show).
        figsize: figure size tuple.
    """
    grid = np.array(individual).reshape(rows, cols)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(grid, cmap="Greys", interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
        print(f"  [plot] Saved: {filename}")
    plt.close(fig)


def plot_ant_game(board, path, title, filename, figsize=(8, 8)):
    """Visualize the ant game board with the ant's path.

    Args:
        board: 2D numpy array (NxN) with 1=food, 0=empty.
        path: list of (row, col) positions visited by the ant.
        title: plot title.
        filename: path to save the PNG.
        figsize: figure size tuple.
    """
    n = board.shape[0]
    fig, ax = plt.subplots(figsize=figsize)

    # Draw board
    display = np.zeros((n, n, 3))
    for i in range(n):
        for j in range(n):
            if board[i, j] == 1:
                display[i, j] = [0.6, 0.9, 0.6]  # light green for food
            else:
                display[i, j] = [1.0, 1.0, 1.0]  # white for empty

    # Mark visited cells
    visited = set()
    for r, c in path:
        visited.add((r, c))
        if board[r, c] == 1:
            display[r, c] = [0.2, 0.7, 0.2]  # dark green = food eaten
        else:
            display[r, c] = [0.8, 0.8, 1.0]  # light blue = visited empty

    ax.imshow(display, interpolation="nearest")

    # Draw path arrows
    for k in range(len(path) - 1):
        r1, c1 = path[k]
        r2, c2 = path[k + 1]
        # Only draw arrow if not wrapping
        if abs(r2 - r1) <= 1 and abs(c2 - c1) <= 1:
            ax.annotate("", xy=(c2, r2), xytext=(c1, r1),
                        arrowprops=dict(arrowstyle="->", color="blue", lw=1.5))

    # Mark start
    ax.plot(path[0][1], path[0][0], "rs", markersize=10, label="Start")
    # Mark end
    ax.plot(path[-1][1], path[-1][0], "b^", markersize=10, label="End")

    ax.set_title(title)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  [plot] Saved: {filename}")
