"""
visualization.py — Matplotlib-based plots and animations.

Functions:
    plot_fitness_curves     — Prey vs. predator fitness over generations.
    plot_behavior_emergence — Bar / timeline of detected behaviours.
    animate_episode         — GIF of a single episode on the grid.
    plot_radius_comparison  — Side-by-side fitness curves for different presets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from coevolution import CoevolutionResult, GenerationStats
from fitness import EpisodeResult


# ---------------------------------------------------------------------------
# 1. Fitness curves
# ---------------------------------------------------------------------------

def plot_fitness_curves(
    result: CoevolutionResult,
    output_path: str | Path = "fitness_curves.png",
) -> Path:
    """Plot mean fitness of prey and predator populations over generations.

    Args:
        result:      CoevolutionResult containing per-generation stats.
        output_path: Where to save the figure.

    Returns:
        Path to the saved image.
    """
    gens = list(range(len(result.generation_stats)))
    prey_means = [s.prey_fitness_mean for s in result.generation_stats]
    pred_means = [s.pred_fitness_mean for s in result.generation_stats]
    prey_stds = [s.prey_fitness_std for s in result.generation_stats]
    pred_stds = [s.pred_fitness_std for s in result.generation_stats]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gens, prey_means, label="Prey (mean)", color="tab:green")
    ax.fill_between(gens,
                    [m - s for m, s in zip(prey_means, prey_stds)],
                    [m + s for m, s in zip(prey_means, prey_stds)],
                    alpha=0.2, color="tab:green")
    ax.plot(gens, pred_means, label="Predator (mean)", color="tab:red")
    ax.fill_between(gens,
                    [m - s for m, s in zip(pred_means, pred_stds)],
                    [m + s for m, s in zip(pred_means, pred_stds)],
                    alpha=0.2, color="tab:red")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Co-evolutionary Fitness Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 2. Behaviour emergence timeline
# ---------------------------------------------------------------------------

def plot_behavior_emergence(
    result: CoevolutionResult,
    output_path: str | Path = "behavior_emergence.png",
) -> Path:
    """Plot which behaviours appear in each generation.

    Args:
        result:      CoevolutionResult.
        output_path: Image save path.

    Returns:
        Path to the saved image.
    """
    all_labels: set[str] = set()
    for s in result.generation_stats:
        all_labels.update(s.behaviors)

    if not all_labels:
        # Nothing to plot — create a minimal figure.
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No emergent behaviours detected",
                ha="center", va="center", transform=ax.transAxes)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    labels_sorted = sorted(all_labels)
    gens = list(range(len(result.generation_stats)))

    matrix = np.zeros((len(labels_sorted), len(gens)), dtype=float)
    for gi, s in enumerate(result.generation_stats):
        for li, lab in enumerate(labels_sorted):
            if lab in s.behaviors:
                matrix[li, gi] = 1.0

    fig, ax = plt.subplots(figsize=(max(8, len(gens) * 0.15), max(3, len(labels_sorted) * 0.6)))
    ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(len(labels_sorted)))
    ax.set_yticklabels(labels_sorted)
    ax.set_xlabel("Generation")
    ax.set_title("Emergent Behaviour Timeline")
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 3. Episode animation (GIF)
# ---------------------------------------------------------------------------

def animate_episode(
    episode: EpisodeResult,
    grid_size: int,
    output_path: str | Path = "episode.gif",
    fps: int = 5,
) -> Path:
    """Create a GIF animation of a single episode on the grid.

    Symbols: green ● = prey, red ▲ = predator, yellow ■ = food.

    Args:
        episode:     Recorded episode with step-by-step snapshots.
        grid_size:   N (side length of the toroidal grid).
        output_path: Where to save the GIF.
        fps:         Frames per second.

    Returns:
        Path to the saved GIF.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    def _draw(step_idx: int) -> None:
        ax.clear()
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_aspect("equal")
        ax.set_title(f"Step {step_idx}")
        ax.grid(True, alpha=0.2)

        record = episode.steps[step_idx]

        # Food positions — approximate via food_count (no per-cell data in snapshot).
        # We plot prey and predators only.
        prey_x = [s.x for s in record.prey_snapshots if s.alive]
        prey_y = [s.y for s in record.prey_snapshots if s.alive]
        pred_x = [s.x for s in record.predator_snapshots if s.alive]
        pred_y = [s.y for s in record.predator_snapshots if s.alive]

        ax.scatter(prey_x, prey_y, c="tab:green", marker="o", s=80,
                   edgecolors="black", linewidths=0.5, label="Prey", zorder=3)
        ax.scatter(pred_x, pred_y, c="tab:red", marker="^", s=100,
                   edgecolors="black", linewidths=0.5, label="Predator", zorder=4)
        ax.legend(loc="upper right", fontsize=8)

    anim = animation.FuncAnimation(
        fig, _draw, frames=len(episode.steps), interval=1000 // fps)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer="pillow", fps=fps)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 4. Radius comparison
# ---------------------------------------------------------------------------

def plot_radius_comparison(
    results: Dict[str, CoevolutionResult],
    output_path: str | Path = "radius_comparison.png",
) -> Path:
    """Side-by-side fitness curves for multiple presets / configurations.

    Args:
        results:     Mapping of preset name → CoevolutionResult.
        output_path: Image save path.

    Returns:
        Path to the saved image.
    """
    n_configs = len(results)
    fig, axes = plt.subplots(1, max(n_configs, 1),
                             figsize=(5 * max(n_configs, 1), 4),
                             squeeze=False)

    for idx, (name, res) in enumerate(results.items()):
        ax = axes[0][idx]
        gens = list(range(len(res.generation_stats)))
        prey_m = [s.prey_fitness_mean for s in res.generation_stats]
        pred_m = [s.pred_fitness_mean for s in res.generation_stats]
        ax.plot(gens, prey_m, label="Prey", color="tab:green")
        ax.plot(gens, pred_m, label="Predator", color="tab:red")
        ax.set_title(name)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Observation Radius Comparison", fontsize=13)
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
