"""Plotting utilities for RL experiments.

Generates publication-quality training curves, comparison plots,
heatmaps, and thermal-control–specific visualisations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

# Matplotlib style
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


def _smooth(values: Sequence[float], window: int = 10) -> np.ndarray:
    """Simple moving average."""
    arr = np.array(values, dtype=np.float64)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


# ==================================================================
# Training curves
# ==================================================================
def plot_training_curves(
    rewards: list[float],
    losses: list[float] | None = None,
    epsilons: list[float] | None = None,
    title: str = "DQN Training Progress",
    save_path: Path | str | None = None,
    window: int = 10,
) -> plt.Figure:
    """Multi-panel training curve plot.

    Panel 1: episode rewards (raw + smoothed)
    Panel 2: loss
    Panel 3: epsilon schedule (if provided)
    """
    n_panels = 1 + (losses is not None) + (epsilons is not None)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 4 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    idx = 0

    # Rewards
    ax = axes[idx]
    ax.plot(rewards, alpha=0.3, color="steelblue", label="Raw")
    smoothed = _smooth(rewards, window)
    ax.plot(
        range(window - 1, window - 1 + len(smoothed)),
        smoothed,
        color="steelblue",
        linewidth=2,
        label=f"Smooth (w={window})",
    )
    ax.set_ylabel("Episode Reward")
    ax.legend()
    ax.set_title(title)
    idx += 1

    # Loss
    if losses is not None:
        ax = axes[idx]
        ax.plot(losses, alpha=0.4, color="tomato", label="Loss")
        if len(losses) > window:
            s_loss = _smooth(losses, window)
            ax.plot(
                range(window - 1, window - 1 + len(s_loss)),
                s_loss,
                color="tomato",
                linewidth=2,
            )
        ax.set_ylabel("Loss")
        ax.legend()
        idx += 1

    # Epsilon
    if epsilons is not None:
        ax = axes[idx]
        ax.plot(epsilons, color="green", linewidth=1.5)
        ax.set_ylabel("Epsilon")
        ax.set_xlabel("Episode")

    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ==================================================================
# Comparison plots
# ==================================================================
def plot_comparison(
    results: dict[str, list[float]],
    title: str = "DQN Variants — CartPole-v1",
    save_path: Path | str | None = None,
    window: int = 20,
) -> plt.Figure:
    """Overlay smoothed reward curves for multiple agents.

    Parameters
    ----------
    results : dict
        ``{label: episode_rewards}``.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    colours = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for i, (label, rewards) in enumerate(results.items()):
        c = colours[i % len(colours)]
        ax.plot(rewards, alpha=0.15, color=c)
        sm = _smooth(rewards, window)
        ax.plot(
            range(window - 1, window - 1 + len(sm)),
            sm,
            color=c,
            linewidth=2,
            label=label,
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ==================================================================
# Thermal control
# ==================================================================
def plot_thermal_trajectory(
    trajectory: dict[str, list],
    config: dict[str, Any],
    title: str = "Thermal Control — DQN Agent",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """4-panel plot for a single thermal-control episode.

    Panels: temperature, fan level, heat generation, cumulative reward.
    """
    infos = trajectory["infos"]
    temps = [info.get("temperature", 0) for info in infos]
    fans = [info.get("fan_level", 0) for info in infos]
    heat = [info.get("heat_generation", 0) for info in infos]
    rewards = trajectory.get("rewards", [])
    cum_reward = np.cumsum(rewards) if rewards else []

    thermal_cfg = config.get("environment", {}).get("thermal", {})
    target = thermal_cfg.get("target_temp", 55.0)
    tol = thermal_cfg.get("temp_tolerance", 5.0)
    critical = thermal_cfg.get("critical_temp", 85.0)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Temperature
    ax = axes[0]
    ax.plot(temps, color="tomato", linewidth=1.5, label="Temperature")
    ax.axhline(target, color="green", linestyle="--", label=f"Target ({target}°C)")
    ax.axhspan(target - tol, target + tol, alpha=0.15, color="green", label="Safe band")
    ax.axhline(critical, color="red", linestyle=":", label=f"Critical ({critical}°C)")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(title)

    # Fan level
    ax = axes[1]
    ax.step(range(len(fans)), fans, color="steelblue", where="post", linewidth=1.5)
    ax.set_ylabel("Fan Level")
    ax.set_ylim(-0.5, max(4, max(fans) + 0.5))

    # Heat generation
    ax = axes[2]
    ax.plot(heat, color="orange", linewidth=1.5)
    ax.set_ylabel("Heat Gen (W)")

    # Cumulative reward
    if len(cum_reward) > 0:
        ax = axes[3]
        ax.plot(cum_reward, color="purple", linewidth=1.5)
        ax.set_ylabel("Cumulative Reward")
    axes[3].set_xlabel("Time Step")

    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_controller_comparison(
    dqn_trajectory: dict[str, list],
    pid_trajectory: dict[str, list],
    config: dict[str, Any],
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Side-by-side DQN vs PID comparison on a single episode.

    Shows temperature, fan levels, and cumulative rewards for both.
    """
    thermal_cfg = config.get("environment", {}).get("thermal", {})
    target = thermal_cfg.get("target_temp", 55.0)
    tol = thermal_cfg.get("temp_tolerance", 5.0)
    critical = thermal_cfg.get("critical_temp", 85.0)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for traj, label, color in [
        (dqn_trajectory, "DQN", "steelblue"),
        (pid_trajectory, "PID", "darkorange"),
    ]:
        infos = traj["infos"]
        temps = [info.get("temperature", 0) for info in infos]
        fans = [info.get("fan_level", 0) for info in infos]
        rewards = traj.get("rewards", [])
        cum_r = np.cumsum(rewards) if rewards else np.array([])

        axes[0].plot(temps, color=color, linewidth=1.5, label=label)
        axes[1].step(range(len(fans)), fans, color=color, where="post", linewidth=1.5, label=label)
        if len(cum_r) > 0:
            axes[2].plot(cum_r, color=color, linewidth=1.5, label=label)

    # Annotations
    axes[0].axhline(target, color="green", linestyle="--", alpha=0.6)
    axes[0].axhspan(target - tol, target + tol, alpha=0.1, color="green")
    axes[0].axhline(critical, color="red", linestyle=":", alpha=0.6)
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].legend()
    axes[0].set_title("DQN vs PID — Thermal Control")

    axes[1].set_ylabel("Fan Level")
    axes[1].legend()

    axes[2].set_ylabel("Cumulative Reward")
    axes[2].set_xlabel("Time Step")
    axes[2].legend()

    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig
