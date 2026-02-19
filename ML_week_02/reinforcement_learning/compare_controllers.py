#!/usr/bin/env python3
"""CLI entry point — DQN vs PID Controller Comparison.

Runs both the trained DQN agent and a classical PID controller on the
thermal-control environment and produces a side-by-side comparison plot
with quantitative control metrics.

Usage:
    python compare_controllers.py \
        --config config/thermal_control.yaml \
        --checkpoint outputs/models/thermal_control/checkpoint_best.pt \
        --episodes 20
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import torch

from src.agents.ddqn_agent import DDQNAgent
from src.agents.dqn_agent import DQNAgent
from src.agents.dueling_dqn_agent import DuelingDQNAgent
from src.control.pid_controller import PIDController
from src.control.control_utils import (
    compute_settling_time,
    compute_overshoot,
    compute_steady_state_error,
    compute_integral_absolute_error,
    compute_energy_cost,
)
from src.environments.thermal_control_env import ThermalControlEnv
from src.training.evaluator import Evaluator
from src.utils.config_loader import load_config, get_device
from src.utils.plotting import plot_controller_comparison

_AGENT_REGISTRY = {
    "dqn": DQNAgent,
    "ddqn": DDQNAgent,
    "dueling_dqn": DuelingDQNAgent,
}


def _run_pid_episode(
    env: ThermalControlEnv,
    pid: PIDController,
    max_steps: int = 500,
    seed: int = 42,
) -> dict:
    """Run a single episode with the PID controller."""
    state, info = env.reset(seed=seed)
    pid.reset()

    trajectory: dict = {
        "states": [state.copy()],
        "actions": [],
        "rewards": [],
        "infos": [info],
    }
    total_reward = 0.0

    for _ in range(max_steps):
        temperature = state[0]  # first obs dimension
        action = pid.compute_action(temperature)
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        trajectory["actions"].append(action)
        trajectory["rewards"].append(reward)
        trajectory["states"].append(next_state.copy())
        trajectory["infos"].append(info)

        state = next_state
        if terminated or truncated:
            break

    return {"total_reward": total_reward, "trajectory": trajectory}


@click.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--episodes", type=int, default=5, help="Episodes for comparison.")
@click.option("--device", type=str, default=None)
def main(
    config_path: str,
    checkpoint: str,
    episodes: int,
    device: str | None,
) -> None:
    """Compare DQN agent against PID controller on thermal control."""
    config = load_config(config_path)
    if device:
        config["experiment"]["device"] = device

    dev = get_device(config)
    thermal_cfg = config.get("environment", {}).get("thermal", {})
    fan_cost_table = thermal_cfg.get("fan_energy_cost", [0, 0.5, 1.5, 3.5, 7.0])
    target = thermal_cfg.get("target_temp", 55.0)
    dt = thermal_cfg.get("dt", 1.0)
    max_steps = config["agent"].get("max_steps_per_episode", 500)

    # Build agent
    agent_type = config["agent"]["type"]
    agent_cls = _AGENT_REGISTRY.get(agent_type)
    if agent_cls is None:
        raise click.ClickException(f"Unknown agent type '{agent_type}'.")

    agent = agent_cls(config, dev)
    agent.load(Path(checkpoint))
    agent.epsilon = 0.0

    # Build PID
    pid = PIDController.from_config(config)

    click.echo("═══ DQN vs PID — Thermal Control Comparison ═══\n")

    dqn_rewards, pid_rewards = [], []
    dqn_metrics_all, pid_metrics_all = [], []

    for ep in range(episodes):
        seed = config["experiment"].get("seed", 42) + ep

        # --- DQN rollout ---
        dqn_env = ThermalControlEnv(config=config)
        eval_cfg_override = {**config, "evaluation": {"n_eval_episodes": 1, **config.get("evaluation", {})}}
        dqn_evaluator = Evaluator(dqn_env, eval_cfg_override)
        dqn_result = dqn_evaluator.evaluate_with_trajectories(agent)
        dqn_traj = dqn_result["trajectories"][0]
        dqn_reward = dqn_result["rewards"][0]
        dqn_rewards.append(dqn_reward)
        dqn_env.close()

        # --- PID rollout ---
        pid_env = ThermalControlEnv(config=config)
        pid_result = _run_pid_episode(pid_env, pid, max_steps=max_steps, seed=seed)
        pid_traj = pid_result["trajectory"]
        pid_rewards.append(pid_result["total_reward"])
        pid_env.close()

        # --- Compute metrics for both ---
        for label, traj, reward_list, metric_list in [
            ("DQN", dqn_traj, dqn_rewards, dqn_metrics_all),
            ("PID", pid_traj, pid_rewards, pid_metrics_all),
        ]:
            temps = np.array([info["temperature"] for info in traj["infos"]])
            actions = np.array(traj["actions"])
            metrics = {
                "settling_time": compute_settling_time(temps, target, tolerance=0.05, dt=dt),
                "overshoot": compute_overshoot(temps, target),
                "sse": compute_steady_state_error(temps, target),
                "iae": compute_integral_absolute_error(temps, target, dt=dt),
                "energy": compute_energy_cost(actions, fan_cost_table),
            }
            metric_list.append(metrics)

    # --- Summary ---
    click.echo(f"{'Metric':<25} {'DQN':>12} {'PID':>12}")
    click.echo("─" * 50)
    click.echo(f"{'Mean Reward':<25} {np.mean(dqn_rewards):>12.1f} {np.mean(pid_rewards):>12.1f}")

    for key in ["settling_time", "overshoot", "sse", "iae", "energy"]:
        dqn_vals = [m[key] for m in dqn_metrics_all if m[key] is not None]
        pid_vals = [m[key] for m in pid_metrics_all if m[key] is not None]
        dqn_avg = np.mean(dqn_vals) if dqn_vals else float("nan")
        pid_avg = np.mean(pid_vals) if pid_vals else float("nan")
        click.echo(f"{key:<25} {dqn_avg:>12.2f} {pid_avg:>12.2f}")

    # --- Plot last episode ---
    plots_dir = Path(config.get("paths", {}).get("plots_dir", "outputs/plots"))
    save_path = plots_dir / "dqn_vs_pid_comparison.png"
    plot_controller_comparison(dqn_traj, pid_traj, config, save_path=save_path)
    click.echo(f"\nPlot saved → {save_path}")


if __name__ == "__main__":
    main()
