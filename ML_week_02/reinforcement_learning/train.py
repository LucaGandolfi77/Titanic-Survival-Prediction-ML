#!/usr/bin/env python3
"""CLI entry point — DQN Training.

Usage examples:
    python train.py --config config/cartpole_dqn.yaml
    python train.py --config config/cartpole_ddqn.yaml --episodes 300
    python train.py --config config/thermal_control.yaml --device cpu
"""

from __future__ import annotations

from pathlib import Path

import click
import gymnasium as gym
import numpy as np
import torch

from src.agents.dqn_agent import DQNAgent
from src.agents.ddqn_agent import DDQNAgent
from src.agents.dueling_dqn_agent import DuelingDQNAgent
from src.environments.thermal_control_env import ThermalControlEnv
from src.environments.wrappers import EpisodeStatsWrapper
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator
from src.utils.config_loader import load_config, get_device
from src.utils.logger import RLLogger
from src.utils.plotting import plot_training_curves

# Agent registry
_AGENT_REGISTRY = {
    "dqn": DQNAgent,
    "ddqn": DDQNAgent,
    "dueling_dqn": DuelingDQNAgent,
}


def _make_env(config: dict) -> gym.Env:
    """Create the appropriate environment from config."""
    env_name = config["environment"]["name"]
    render_mode = config["environment"].get("render_mode")

    if env_name == "ThermalControl-v0":
        env = ThermalControlEnv(config=config, render_mode=render_mode)
    else:
        env = gym.make(env_name, render_mode=render_mode)

    env = EpisodeStatsWrapper(env)
    return env


@click.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True),
              help="Path to YAML configuration file.")
@click.option("--resume", type=click.Path(exists=True), default=None,
              help="Path to checkpoint to resume training from.")
@click.option("--episodes", type=int, default=None,
              help="Override number of training episodes.")
@click.option("--device", type=str, default=None,
              help="Override device (cpu / mps / cuda).")
@click.option("--seed", type=int, default=None,
              help="Override random seed.")
def main(
    config_path: str,
    resume: str | None,
    episodes: int | None,
    device: str | None,
    seed: int | None,
) -> None:
    """Train a DQN agent."""
    config = load_config(config_path)

    # CLI overrides
    if episodes:
        config["agent"]["n_episodes"] = episodes
    if device:
        config["experiment"]["device"] = device
    if seed is not None:
        config["experiment"]["seed"] = seed

    # Seed everything
    exp_seed = config["experiment"].get("seed", 42)
    np.random.seed(exp_seed)
    torch.manual_seed(exp_seed)

    dev = get_device(config)
    exp_name = config["experiment"].get("name", "rl_experiment")

    click.echo(f"═══ Training: {exp_name} ═══")
    click.echo(f"Agent : {config['agent']['type']}")
    click.echo(f"Env   : {config['environment']['name']}")
    click.echo(f"Device: {dev}")
    click.echo(f"Seed  : {exp_seed}")
    click.echo("─" * 50)

    # Build environment
    train_env = _make_env(config)
    eval_env = _make_env(config)

    # Build agent
    agent_type = config["agent"]["type"]
    agent_cls = _AGENT_REGISTRY.get(agent_type)
    if agent_cls is None:
        raise click.ClickException(
            f"Unknown agent type '{agent_type}'. Choose from: {list(_AGENT_REGISTRY)}"
        )
    agent = agent_cls(config, dev)

    # Logger
    log_cfg = config.get("logging", {})
    logger = RLLogger(
        experiment_name=exp_name,
        log_dir=log_cfg.get("log_dir", "outputs/tensorboard"),
        use_tensorboard=log_cfg.get("tensorboard", True),
    )

    # Evaluator
    evaluator = Evaluator(eval_env, config)

    # Resume
    resume_episode = 0
    if resume:
        ckpt = agent.load(Path(resume))
        resume_episode = ckpt.get("episode", 0) + 1
        click.echo(f"Resumed from episode {resume_episode}")

    # Train
    trainer = Trainer(agent, train_env, config, logger, evaluator)
    history = trainer.train(resume_episode=resume_episode)

    # Save training curves
    plots_dir = Path(config.get("paths", {}).get("plots_dir", "outputs/plots"))
    plot_training_curves(
        rewards=history["rewards"],
        losses=history["losses"],
        title=f"{exp_name} — Training Progress",
        save_path=plots_dir / f"{exp_name}_training_curves.png",
    )
    click.echo(f"\nTraining complete. Plots saved to {plots_dir}")

    logger.close()
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
