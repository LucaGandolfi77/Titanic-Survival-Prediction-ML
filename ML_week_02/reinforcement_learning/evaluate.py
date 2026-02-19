#!/usr/bin/env python3
"""CLI entry point — Policy Evaluation.

Usage:
    python evaluate.py --config config/cartpole_dqn.yaml \
                       --checkpoint outputs/models/cartpole/checkpoint_best.pt \
                       --episodes 50
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
from src.training.evaluator import Evaluator
from src.utils.config_loader import load_config, get_device

_AGENT_REGISTRY = {
    "dqn": DQNAgent,
    "ddqn": DDQNAgent,
    "dueling_dqn": DuelingDQNAgent,
}


def _make_env(config: dict) -> gym.Env:
    env_name = config["environment"]["name"]
    render_mode = config["environment"].get("render_mode")
    if env_name == "ThermalControl-v0":
        env = ThermalControlEnv(config=config, render_mode=render_mode)
    else:
        env = gym.make(env_name, render_mode=render_mode)
    return EpisodeStatsWrapper(env)


@click.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--checkpoint", required=True, type=click.Path(exists=True),
              help="Path to trained model checkpoint.")
@click.option("--episodes", type=int, default=None,
              help="Number of evaluation episodes (overrides config).")
@click.option("--device", type=str, default=None)
@click.option("--render", is_flag=True, help="Render environment during evaluation.")
def main(
    config_path: str,
    checkpoint: str,
    episodes: int | None,
    device: str | None,
    render: bool,
) -> None:
    """Evaluate a trained DQN agent."""
    config = load_config(config_path)
    if device:
        config["experiment"]["device"] = device
    if episodes:
        config["evaluation"]["n_eval_episodes"] = episodes
    if render:
        config["environment"]["render_mode"] = "human"

    dev = get_device(config)
    agent_type = config["agent"]["type"]
    agent_cls = _AGENT_REGISTRY.get(agent_type)
    if agent_cls is None:
        raise click.ClickException(f"Unknown agent type '{agent_type}'.")

    agent = agent_cls(config, dev)
    agent.load(Path(checkpoint))
    agent.epsilon = 0.0  # fully greedy

    env = _make_env(config)
    evaluator = Evaluator(env, config)
    result = evaluator.evaluate(agent)

    click.echo("═══ Evaluation Results ═══")
    click.echo(f"  Mean Reward : {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
    click.echo(f"  Min  Reward : {result['min_reward']:.2f}")
    click.echo(f"  Max  Reward : {result['max_reward']:.2f}")
    click.echo(f"  Mean Length : {result['mean_length']:.1f}")

    env.close()


if __name__ == "__main__":
    main()
