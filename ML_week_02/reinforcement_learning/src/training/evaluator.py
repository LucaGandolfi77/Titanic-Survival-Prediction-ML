"""Policy evaluation module.

Provides deterministic (greedy) evaluation of a trained agent over
multiple episodes, and returns summary statistics.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from ..agents.base_agent import BaseAgent


class Evaluator:
    """Evaluate a trained agent in greedy mode.

    Parameters
    ----------
    env : gym.Env
        Evaluation environment (can differ from training env).
    config : dict
        Full experiment configuration.
    """

    def __init__(self, env: gym.Env, config: dict[str, Any]) -> None:
        self.env = env
        eval_cfg = config.get("evaluation", {})
        self.n_episodes: int = eval_cfg.get("n_eval_episodes", 10)
        self.max_steps: int = config.get("agent", {}).get("max_steps_per_episode", 500)

    # ------------------------------------------------------------------
    def evaluate(self, agent: BaseAgent) -> dict[str, float]:
        """Run *n_episodes* greedy rollouts and return stats.

        Returns
        -------
        dict
            ``mean_reward``, ``std_reward``, ``min_reward``, ``max_reward``,
            ``mean_length``.
        """
        rewards: list[float] = []
        lengths: list[int] = []

        for _ in range(self.n_episodes):
            state, _ = self.env.reset()
            total_reward = 0.0

            for step in range(self.max_steps):
                action = agent.select_action(state, eval_mode=True)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                state = next_state
                if terminated or truncated:
                    break

            rewards.append(total_reward)
            lengths.append(step + 1)

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "mean_length": float(np.mean(lengths)),
        }

    # ------------------------------------------------------------------
    def evaluate_with_trajectories(
        self, agent: BaseAgent
    ) -> dict[str, Any]:
        """Full evaluation returning per-step trajectories.

        Useful for the thermal-control environment where we want to plot
        temperature vs. time.

        Returns
        -------
        dict
            ``rewards`` (list[float]), ``trajectories`` (list of dicts with
            per-step ``states``, ``actions``, ``rewards``, ``infos``).
        """
        all_rewards: list[float] = []
        trajectories: list[dict[str, list]] = []

        for _ in range(self.n_episodes):
            state, info = self.env.reset()
            episode: dict[str, list] = {
                "states": [state.copy()],
                "actions": [],
                "rewards": [],
                "infos": [info],
            }
            total_reward = 0.0

            for step in range(self.max_steps):
                action = agent.select_action(state, eval_mode=True)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward

                episode["actions"].append(action)
                episode["rewards"].append(reward)
                episode["states"].append(next_state.copy())
                episode["infos"].append(info)

                state = next_state
                if terminated or truncated:
                    break

            all_rewards.append(total_reward)
            trajectories.append(episode)

        return {
            "rewards": all_rewards,
            "mean_reward": float(np.mean(all_rewards)),
            "trajectories": trajectories,
        }
