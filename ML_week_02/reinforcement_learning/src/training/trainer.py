"""Training loop orchestration.

Manages the agent–environment interaction loop, epsilon scheduling,
target-network updates, logging, checkpointing, and evaluation calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from ..agents.base_agent import BaseAgent
from ..utils.logger import RLLogger
from .evaluator import Evaluator


class Trainer:
    """Runs the full DQN training loop.

    Parameters
    ----------
    agent : BaseAgent
        Any DQN-variant agent.
    env : gym.Env
        Training environment.
    config : dict
        Full experiment config.
    logger : RLLogger
        TensorBoard + file logger.
    evaluator : Evaluator | None
        Used for periodic evaluation runs.
    """

    def __init__(
        self,
        agent: BaseAgent,
        env: gym.Env,
        config: dict[str, Any],
        logger: RLLogger,
        evaluator: Evaluator | None = None,
    ) -> None:
        self.agent = agent
        self.env = env
        self.config = config
        self.logger = logger
        self.evaluator = evaluator

        agent_cfg = config["agent"]
        self.n_episodes: int = agent_cfg.get("n_episodes", 500)
        self.max_steps: int = agent_cfg.get("max_steps_per_episode", 500)
        self.train_frequency: int = agent_cfg.get("train_frequency", 4)

        eval_cfg = config.get("evaluation", {})
        self.eval_frequency: int = eval_cfg.get("eval_frequency", 50)

        log_cfg = config.get("logging", {})
        self.ckpt_frequency: int = log_cfg.get("checkpoint_frequency", 50)
        self.save_best: bool = log_cfg.get("save_best_only", True)

        paths = config.get("paths", {})
        self.models_dir = Path(paths.get("models_dir", "outputs/models"))

        # Tracking
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.losses: list[float] = []
        self.best_eval_reward: float = -float("inf")

    # ==================================================================
    def train(self, resume_episode: int = 0) -> dict[str, list]:
        """Run training for ``n_episodes``.

        Returns
        -------
        dict
            History with keys ``rewards``, ``lengths``, ``losses``,
            ``epsilons``, ``eval_rewards``.
        """
        eval_rewards: list[tuple[int, float]] = []

        for episode in range(resume_episode, self.n_episodes):
            ep_reward, ep_length, ep_losses = self._run_episode()

            self.episode_rewards.append(ep_reward)
            self.episode_lengths.append(ep_length)
            if ep_losses:
                self.losses.extend(ep_losses)

            # Epsilon decay (per episode)
            self.agent.decay_epsilon()

            # PER beta annealing
            if hasattr(self.agent, "anneal_per_beta"):
                self.agent.anneal_per_beta(episode)

            # Target network update
            if (episode + 1) % self.agent.target_update_frequency == 0:
                self.agent.update_target()

            # Logging
            avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
            self.logger.log_episode(
                episode=episode,
                reward=ep_reward,
                length=ep_length,
                loss=avg_loss,
                epsilon=self.agent.epsilon,
            )

            # Periodic evaluation
            if self.evaluator and (episode + 1) % self.eval_frequency == 0:
                eval_result = self.evaluator.evaluate(self.agent)
                eval_rewards.append((episode, eval_result["mean_reward"]))
                self.logger.log_eval(episode, eval_result)

                # Save best model
                if self.save_best and eval_result["mean_reward"] > self.best_eval_reward:
                    self.best_eval_reward = eval_result["mean_reward"]
                    self._save_checkpoint(episode, tag="best")

            # Periodic checkpoint
            if (episode + 1) % self.ckpt_frequency == 0:
                self._save_checkpoint(episode, tag=f"ep{episode + 1}")

            # Console progress
            if (episode + 1) % 10 == 0:
                recent = self.episode_rewards[-10:]
                self.logger.info(
                    f"Episode {episode + 1:4d}/{self.n_episodes} | "
                    f"Reward: {ep_reward:7.1f} | "
                    f"Avg10: {np.mean(recent):7.1f} | "
                    f"ε: {self.agent.epsilon:.3f}"
                )

        # Final checkpoint
        self._save_checkpoint(self.n_episodes - 1, tag="final")

        return {
            "rewards": self.episode_rewards,
            "lengths": self.episode_lengths,
            "losses": self.losses,
            "eval_rewards": eval_rewards,
        }

    # ==================================================================
    def _run_episode(self) -> tuple[float, int, list[float]]:
        """Execute one full episode and return (total_reward, length, losses)."""
        state, _ = self.env.reset()
        total_reward = 0.0
        losses: list[float] = []

        for step in range(self.max_steps):
            action = self.agent.select_action(state)
            next_state, reward, terminated, truncated, _info = self.env.step(action)

            self.agent.store_transition(state, action, reward, next_state, terminated)
            total_reward += reward

            # Train every N steps (after warm-up)
            if (
                self.agent.total_steps >= self.agent.learning_starts
                and self.agent.total_steps % self.train_frequency == 0
            ):
                result = self.agent.train_step()
                if result and "loss" in result:
                    losses.append(result["loss"])

            state = next_state
            if terminated or truncated:
                break

        return total_reward, step + 1, losses

    # ------------------------------------------------------------------
    def _save_checkpoint(self, episode: int, tag: str = "") -> None:
        filename = f"checkpoint_{tag}.pt" if tag else f"checkpoint_ep{episode + 1}.pt"
        path = self.models_dir / filename
        self.agent.save(
            path,
            extra={
                "episode": episode,
                "best_eval_reward": self.best_eval_reward,
                "episode_rewards": self.episode_rewards,
            },
        )
        self.logger.info(f"Saved checkpoint → {path}")
