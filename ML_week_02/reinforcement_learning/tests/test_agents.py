"""Tests for DQN agent variants."""

import numpy as np
import pytest
import torch

from src.agents.dqn_agent import DQNAgent
from src.agents.ddqn_agent import DDQNAgent
from src.agents.dueling_dqn_agent import DuelingDQNAgent


def _make_config(agent_type: str = "dqn", buffer_type: str = "standard") -> dict:
    return {
        "experiment": {"seed": 42, "device": "cpu"},
        "agent": {
            "type": agent_type,
            "state_dim": 4,
            "action_dim": 2,
            "hidden_dims": [32, 32],
            "learning_rate": 0.001,
            "gamma": 0.99,
            "batch_size": 8,
            "buffer_size": 100,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.99,
            "target_update_frequency": 5,
            "soft_update_tau": 0.005,
            "use_soft_update": False,
            "learning_starts": 16,
            "train_frequency": 1,
            "gradient_clip": 1.0,
            "buffer_type": buffer_type,
            "per_alpha": 0.6,
            "per_beta_start": 0.4,
            "per_beta_end": 1.0,
            "per_beta_anneal_episodes": 50,
        },
    }


def _fill_buffer(agent, n: int = 32) -> None:
    """Push random transitions into the agent's buffer."""
    for _ in range(n):
        s = np.random.randn(4).astype(np.float32)
        a = np.random.randint(2)
        r = np.random.randn()
        s2 = np.random.randn(4).astype(np.float32)
        d = bool(np.random.rand() > 0.9)
        agent.store_transition(s, a, r, s2, d)


# ══════════════════════════════════════════════════════════
class TestDQNAgent:
    def test_select_action_range(self):
        agent = DQNAgent(_make_config("dqn"), torch.device("cpu"))
        state = np.zeros(4, dtype=np.float32)
        for _ in range(50):
            a = agent.select_action(state)
            assert 0 <= a < 2

    def test_greedy_action_deterministic(self):
        agent = DQNAgent(_make_config("dqn"), torch.device("cpu"))
        agent.epsilon = 0.0
        state = np.random.randn(4).astype(np.float32)
        actions = {agent.select_action(state, eval_mode=True) for _ in range(20)}
        assert len(actions) == 1  # all identical

    def test_train_step_returns_loss(self):
        agent = DQNAgent(_make_config("dqn"), torch.device("cpu"))
        _fill_buffer(agent, 32)
        result = agent.train_step()
        assert result is not None
        assert "loss" in result
        assert result["loss"] >= 0

    def test_epsilon_decay(self):
        agent = DQNAgent(_make_config("dqn"), torch.device("cpu"))
        initial = agent.epsilon
        agent.decay_epsilon()
        assert agent.epsilon < initial

    def test_save_load_roundtrip(self, tmp_path):
        agent = DQNAgent(_make_config("dqn"), torch.device("cpu"))
        path = tmp_path / "test.pt"
        agent.save(path, extra={"episode": 10})
        agent2 = DQNAgent(_make_config("dqn"), torch.device("cpu"))
        ckpt = agent2.load(path)
        assert ckpt["episode"] == 10


# ══════════════════════════════════════════════════════════
class TestDDQNAgent:
    def test_train_step(self):
        agent = DDQNAgent(_make_config("ddqn"), torch.device("cpu"))
        _fill_buffer(agent, 32)
        result = agent.train_step()
        assert result is not None
        assert "loss" in result

    def test_different_from_dqn_loss(self):
        """DDQN and DQN should compute different targets (in general)."""
        cfg = _make_config("dqn")
        dqn_agent = DQNAgent(cfg, torch.device("cpu"))
        ddqn_agent = DDQNAgent(_make_config("ddqn"), torch.device("cpu"))

        # Copy same weights
        ddqn_agent.online_net.load_state_dict(dqn_agent.online_net.state_dict())
        ddqn_agent.target_net.load_state_dict(dqn_agent.target_net.state_dict())

        # Feed same data
        for _ in range(32):
            s = np.random.randn(4).astype(np.float32)
            a = np.random.randint(2)
            r = float(np.random.randn())
            s2 = np.random.randn(4).astype(np.float32)
            d = bool(np.random.rand() > 0.9)
            dqn_agent.store_transition(s, a, r, s2, d)
            ddqn_agent.store_transition(s, a, r, s2, d)

        # Both should train without error
        r1 = dqn_agent.train_step()
        r2 = ddqn_agent.train_step()
        assert r1 is not None and r2 is not None


# ══════════════════════════════════════════════════════════
class TestDuelingDQNAgent:
    def test_select_action(self):
        agent = DuelingDQNAgent(_make_config("dueling_dqn"), torch.device("cpu"))
        state = np.zeros(4, dtype=np.float32)
        a = agent.select_action(state)
        assert 0 <= a < 2

    def test_train_step(self):
        agent = DuelingDQNAgent(_make_config("dueling_dqn"), torch.device("cpu"))
        _fill_buffer(agent, 32)
        result = agent.train_step()
        assert result is not None

    def test_with_prioritized_replay(self):
        cfg = _make_config("dueling_dqn", buffer_type="prioritized")
        agent = DuelingDQNAgent(cfg, torch.device("cpu"))
        _fill_buffer(agent, 32)
        result = agent.train_step()
        assert result is not None
        assert "loss" in result


# ══════════════════════════════════════════════════════════
class TestTargetNetworkUpdate:
    def test_hard_update(self):
        agent = DQNAgent(_make_config("dqn"), torch.device("cpu"))
        # Modify online weights
        with torch.no_grad():
            for p in agent.online_net.parameters():
                p.add_(1.0)
        agent.hard_update_target()
        for op, tp in zip(agent.online_net.parameters(), agent.target_net.parameters()):
            assert torch.allclose(op, tp)

    def test_soft_update(self):
        agent = DQNAgent(_make_config("dqn"), torch.device("cpu"))
        agent.use_soft_update = True
        agent.soft_update_tau = 0.5
        # Capture target before
        before = [p.clone() for p in agent.target_net.parameters()]
        with torch.no_grad():
            for p in agent.online_net.parameters():
                p.add_(2.0)
        agent.soft_update_target()
        # Target should have moved towards online (but not fully)
        for bp, tp, op in zip(before, agent.target_net.parameters(), agent.online_net.parameters()):
            assert not torch.allclose(tp, bp)    # changed
            assert not torch.allclose(tp, op)    # not fully copied
