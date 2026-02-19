"""Tests for replay buffers."""

import numpy as np
import pytest
import torch

from src.memory.replay_buffer import ReplayBuffer
from src.memory.prioritized_buffer import PrioritizedReplayBuffer


def _random_transition():
    s = np.random.randn(4).astype(np.float32)
    a = np.random.randint(2)
    r = float(np.random.randn())
    s2 = np.random.randn(4).astype(np.float32)
    d = bool(np.random.rand() > 0.8)
    return s, a, r, s2, d


# ══════════════════════════════════════════════════════════
class TestReplayBuffer:
    def test_push_and_len(self):
        buf = ReplayBuffer(capacity=100)
        for _ in range(10):
            buf.push(*_random_transition())
        assert len(buf) == 10

    def test_capacity_limit(self):
        buf = ReplayBuffer(capacity=5)
        for _ in range(20):
            buf.push(*_random_transition())
        assert len(buf) == 5

    def test_sample_shapes(self):
        buf = ReplayBuffer(capacity=100, seed=42)
        for _ in range(50):
            buf.push(*_random_transition())
        batch = buf.sample(8, device="cpu")
        assert batch["states"].shape == (8, 4)
        assert batch["actions"].shape == (8, 1)
        assert batch["rewards"].shape == (8, 1)
        assert batch["next_states"].shape == (8, 4)
        assert batch["dones"].shape == (8, 1)

    def test_sample_dtypes(self):
        buf = ReplayBuffer(capacity=100, seed=0)
        for _ in range(20):
            buf.push(*_random_transition())
        batch = buf.sample(4)
        assert batch["states"].dtype == torch.float32
        assert batch["actions"].dtype == torch.int64
        assert batch["dones"].dtype == torch.float32


# ══════════════════════════════════════════════════════════
class TestPrioritizedReplayBuffer:
    def test_push_and_len(self):
        buf = PrioritizedReplayBuffer(capacity=100, alpha=0.6, seed=42)
        for _ in range(15):
            buf.push(*_random_transition())
        assert len(buf) == 15

    def test_capacity_limit(self):
        buf = PrioritizedReplayBuffer(capacity=5, alpha=0.6, seed=42)
        for _ in range(20):
            buf.push(*_random_transition())
        assert len(buf) == 5

    def test_sample_shapes(self):
        buf = PrioritizedReplayBuffer(capacity=100, alpha=0.6, seed=42)
        for _ in range(50):
            buf.push(*_random_transition())
        batch = buf.sample(8, beta=0.4)
        assert batch["states"].shape == (8, 4)
        assert batch["weights"].shape == (8, 1)
        assert batch["indices"].shape == (8,)

    def test_update_priorities(self):
        buf = PrioritizedReplayBuffer(capacity=100, alpha=0.6, seed=42)
        for _ in range(50):
            buf.push(*_random_transition())
        batch = buf.sample(8, beta=0.4)
        td_errors = torch.rand(8)
        # Should not raise
        buf.update_priorities(batch["indices"], td_errors)

    def test_weights_sum_positive(self):
        buf = PrioritizedReplayBuffer(capacity=100, alpha=0.6, seed=42)
        for _ in range(50):
            buf.push(*_random_transition())
        batch = buf.sample(16, beta=0.4)
        assert (batch["weights"] > 0).all()
        assert batch["weights"].max() <= 1.0 + 1e-5  # normalised
