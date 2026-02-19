"""Prioritized Experience Replay (Schaul et al., 2016).

Uses a Sum-Tree data structure for O(log N) sampling proportional to
transition priorities, and importance-sampling weights to correct the
induced bias.
"""

from __future__ import annotations

import numpy as np
import torch


class _SumTree:
    """Binary tree where each leaf stores a priority and the internal nodes
    store the sum of their children.  This allows O(log N) proportional
    sampling and O(log N) priority updates.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._data: list = [None] * capacity
        self._write_idx = 0
        self._size = 0

    # ----- public API --------------------------------------------------
    @property
    def total(self) -> float:
        return float(self._tree[0])

    @property
    def min_priority(self) -> float:
        leaf_nodes = self._tree[self.capacity - 1 : self.capacity - 1 + self._size]
        if len(leaf_nodes) == 0:
            return 0.0
        return float(leaf_nodes.min())

    def add(self, priority: float, data: tuple) -> None:
        idx = self._write_idx + self.capacity - 1
        self._data[self._write_idx] = data
        self._update(idx, priority)
        self._write_idx = (self._write_idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def update(self, tree_idx: int, priority: float) -> None:
        self._update(tree_idx, priority)

    def get(self, cumsum: float) -> tuple[int, float, tuple]:
        """Retrieve the leaf whose cumulative priority range contains *cumsum*."""
        idx = self._retrieve(0, cumsum)
        data_idx = idx - self.capacity + 1
        return idx, float(self._tree[idx]), self._data[data_idx]

    # ----- internals ---------------------------------------------------
    def _update(self, idx: int, priority: float) -> None:
        delta = priority - self._tree[idx]
        self._tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self._tree[idx] += delta

    def _retrieve(self, idx: int, cumsum: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self._tree):
            return idx
        if cumsum <= self._tree[left]:
            return self._retrieve(left, cumsum)
        return self._retrieve(right, cumsum - self._tree[left])

    def __len__(self) -> int:
        return self._size


# ======================================================================
class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions.
    alpha : float
        Prioritization exponent (0 = uniform, 1 = full prioritization).
    seed : int | None
        Random seed.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        seed: int | None = None,
    ) -> None:
        self._tree = _SumTree(capacity)
        self._alpha = alpha
        self._capacity = capacity
        self._max_priority = 1.0
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition with maximum priority (ensures it gets sampled)."""
        priority = self._max_priority ** self._alpha
        self._tree.add(priority, (state, action, reward, next_state, done))

    # ------------------------------------------------------------------
    def sample(
        self,
        batch_size: int,
        beta: float = 0.4,
        device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Sample proportional to stored priorities.

        Returns
        -------
        dict  with keys  states, actions, rewards, next_states, dones,
              weights (IS weights), indices (tree indices for update).
        """
        indices: list[int] = []
        priorities: list[float] = []
        samples: list[tuple] = []

        total = self._tree.total
        segment = total / batch_size

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            cumsum = self._rng.uniform(low, high)
            idx, pri, data = self._tree.get(cumsum)
            indices.append(idx)
            priorities.append(pri)
            samples.append(data)

        # Importance-sampling weights
        n = len(self._tree)
        min_prob = self._tree.min_priority / total if total > 0 else 1e-8
        min_prob = max(min_prob, 1e-8)
        max_weight = (n * min_prob) ** (-beta)

        probs = np.array(priorities) / total
        probs = np.clip(probs, 1e-8, None)
        weights = (n * probs) ** (-beta)
        weights /= max_weight  # normalise

        # Build tensors
        states = torch.tensor(
            np.array([s[0] for s in samples]), dtype=torch.float32, device=device
        )
        actions = torch.tensor(
            [s[1] for s in samples], dtype=torch.int64, device=device
        ).unsqueeze(1)
        rewards = torch.tensor(
            [s[2] for s in samples], dtype=torch.float32, device=device
        ).unsqueeze(1)
        next_states = torch.tensor(
            np.array([s[3] for s in samples]), dtype=torch.float32, device=device
        )
        dones = torch.tensor(
            [s[4] for s in samples], dtype=torch.float32, device=device
        ).unsqueeze(1)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(1)
        indices_t = torch.tensor(indices, dtype=torch.int64, device=device)

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
            "weights": weights_t,
            "indices": indices_t,
        }

    # ------------------------------------------------------------------
    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor) -> None:
        """Update priorities for sampled transitions.

        Parameters
        ----------
        indices : Tensor of tree indices.
        td_errors : Tensor of |δ| (absolute TD-errors).
        """
        priorities = (td_errors.detach().cpu().numpy().flatten() + 1e-6) ** self._alpha
        for idx, pri in zip(indices.cpu().numpy().flatten(), priorities):
            self._tree.update(int(idx), float(pri))
            self._max_priority = max(self._max_priority, float(pri) ** (1.0 / self._alpha))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._tree)

    @property
    def capacity(self) -> int:
        return self._capacity
