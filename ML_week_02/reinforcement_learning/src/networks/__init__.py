"""Q-network architectures."""

from .dqn_network import DQNNetwork
from .dueling_network import DuelingNetwork

__all__ = ["DQNNetwork", "DuelingNetwork"]
