# ecosim/entities.py
from __future__ import annotations
import random
from typing import List, Deque
from collections import deque

class Herbivore:
    """Individual herbivore agent."""
    _next_id: int = 0

    def __init__(self, config: 'HerbivoreConfig'):
        self.id: int = Herbivore._next_id
        Herbivore._next_id += 1
        self.config = config
        self.age: int = 0
        self.consecutive_low_intake: int = 0   # steps with intake < min
        self.last_intake: float = 0.0          # grass eaten last step
        self.health: float = 1.0               # simple proxy (1 = full)

    def reset_step(self) -> None:
        """Call at start of each step to clear per‑step counters."""
        self.last_intake = 0.0

    def eat(self, available_grass: float) -> float:
        """Consume grass, update intake and health, return amount eaten."""
        intake = min(self.config.consumption_per_herbivore, available_grass)
        self.last_intake = intake
        # health improves with food, decays otherwise
        self.health = min(1.0, self.health + 0.05 * (intake / self.config.consumption_per_herbivore))
        self.health = max(0.0, self.health - 0.02)  # baseline decay
        return intake

    def update_survival(self) -> bool:
        """Return True if herbivore survives this step."""
        if self.last_intake >= self.config.min_intake_to_survive:
            self.consecutive_low_intake = 0
        else:
            self.consecutive_low_intake += 1
        if self.consecutive_low_intake >= self.config.starvation_steps_before_death:
            return False
        return True

    def reproduction_probability(self) -> float:
        """Probability of producing an offspring this step."""
        if self.last_intake >= self.config.intake_for_reproduction:
            return self.config.base_reproduction_rate
        # linear scaling between min and reproduction intake
        if self.last_intake >= self.config.min_intake_to_survive:
            ratio = (self.last_intake - self.config.min_intake_to_survive) / \
                    (self.config.intake_for_reproduction - self.config.min_intake_to_survive)
            return self.config.base_reproduction_rate * ratio
        return 0.0

    def age_one_step(self) -> None:
        self.age += 1


class Carnivore:
    """Individual carnivore agent with sliding kill window."""
    _next_id: int = 0

    def __init__(self, config: 'CarnivoreConfig'):
        self.id: int = Carnivore._next_id
        Carnivore._next_id += 1
        self.config = config
        self.age: int = 0
        self.kill_history: Deque[int] = deque(maxlen=config.kill_window_length)
        self.health: float = 1.0

    def reset_step(self) -> None:
        """Prepare for a new step (no per‑step counters needed)."""
        pass

    def record_kill(self, killed: int = 1) -> None:
        """Add kill count for this step to the sliding window."""
        self.kill_history.append(killed)

    def kills_in_window(self) -> int:
        return sum(self.kill_history)

    def update_survival(self, herbivore_population: int) -> bool:
        """Return True if carnivore survives this step."""
        # Starvation if not enough kills in window
        if self.kills_in_window() < self.config.required_kills_per_window:
            # Increase mortality when herbivores are critically low
            if herbivore_population < self.config.critical_herbivore_level:
                return random.random() < 0.7  # 30% extra death chance
            return False
        # Optional health decay
        self.health = max(0.0, self.health - 0.01)
        return self.health > 0.0

    def reproduction_probability(self, herbivore_population: int) -> float:
        """Conservative reproduction: only if herbivores abundant and kill rate sustainable."""
        if herbivore_population < self.config.herbivore_safety_threshold:
            return 0.0
        avg_kills = self.kills_in_window() / len(self.kill_history) if self.kill_history else 0
        # Reproduce only if average kills meets requirement but not excessively high
        if avg_kills < self.config.required_kills_per_window:
            return 0.0
        # If kill rate is very high, reduce reproduction to avoid over‑predation
        excess = avg_kills - self.config.required_kills_per_window
        factor = max(0.0, 1.0 - excess * 0.5)  # simple damping
        return self.config.reproduction_rate * factor

    def age_one_step(self) -> None:
        self.age += 1
