# ecosim/simulation.py
from __future__ import annotations
import random
from typing import List, Dict
from .config import SimulationConfig
from .environment import Ecosystem

class Simulation:
    """Runs a full simulation and returns time‑series data."""
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = random.Random(config.random_seed)

    def run(self) -> List[Dict]:
        eco = Ecosystem(
            grass_config=self.config.grass,
            herbivore_config=self.config.herbivore,
            carnivore_config=self.config.carnivore,
            initial_grass=self.config.initial_grass_amount,
            initial_herbivores=self.config.initial_herbivores,
            initial_carnivores=self.config.initial_carnivores,
            rng=self.rng,
        )
        results: List[Dict] = []
        for step in range(self.config.num_steps):
            eco._step_num = step  # so environment.step can label it
            step_stats = eco.step()
            if (step % self.config.logging_frequency) == 0:
                results.append(step_stats)
        return results
