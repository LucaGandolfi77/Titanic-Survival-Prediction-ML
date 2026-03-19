# ecosim/config.py
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class GrassConfig:
    carrying_capacity: float = 1000.0
    regrowth_rate: float = 0.1          # fraction of (capacity - current) per step
    consumption_per_herbivore: float = 10.0

@dataclass
class HerbivoreConfig:
    min_intake_to_survive: float = 5.0
    intake_for_reproduction: float = 15.0
    starvation_steps_before_death: int = 3
    base_reproduction_rate: float = 0.05   # probability per step when well‑fed

@dataclass
class CarnivoreConfig:
    required_kills_per_window: int = 2
    kill_window_length: int = 5
    reproduction_rate: float = 0.02       # intentionally low
    max_density_factor: Optional[float] = 0.2  # carnivores ≤ factor * herbivores
    base_hunt_success_prob: float = 0.3
    herbivore_safety_threshold: int = 10   # min herbivores for reproduction
    critical_herbivore_level: int = 5      # below this, suppress reproduction & increase mortality

@dataclass
class SimulationConfig:
    num_steps: int = 500
    random_seed: Optional[int] = 42
    logging_frequency: int = 1            # log every step (set >1 to thin)
    initial_grass_amount: float = 500.0
    initial_herbivores: int = 50
    initial_carnivores: int = 10

    # Sub‑configs
    grass: GrassConfig = field(default_factory=GrassConfig)
    herbivore: HerbivoreConfig = field(default_factory=HerbivoreConfig)
    carnivore: CarnivoreConfig = field(default_factory=CarnivoreConfig)

    # Presets (class method)
    @classmethod
    def preset(cls, name: str) -> "SimulationConfig":
        base = cls()
        if name == "balanced":
            pass  # use defaults
        elif name == "high_grass":
            base.grass.carrying_capacity = 2000.0
            base.grass.regrowth_rate = 0.15
        elif name == "predator_heavy":
            base.initial_carnivores = 25
            base.carnivore.reproduction_rate = 0.04
        elif name == "low_regrowth":
            base.grass.regrowth_rate = 0.02
        else:
            raise ValueError(f"Unknown preset: {name}")
        return base
