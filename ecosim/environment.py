# ecosim/environment.py
from __future__ import annotations
import random
from typing import List, Tuple
from .entities import Herbivore, Carnivore
from .config import GrassConfig, HerbivoreConfig, CarnivoreConfig

class Ecosystem:
    """Holds state and advances the simulation by one discrete step."""
    def __init__(
        self,
        grass_config: GrassConfig,
        herbivore_config: HerbivoreConfig,
        carnivore_config: CarnivoreConfig,
        initial_grass: float,
        initial_herbivores: int,
        initial_carnivores: int,
        rng: random.Random,
    ):
        self.grass_config = grass_config
        self.herbivore_config = herbivore_config
        self.carnivore_config = carnivore_config
        self.rng = rng

        self.grass: float = max(0.0, min(initial_grass, grass_config.carrying_capacity))

        self.herbivores: List[Herbivore] = [
            Herbivore(herbivore_config) for _ in range(initial_herbivores)
        ]
        self.carnivores: List[Carnivore] = [
            Carnivore(carnivore_config) for _ in range(initial_carnivores)
        ]

        # Counters for logging
        self.stats: dict = {}

    def _regrow_grass(self) -> None:
        """Logistic‑like regrowth toward carrying capacity."""
        capacity = self.grass_config.carrying_capacity
        rate = self.grass_config.regrowth_rate
        # Simple proportional regrowth: dG = r*(K - G)
        growth = rate * (capacity - self.grass)
        self.grass = min(capacity, self.grass + growth)

    def _feed_herbivores(self) -> Tuple[float, List[float]]:
        """Each herbivore attempts to eat grass; returns total consumed and per‑herbivore intakes."""
        intake_per = []
        total_requested = 0.0
        for h in self.herbivores:
            request = self.grass_config.consumption_per_herbivore
            total_requested += request
        # If request exceeds available grass, scale down proportionally
        if total_requested > self.grass and self.grass > 0:
            scale = self.grass / total_requested
            for h in self.herbivores:
                eaten = h.eat(self.grass_config.consumption_per_herbivore * scale)
                intake_per.append(eaten)
        else:
            for h in self.herbivores:
                eaten = h.eat(self.grass_config.consumption_per_herbivore)
                intake_per.append(eaten)
        total_consumed = sum(intake_per)
        self.grass = max(0.0, self.grass - total_consumed)
        return total_consumed, intake_per

    def _update_herbivores(self) -> Tuple[List[Herbivore], int, int]:
        """Survival, reproduction, aging; returns new list, births, deaths."""
        new_herbivores: List[Herbivore] = []
        births = deaths = 0
        for h in self.herbivores:
            h.reset_step()
            if h.update_survival():
                new_herbivores.append(h)
                # reproduction
                if self.rng.random() < h.reproduction_probability():
                    new_herbivores.append(Herbivore(self.herbivore_config))
                    births += 1
                h.age_one_step()
            else:
                deaths += 1
        return new_herbivores, births, deaths

    def _hunt_herbivores(self) -> Tuple[int, List[int]]:
        """Determine total kills, assign to carnivores, return total kills and per‑carnivore kills."""
        herbivore_count = len(self.herbivores)
        carnivore_count = len(self.carnivores)
        if herbivore_count == 0 or carnivore_count == 0:
            return 0, [0] * carnivore_count

        # Base hunt success probability modulated by herbivore‑to‑carnivore ratio
        ratio = herbivore_count / max(1, carnivore_count)
        # More herbivores → higher success, more carnivores → lower success (simple linear)
        prob = self.carnivore_config.base_hunt_success_prob * (0.5 + 0.5 * min(ratio, 2.0) / 2.0)
        prob = max(0.0, min(1.0, prob))

        # Expected number of successful hunts this step
        expected_kills = prob * carnivore_count
        # Cap by available herbivores
        total_possible = min(int(self.rng.poisson(expected_kills)) if expected_kills > 0 else 0,
                             herbivore_count)
        # Actually draw kills from binomial distribution for more realism
        total_killed = self.rng.binomial(herbivore_count, prob)
        total_killed = min(total_killed, herbivore_count)

        # Distribute kills among carnivores (multinomial)
        kill_per_carn = [0] * carnivore_count
        if total_killed > 0 and carnivore_count > 0:
            # Assign each kill to a random carnivore
            for _ in range(total_killed):
                idx = self.rng.randrange(carnivore_count)
                kill_per_carn[idx] += 1

        # Remove killed herbivores (randomly chosen)
        if total_killed > 0:
            indices = self.rng.sample(range(herbivore_count), total_killed)
            # Delete from end to avoid index shifting
            for idx in sorted(indices, reverse=True):
                del self.herbivores[idx]
        return total_killed, kill_per_carn

    def _update_carnivores(
        self,
        kills_per_carn: List[int],
        herbivore_population: int
    ) -> Tuple[List[Carnivore], int, int]:
        """Survival, reproduction, aging; returns new list, births, deaths."""
        new_carnivores: List[Carnivore] = []
        births = deaths = 0
        for c, k in zip(self.carnivores, kills_per_carn):
            c.reset_step()
            c.record_kill(k)
            if c.update_survival(herbivore_population):
                new_carnivores.append(c)
                # reproduction
                if self.rng.random() < c.reproduction_probability(herbivore_population):
                    new_carnivores.append(Carnivore(self.carnivore_config))
                    births += 1
                c.age_one_step()
            else:
                deaths += 1
        return new_carnivores, births, deaths

    def step(self) -> dict:
        """Advend one time step, populate stats dict, and return it."""
        self._regrow_grass()
        total_consumed, intake_per = self._feed_herbivores()
        avg_intake = sum(intake_per) / len(intake_per) if intake_per else 0.0

        self.herbivores, herb_births, herb_deaths = self._update_herbivores()
        herbivore_after = len(self.herbivores)

        total_killed, kills_per_carn = self._hunt_herbivores()
        self.carnivores, carn_births, carn_deaths = self._update_carnivores(
            kills_per_carn, herbivore_after
        )
        carnivore_after = len(self.carnivores)

        avg_kills = (
            sum(c.kills_in_window() for c in self.carnivores) /
            len(self.carnivores) if self.carnivores else 0.0
        )

        self.stats = {
            "step": getattr(self, "_step_num", -1) + 1,
            "grass": self.grass,
            "herbivores": herbivore_after,
            "carnivores": carnivore_after,
            "herb_births": herb_births,
            "herb_deaths": herb_deaths,
            "carn_births": carn_births,
            "carn_deaths": carn_deaths,
            "avg_herb_intake": avg_intake,
            "avg_carn_kills": avg_kills,
            "total_killed_this_step": total_killed,
            "grass_fraction": self.grass / self.grass_config.carrying_capacity,
        }
        # store step number for next call
        self._step_num = self.stats["step"]
        return self.stats
