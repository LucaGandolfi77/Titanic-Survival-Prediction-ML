"""
Population Initializer
======================
Generate initial populations biased toward small architectures.
"""

from __future__ import annotations

from typing import List

import numpy as np

from search_space.mlp_space import random_mlp_genome
from search_space.cnn_space import random_cnn_genome
from search_space.genome_encoder import repair


def random_population(
    pop_size: int,
    net_type: str,
    rng: np.random.Generator,
) -> List[List[float]]:
    """Generate a random population of genomes."""
    gen_fn = random_mlp_genome if net_type == "mlp" else random_cnn_genome
    population = []
    for _ in range(pop_size):
        genome = gen_fn(rng)
        genome = repair(genome, net_type)
        population.append(genome)
    return population


def biased_small_population(
    pop_size: int,
    net_type: str,
    rng: np.random.Generator,
    small_fraction: float = 0.5,
) -> List[List[float]]:
    """Generate a population biased toward compact architectures.

    A fraction of individuals is initialized with fewer layers/filters
    to guide the search toward lightweight models.
    """
    gen_fn = random_mlp_genome if net_type == "mlp" else random_cnn_genome
    n_small = int(pop_size * small_fraction)
    population = []

    for i in range(pop_size):
        genome = gen_fn(rng)
        if i < n_small:
            if net_type == "mlp":
                genome[0] = float(rng.integers(1, 3))  # 1-2 layers
                for j in range(1, 7):
                    genome[j] = min(genome[j], 128.0)  # cap width
            else:
                genome[0] = float(rng.integers(1, 3))  # 1-2 blocks
                for j in range(1, 6):
                    genome[j] = min(genome[j], 2.0)    # smaller filter index
        genome = repair(genome, net_type)
        population.append(genome)

    return population


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    pop = biased_small_population(10, "mlp", rng)
    for i, g in enumerate(pop[:3]):
        print(f"Individual {i}: n_layers={int(g[0])}, widths={[int(g[1+j]) for j in range(int(g[0]))]}")
