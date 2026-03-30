"""
Population initialization strategies.

Provides functions to create the initial population of chromosomes,
including pure random initialization and seeded initialization where
known good configurations are injected into the population.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .chromosome import CHROMOSOME_LENGTH, random_chromosome


def random_population(pop_size: int, seed: int = 42) -> List[List[float]]:
    """Create a fully random initial population."""
    rng = np.random.default_rng(seed)
    return [random_chromosome(rng) for _ in range(pop_size)]


# Known-good seed individuals (hand-coded sensible pipelines)
SEED_INDIVIDUALS = [
    # StandardScaler + SelectKBest(50%) + no DimRed + RandomForest(100, depth=10)
    [0.33, 0.33, 0.5, 0.0, 0.17, 0.31, 0.27, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5],
    # RobustScaler + none + PCA + LogisticRegression(C=1, lbfgs, 200, l2)
    [1.0, 0.0, 0.5, 0.5, 0.83, 0.33, 0.0, 0.22, 0.0, 0.5, 0.5, 0.5, 0.5],
    # MinMaxScaler + none + none + KNN(5, distance, euclidean)
    [0.67, 0.0, 0.5, 0.0, 0.67, 0.14, 1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5],
]


def seeded_population(pop_size: int, seed: int = 42) -> List[List[float]]:
    """Create a population with injected seed individuals + random fill.

    The first few slots are filled with hand-designed good configurations.
    The rest are random.
    """
    rng = np.random.default_rng(seed)
    pop = [list(ind) for ind in SEED_INDIVIDUALS]
    remaining = pop_size - len(pop)
    for _ in range(remaining):
        pop.append(random_chromosome(rng))
    return pop[:pop_size]


if __name__ == "__main__":
    pop = seeded_population(10, seed=42)
    print(f"Population size: {len(pop)}")
    for i, ind in enumerate(pop):
        print(f"  Individual {i}: [{', '.join(f'{g:.2f}' for g in ind)}]")
