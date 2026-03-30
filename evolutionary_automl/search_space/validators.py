"""
Chromosome validation and repair.

Ensures gene values are within valid bounds and that gene combinations
are feasible. Invalid chromosomes are repaired rather than discarded to
avoid wasting evolutionary computation budget.
"""
from __future__ import annotations

from typing import List

from .space_definition import (
    CLASSIFIER_OPTIONS,
    FEATURE_SEL_OPTIONS,
    HYPERPARAMETER_SPACE,
    categorical_index,
)


def clip_genes(chromosome: List[float]) -> List[float]:
    """Clip all gene values to [0, 1]."""
    return [max(0.0, min(1.0, g)) for g in chromosome]


def repair_chromosome(chromosome: List[float]) -> List[float]:
    """Repair an invalid chromosome in-place and return it.

    Repairs include:
    - Clipping all genes to [0,1]
    - Fixing LogisticRegression penalty/solver incompatibilities
      (handled at pipeline build time, but we ensure genes stay bounded)
    """
    chromosome = clip_genes(chromosome)

    if len(chromosome) < 13:
        chromosome.extend([0.5] * (13 - len(chromosome)))
    elif len(chromosome) > 13:
        chromosome = chromosome[:13]

    return chromosome


def is_valid(chromosome: List[float]) -> bool:
    """Check whether a chromosome has valid gene values."""
    if len(chromosome) != 13:
        return False
    return all(0.0 <= g <= 1.0 for g in chromosome)


if __name__ == "__main__":
    bad = [1.5, -0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    repaired = repair_chromosome(bad)
    print("Repaired:", repaired)
    print("Valid:", is_valid(repaired))
