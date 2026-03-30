"""
Custom genetic operators: crossover and mutation.

All operators are gene-type-aware:
  - Genes 0, 1, 3, 4 are categorical → uniform swap / random resample
  - Gene 2 is a float ratio → arithmetic crossover / Gaussian perturbation
  - Genes 5-12 are mixed (float, log_float, int, cat per classifier)
    → blended crossover with Gaussian or uniform perturbation

Operators modify individuals in-place and return them as tuples, following
the DEAP convention for registered operators.
"""
from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np

CATEGORICAL_GENES = {0, 1, 3, 4}
FLOAT_GENES = {2, 5, 6, 7, 8, 9, 10, 11, 12}


def cx_two_point_typed(
    ind1: List[float], ind2: List[float]
) -> Tuple[List[float], List[float]]:
    """Two-point crossover that respects gene types.

    For categorical genes at crossover boundaries, values are swapped
    directly. For continuous genes, a blend (BLX-alpha) is applied.
    """
    size = min(len(ind1), len(ind2))
    pt1 = random.randint(1, size - 2)
    pt2 = random.randint(pt1 + 1, size - 1)

    for i in range(pt1, pt2):
        if i in CATEGORICAL_GENES:
            ind1[i], ind2[i] = ind2[i], ind1[i]
        else:
            alpha = random.uniform(-0.1, 1.1)
            v1 = alpha * ind1[i] + (1 - alpha) * ind2[i]
            v2 = alpha * ind2[i] + (1 - alpha) * ind1[i]
            ind1[i] = max(0.0, min(1.0, v1))
            ind2[i] = max(0.0, min(1.0, v2))

    return ind1, ind2


def mut_mixed_type(
    individual: List[float], indpb: float = 0.15
) -> Tuple[List[float]]:
    """Gene-type-aware mutation.

    - Categorical genes: uniform random resample from [0, 1]
    - Continuous genes: Gaussian perturbation (sigma=0.15), clipped to [0, 1]
    """
    for i in range(len(individual)):
        if random.random() < indpb:
            if i in CATEGORICAL_GENES:
                individual[i] = random.random()
            else:
                individual[i] += random.gauss(0, 0.15)
                individual[i] = max(0.0, min(1.0, individual[i]))

    return (individual,)


if __name__ == "__main__":
    random.seed(42)
    a = [random.random() for _ in range(13)]
    b = [random.random() for _ in range(13)]
    print("Before CX:")
    print("  A:", [f"{x:.2f}" for x in a])
    print("  B:", [f"{x:.2f}" for x in b])
    cx_two_point_typed(a, b)
    print("After CX:")
    print("  A:", [f"{x:.2f}" for x in a])
    print("  B:", [f"{x:.2f}" for x in b])

    mut_mixed_type(a, indpb=0.3)
    print("After MUT A:", [f"{x:.2f}" for x in a])
