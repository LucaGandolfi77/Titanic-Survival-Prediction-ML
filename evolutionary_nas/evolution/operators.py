"""
Genetic Operators
=================
Gene-type-aware crossover and mutation operators compatible with DEAP.
After every operator, repair() is called to fix constraint violations.
"""

from __future__ import annotations

import math
import random
from typing import Any, List, Tuple

from search_space.genome_encoder import repair
from search_space.mlp_space import MLP_GENE_SPECS, MLP_GENOME_LENGTH
from search_space.cnn_space import CNN_GENE_SPECS, CNN_GENOME_LENGTH
from config import CFG


def _get_specs(net_type: str):
    if net_type == "mlp":
        return MLP_GENE_SPECS
    return CNN_GENE_SPECS


def cx_two_point_typed(
    ind1: List[float],
    ind2: List[float],
    net_type: str,
) -> Tuple[List[float], List[float]]:
    """Gene-type-aware two-point crossover.

    - Categorical genes: swap directly between parents
    - Float genes: SBX crossover with eta=CFG.SBX_ETA
    - Integer genes: swap or blend randomly
    """
    specs = _get_specs(net_type)
    size = len(specs)
    pt1, pt2 = sorted(random.sample(range(size), 2))

    child1 = list(ind1)
    child2 = list(ind2)

    for i in range(pt1, pt2 + 1):
        spec = specs[i]
        if spec.gene_type in ("cat", "cat_zero"):
            child1[i], child2[i] = child2[i], child1[i]
        elif spec.gene_type == "float" or spec.gene_type == "log_float":
            child1[i], child2[i] = _sbx(ind1[i], ind2[i], spec.low, spec.high)
        elif spec.gene_type == "int":
            if random.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]
            else:
                avg = (ind1[i] + ind2[i]) / 2.0
                child1[i] = float(round(avg))
                child2[i] = float(round(avg))

    child1 = repair(child1, net_type)
    child2 = repair(child2, net_type)
    ind1[:] = child1
    ind2[:] = child2
    return ind1, ind2


def _sbx(x1: float, x2: float, low: float, high: float) -> Tuple[float, float]:
    """Simulated Binary Crossover for a single gene."""
    eta = CFG.SBX_ETA
    if abs(x1 - x2) < 1e-14:
        return x1, x2
    u = random.random()
    if u <= 0.5:
        beta = (2.0 * u) ** (1.0 / (eta + 1.0))
    else:
        beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
    c1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
    c2 = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
    c1 = max(low, min(high, c1))
    c2 = max(low, min(high, c2))
    return c1, c2


def mut_mixed_type(
    individual: List[float],
    net_type: str,
    indpb: float = 0.15,
) -> Tuple[List[float]]:
    """Gene-type-aware mutation.

    - Categorical genes: uniform random resample
    - Integer genes: random perturbation ±1..3
    - Float/log_float genes: Gaussian perturbation
    """
    specs = _get_specs(net_type)

    for i, spec in enumerate(specs):
        if random.random() > indpb:
            continue

        if spec.gene_type in ("cat", "cat_zero"):
            n = len(spec.choices)
            individual[i] = float(random.randint(0, n - 1))

        elif spec.gene_type == "int":
            delta = random.randint(-3, 3)
            val = int(individual[i]) + delta
            individual[i] = float(max(int(spec.low), min(int(spec.high), val)))

        elif spec.gene_type == "float":
            sigma = CFG.GAUSSIAN_SIGMA * (spec.high - spec.low)
            val = individual[i] + random.gauss(0, sigma)
            individual[i] = max(spec.low, min(spec.high, val))

        elif spec.gene_type == "log_float":
            log_val = math.log10(max(individual[i], spec.low))
            log_range = math.log10(spec.high) - math.log10(spec.low)
            sigma = CFG.GAUSSIAN_SIGMA * log_range
            new_log = log_val + random.gauss(0, sigma)
            new_log = max(math.log10(spec.low), min(math.log10(spec.high), new_log))
            individual[i] = 10 ** new_log

    individual[:] = repair(individual, net_type)
    return (individual,)


if __name__ == "__main__":
    from search_space.mlp_space import random_mlp_genome
    import numpy as np
    rng = np.random.default_rng(42)
    p1 = random_mlp_genome(rng)
    p2 = random_mlp_genome(rng)
    c1, c2 = cx_two_point_typed(list(p1), list(p2), "mlp")
    print(f"Parent1: {p1[:5]}")
    print(f"Child1:  {c1[:5]}")
    m, = mut_mixed_type(list(p1), "mlp", indpb=1.0)
    print(f"Mutated: {m[:5]}")
