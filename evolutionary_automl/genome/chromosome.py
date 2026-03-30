"""
Chromosome encoding and decoding.

Each individual in the evolutionary population is a fixed-length list of
13 floats in [0, 1]. This module provides helper functions for creating,
hashing, and converting chromosomes.
"""
from __future__ import annotations

import hashlib
from typing import List, Tuple

import numpy as np

from ..config import CFG
from ..search_space.pipeline_builder import build_pipeline, describe_pipeline
from ..search_space.validators import repair_chromosome

CHROMOSOME_LENGTH = CFG.CHROMOSOME_LENGTH


def random_chromosome(rng: np.random.Generator) -> List[float]:
    """Generate a single random chromosome using the provided RNG."""
    return rng.uniform(0.0, 1.0, size=CHROMOSOME_LENGTH).tolist()


def chromosome_hash(chromosome: List[float], dataset_name: str = "") -> str:
    """Compute a deterministic hash for a chromosome + dataset pair.

    Used by the fitness cache to avoid redundant evaluations.
    Genes are rounded to 6 decimals to absorb floating-point noise.
    """
    rounded = tuple(round(g, 6) for g in chromosome)
    key = f"{dataset_name}:{rounded}"
    return hashlib.md5(key.encode()).hexdigest()


def chromosome_to_pipeline(
    chromosome: List[float], n_features: int, random_state: int = 0
):
    """Convert a chromosome to a sklearn Pipeline (delegates to pipeline_builder)."""
    repaired = repair_chromosome(list(chromosome))
    return build_pipeline(repaired, n_features, random_state)


def chromosome_description(chromosome: List[float], n_features: int) -> str:
    """Return a human-readable pipeline description."""
    repaired = repair_chromosome(list(chromosome))
    return describe_pipeline(repaired, n_features)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    chrom = random_chromosome(rng)
    print("Chromosome:", [f"{g:.3f}" for g in chrom])
    print("Hash:", chromosome_hash(chrom, "iris"))
    print("Description:", chromosome_description(chrom, 30))
