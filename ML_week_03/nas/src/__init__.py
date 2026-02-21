"""NAS library — evolutionary neural architecture search for CIFAR-10."""

from src.genome import LayerGene, Genome
from src.search_space import SearchSpace
from src.builder import build_model
from src.evolution import (
    tournament_select,
    crossover,
    mutate,
)

__all__ = [
    "LayerGene",
    "Genome",
    "SearchSpace",
    "build_model",
    "tournament_select",
    "crossover",
    "mutate",
]
