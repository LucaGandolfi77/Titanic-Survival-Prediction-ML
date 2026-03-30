"""
Thread-safe fitness cache.

Caches fitness evaluations keyed by chromosome hash + dataset name.
Avoids redundant evaluations of identical or near-identical individuals
across generations.
"""
from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

from ..genome.chromosome import chromosome_hash


class FitnessCache:
    """Thread-safe cache for fitness evaluation results."""

    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[float, ...]] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(
        self, chromosome: list, dataset_name: str
    ) -> Optional[Tuple[float, ...]]:
        """Look up cached fitness. Returns None on miss."""
        key = chromosome_hash(chromosome, dataset_name)
        with self._lock:
            result = self._cache.get(key)
            if result is not None:
                self._hits += 1
            else:
                self._misses += 1
            return result

    def put(
        self, chromosome: list, dataset_name: str, fitness: Tuple[float, ...]
    ) -> None:
        """Store a fitness result in the cache."""
        key = chromosome_hash(chromosome, dataset_name)
        with self._lock:
            self._cache[key] = fitness

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    @property
    def hit_rate(self) -> float:
        with self._lock:
            total = self._hits + self._misses
            return self._hits / total if total > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"FitnessCache(size={self.size}, "
            f"hits={self._hits}, misses={self._misses}, "
            f"hit_rate={self.hit_rate:.2%})"
        )


if __name__ == "__main__":
    cache = FitnessCache()
    chrom = [0.5] * 13
    print(cache.get(chrom, "iris"))
    cache.put(chrom, "iris", (0.95, 1.2))
    print(cache.get(chrom, "iris"))
    print(cache)
