"""
Fitness Cache
=============
Thread-safe cache for fitness evaluations, keyed by genome hash + dataset.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Tuple


class FitnessCache:
    """Thread-safe dictionary cache for fitness values."""

    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[float, ...]] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Tuple[float, ...]]:
        with self._lock:
            if key in self._cache:
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: str, fitness: Tuple[float, ...]) -> None:
        with self._lock:
            self._cache[key] = fitness

    def contains(self, key: str) -> bool:
        with self._lock:
            return key in self._cache

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def all_entries(self) -> Dict[str, Tuple[float, ...]]:
        with self._lock:
            return dict(self._cache)


if __name__ == "__main__":
    cache = FitnessCache()
    cache.put("abc", (0.95, 100_000))
    print(f"Get: {cache.get('abc')}")
    print(f"Hit rate: {cache.hit_rate:.2%}")
    cache.get("xyz")
    print(f"Hit rate after miss: {cache.hit_rate:.2%}")
