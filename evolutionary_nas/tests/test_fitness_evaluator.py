"""Tests for fitness evaluator and cache."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fitness.evaluator import FitnessEvaluator
from fitness.cache import FitnessCache
from fitness.metrics import compute_accuracy, compute_f1
import numpy as np
from search_space.mlp_space import random_mlp_genome
from search_space.genome_encoder import repair_mlp

rng = np.random.default_rng(42)


def _make_dummy_loaders(n=100, input_dim=784, num_classes=10):
    X = torch.randn(n, input_dim)
    y = torch.randint(0, num_classes, (n,))
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=32), DataLoader(ds, batch_size=32)


class TestFitnessCache:
    def test_put_get(self):
        cache = FitnessCache()
        cache.put("abc", (0.9, 1000))
        assert cache.get("abc") == (0.9, 1000)

    def test_miss(self):
        cache = FitnessCache()
        assert cache.get("nonexistent") is None

    def test_hits_misses(self):
        cache = FitnessCache()
        cache.put("k", (0.5, 500))
        cache.get("k")
        cache.get("missing")
        assert cache.hits == 1
        assert cache.misses == 1

    def test_contains(self):
        cache = FitnessCache()
        cache.put("k", (0.5, 500))
        assert cache.contains("k")
        assert not cache.contains("other")


class TestFitnessEvaluator:
    def test_evaluate_returns_tuple(self):
        train_loader, val_loader = _make_dummy_loaders()
        evaluator = FitnessEvaluator(
            train_loader=train_loader,
            val_loader=val_loader,
            dataset_name="MNIST",
            net_type="mlp",
            in_channels=1,
            num_classes=10,
            device="cpu",
            fast_epochs=1,
            cache=FitnessCache(),
        )
        g = random_mlp_genome(rng)
        g = repair_mlp(g)
        result = evaluator.evaluate(g)
        assert isinstance(result, tuple)
        assert len(result) == 2
        acc, params = result
        assert 0.0 <= acc <= 1.0
        assert params > 0

    def test_cache_hit(self):
        train_loader, val_loader = _make_dummy_loaders()
        cache = FitnessCache()
        evaluator = FitnessEvaluator(
            train_loader=train_loader,
            val_loader=val_loader,
            dataset_name="MNIST",
            net_type="mlp",
            in_channels=1,
            num_classes=10,
            device="cpu",
            fast_epochs=1,
            cache=cache,
        )
        g = random_mlp_genome(rng)
        g = repair_mlp(g)
        r1 = evaluator.evaluate(g)
        r2 = evaluator.evaluate(g)
        assert r1 == r2
        assert cache.hits >= 1

    def test_single_objective(self):
        train_loader, val_loader = _make_dummy_loaders()
        evaluator = FitnessEvaluator(
            train_loader=train_loader,
            val_loader=val_loader,
            dataset_name="MNIST",
            net_type="mlp",
            in_channels=1,
            num_classes=10,
            device="cpu",
            fast_epochs=1,
            cache=FitnessCache(),
        )
        g = random_mlp_genome(rng)
        g = repair_mlp(g)
        score = evaluator.evaluate_single_objective(g)
        assert isinstance(score, tuple)
        assert len(score) == 1
        assert isinstance(score[0], float)


class TestMetrics:
    def test_compute_accuracy(self):
        import torch.nn as nn
        # Build a simple model that always predicts class 0
        model = nn.Linear(5, 5)
        X = torch.randn(10, 5)
        y = torch.zeros(10, dtype=torch.long)
        ds = TensorDataset(X, y)
        loader = DataLoader(ds, batch_size=5)
        acc = compute_accuracy(model, loader, device="cpu")
        assert 0.0 <= acc <= 1.0

    def test_compute_f1(self):
        import torch.nn as nn
        model = nn.Linear(5, 3)
        X = torch.randn(10, 5)
        y = torch.randint(0, 3, (10,))
        ds = TensorDataset(X, y)
        loader = DataLoader(ds, batch_size=5)
        f1 = compute_f1(model, loader, device="cpu")
        assert 0.0 <= f1 <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
