"""Tests for the single-architecture trainer (no GPU required)."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.builder import build_model
from src.genome import Genome, LayerGene
from src.trainer import _evaluate, train_genome


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_genome():
    """Minimal genome that trains quickly."""
    return Genome(layers=[
        LayerGene("conv2d", {"filters": 8, "kernel_size": 3, "activation": "relu"}),
        LayerGene("maxpool", {"size": 2}),
    ])


@pytest.fixture
def tiny_loaders():
    """Fake 8×8 images, 3 classes, 64 samples."""
    X = torch.randn(64, 3, 32, 32)
    y = torch.randint(0, 10, (64,))
    ds = TensorDataset(X, y)
    train = DataLoader(ds, batch_size=16, shuffle=True)
    val = DataLoader(ds, batch_size=16)
    return train, val


@pytest.fixture
def cfg():
    return {
        "training": {
            "epochs": 2,
            "batch_size": 16,
            "lr": 0.01,
            "optimizer": "sgd",
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "scheduler": "cosine",
            "early_stop": {"enabled": False},
        },
        "device": "cpu",
    }


# ── tests ────────────────────────────────────────────────────────────────────

class TestTrainer:
    def test_train_returns_result(self, tiny_genome, tiny_loaders, cfg):
        train_loader, val_loader = tiny_loaders
        result = train_genome(tiny_genome, train_loader, val_loader, cfg, device=torch.device("cpu"))
        assert "fitness" in result
        assert "epochs_trained" in result
        assert result["epochs_trained"] == 2
        assert 0.0 <= result["fitness"] <= 1.0

    def test_early_stop_kills_bad_arch(self, tiny_genome, tiny_loaders):
        train_loader, val_loader = tiny_loaders
        cfg = {
            "training": {
                "epochs": 5,
                "batch_size": 16,
                "lr": 0.0001,  # very low lr → slow learning
                "optimizer": "sgd",
                "momentum": 0.0,
                "weight_decay": 0.0,
                "scheduler": "cosine",
                "early_stop": {
                    "enabled": True,
                    "patience_epoch": 1,
                    "min_accuracy": 0.99,  # impossible threshold
                },
            },
            "device": "cpu",
        }
        result = train_genome(tiny_genome, train_loader, val_loader, cfg, device=torch.device("cpu"))
        assert result["early_stopped"] is True
        assert result["epochs_trained"] < 5

    def test_evaluate_function(self, tiny_genome):
        model = build_model(tiny_genome).to("cpu")
        X = torch.randn(32, 3, 32, 32)
        y = torch.randint(0, 10, (32,))
        loader = DataLoader(TensorDataset(X, y), batch_size=16)
        acc = _evaluate(model, loader, torch.device("cpu"))
        assert 0.0 <= acc <= 1.0

    def test_fitness_set_on_genome(self, tiny_genome, tiny_loaders, cfg):
        train_loader, val_loader = tiny_loaders
        train_genome(tiny_genome, train_loader, val_loader, cfg, device=torch.device("cpu"))
        assert tiny_genome.fitness is not None
