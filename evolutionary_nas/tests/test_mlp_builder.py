"""Tests for DynamicMLP builder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import torch
from models.mlp_builder import build_mlp, DynamicMLP
from search_space.mlp_space import random_mlp_genome
from search_space.genome_encoder import decode, repair_mlp
from models.model_utils import count_parameters

rng = np.random.default_rng(42)


class TestMLPBuilder:
    def _make_config(self):
        g = random_mlp_genome(rng)
        g = repair_mlp(g)
        return decode(g, "mlp")

    def test_build_returns_module(self):
        cfg = self._make_config()
        model = build_mlp(cfg, input_dim=784, num_classes=10)
        assert isinstance(model, DynamicMLP)

    def test_forward_shape(self):
        cfg = self._make_config()
        model = build_mlp(cfg, input_dim=784, num_classes=10)
        x = torch.randn(4, 784)
        out = model(x)
        assert out.shape == (4, 10)

    def test_parameter_count_positive(self):
        cfg = self._make_config()
        model = build_mlp(cfg, input_dim=784, num_classes=10)
        assert count_parameters(model) > 0

    def test_multiple_configs(self):
        for _ in range(10):
            cfg = self._make_config()
            model = build_mlp(cfg, input_dim=784, num_classes=10)
            x = torch.randn(2, 784)
            out = model(x)
            assert out.shape == (2, 10)

    def test_single_layer(self):
        cfg = {
            "n_layers": 1,
            "hidden_sizes": [64],
            "activation": "relu",
            "dropout_rate": 0.0,
            "use_batch_norm": False,
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 32,
        }
        model = build_mlp(cfg, input_dim=100, num_classes=5)
        x = torch.randn(3, 100)
        out = model(x)
        assert out.shape == (3, 5)

    def test_with_batchnorm_and_dropout(self):
        cfg = {
            "n_layers": 3,
            "hidden_sizes": [128, 64, 32],
            "activation": "relu",
            "dropout_rate": 0.5,
            "use_batch_norm": True,
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "batch_size": 32,
        }
        model = build_mlp(cfg, input_dim=784, num_classes=10)
        model.eval()
        x = torch.randn(4, 784)
        out = model(x)
        assert out.shape == (4, 10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
