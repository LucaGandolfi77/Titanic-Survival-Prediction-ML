"""Tests for DynamicCNN builder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import torch
from models.cnn_builder import build_cnn, DynamicCNN
from search_space.cnn_space import random_cnn_genome
from search_space.genome_encoder import decode, repair_cnn
from models.model_utils import count_parameters

rng = np.random.default_rng(42)


class TestCNNBuilder:
    def _make_config(self):
        g = random_cnn_genome(rng)
        g = repair_cnn(g)
        return decode(g, "cnn")

    def test_build_returns_module(self):
        cfg = self._make_config()
        model = build_cnn(cfg, in_channels=1, num_classes=10)
        assert isinstance(model, DynamicCNN)

    def test_forward_shape_1ch(self):
        cfg = self._make_config()
        model = build_cnn(cfg, in_channels=1, num_classes=10)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, 10)

    def test_forward_shape_3ch(self):
        cfg = self._make_config()
        model = build_cnn(cfg, in_channels=3, num_classes=100)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 100)

    def test_parameter_count_positive(self):
        cfg = self._make_config()
        model = build_cnn(cfg, in_channels=1, num_classes=10)
        assert count_parameters(model) > 0

    def test_multiple_configs(self):
        for _ in range(10):
            cfg = self._make_config()
            model = build_cnn(cfg, in_channels=3, num_classes=10)
            x = torch.randn(2, 3, 32, 32)
            out = model(x)
            assert out.shape == (2, 10)

    def test_minimal_cnn(self):
        cfg = {
            "n_conv_blocks": 1,
            "filters": [16],
            "kernel_size": 3,
            "use_depthwise": False,
            "use_skip_conn": False,
            "pooling_type": "max",
            "activation": "relu",
            "dropout_rate": 0.0,
            "use_batch_norm": True,
            "dense_layers": 1,
            "dense_width": 64,
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "batch_size": 32,
        }
        model = build_cnn(cfg, in_channels=1, num_classes=10)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, 10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
