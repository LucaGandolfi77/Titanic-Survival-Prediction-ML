"""Tests for the model builder (genome → PyTorch module)."""

import pytest
import torch

from src.builder import NASModel, build_model, count_params
from src.genome import Genome, LayerGene


# ── helpers ──────────────────────────────────────────────────────────────────

def _simple_genome() -> Genome:
    """Minimal valid genome: conv → pool → conv → dense."""
    return Genome(layers=[
        LayerGene("conv2d", {"filters": 32, "kernel_size": 3, "activation": "relu"}),
        LayerGene("maxpool", {"size": 2}),
        LayerGene("conv2d", {"filters": 64, "kernel_size": 3, "activation": "relu"}),
        LayerGene("dense", {"units": 128, "activation": "relu"}),
    ])


def _conv_only_genome() -> Genome:
    """Genome with no dense layers (head should add global avg pool)."""
    return Genome(layers=[
        LayerGene("conv2d", {"filters": 32, "kernel_size": 3, "activation": "relu"}),
        LayerGene("batchnorm", {}),
        LayerGene("conv2d", {"filters": 64, "kernel_size": 5, "activation": "elu"}),
        LayerGene("maxpool", {"size": 2}),
        LayerGene("dropout", {"rate": 0.3}),
    ])


def _skip_genome() -> Genome:
    """Genome with a skip connection."""
    return Genome(
        layers=[
            LayerGene("conv2d", {"filters": 32, "kernel_size": 3, "activation": "relu"}),
            LayerGene("conv2d", {"filters": 64, "kernel_size": 3, "activation": "relu"}),
            LayerGene("conv2d", {"filters": 64, "kernel_size": 3, "activation": "relu"}),
        ],
        skip_connections=[(0, 2)],
    )


# ── tests ────────────────────────────────────────────────────────────────────

class TestBuilder:
    def test_simple_forward(self):
        model = build_model(_simple_genome())
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_conv_only_forward(self):
        model = build_model(_conv_only_genome())
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_skip_forward(self):
        model = build_model(_skip_genome())
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_count_params_positive(self):
        model = build_model(_simple_genome())
        assert count_params(model) > 0

    def test_different_genomes_different_params(self):
        m1 = build_model(_simple_genome())
        m2 = build_model(_conv_only_genome())
        assert count_params(m1) != count_params(m2)

    def test_gradients_flow(self):
        model = build_model(_simple_genome())
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # Check that at least the stem conv has gradients
        assert model.stem[0].weight.grad is not None
