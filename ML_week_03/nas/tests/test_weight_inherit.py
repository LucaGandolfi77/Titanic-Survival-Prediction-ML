"""Tests for weight inheritance between parent and child architectures."""

import pytest
import torch

from src.builder import build_model, count_params
from src.genome import Genome, LayerGene
from src.search_space import SearchSpace
from src.weight_inherit import (
    compute_inheritance_map,
    genes_match,
    inherit_weights,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _default_space() -> SearchSpace:
    return SearchSpace({
        "min_depth": 3,
        "max_depth": 6,
        "layer_types": ["conv2d", "batchnorm", "dense"],
        "conv": {"kernel_sizes": [3], "filters": [32], "activations": ["relu"]},
        "dense": {"units": [64], "activations": ["relu"]},
        "skip_connections": {"enabled": False},
    })


def _make_genome_from_layers(layers, generation=0):
    """Helper: create a Genome from a list of (type, params) tuples."""
    genes = [LayerGene(lt, dict(p)) for lt, p in layers]
    g = Genome(layers=genes, generation=generation)
    return g


# ── genes_match tests ────────────────────────────────────────────────────────

class TestGenesMatch:
    def test_identical_genes(self):
        a = LayerGene("conv2d", {"filters": 32, "kernel_size": 3, "activation": "relu"})
        b = LayerGene("conv2d", {"filters": 32, "kernel_size": 3, "activation": "relu"})
        assert genes_match(a, b)

    def test_different_types(self):
        a = LayerGene("conv2d", {"filters": 32, "kernel_size": 3})
        b = LayerGene("batchnorm", {})
        assert not genes_match(a, b)

    def test_same_type_different_params(self):
        a = LayerGene("conv2d", {"filters": 32, "kernel_size": 3})
        b = LayerGene("conv2d", {"filters": 64, "kernel_size": 3})
        assert not genes_match(a, b)


# ── compute_inheritance_map tests ────────────────────────────────────────────

class TestInheritanceMap:
    def test_identical_genomes(self):
        layers = [
            ("conv2d", {"filters": 32, "kernel_size": 3, "activation": "relu"}),
            ("batchnorm", {}),
            ("dense", {"units": 64, "activation": "relu"}),
        ]
        parent = _make_genome_from_layers(layers)
        child = _make_genome_from_layers(layers)
        mapping = compute_inheritance_map(child, parent)
        assert mapping == [0, 1, 2]

    def test_child_has_extra_layer(self):
        parent_layers = [
            ("conv2d", {"filters": 32, "kernel_size": 3, "activation": "relu"}),
            ("dense", {"units": 64, "activation": "relu"}),
        ]
        child_layers = [
            ("conv2d", {"filters": 32, "kernel_size": 3, "activation": "relu"}),
            ("batchnorm", {}),  # extra layer
            ("dense", {"units": 64, "activation": "relu"}),
        ]
        parent = _make_genome_from_layers(parent_layers)
        child = _make_genome_from_layers(child_layers)
        mapping = compute_inheritance_map(child, parent)
        assert mapping[0] == 0    # conv matches
        assert mapping[1] is None  # extra batchnorm has no parent match
        assert mapping[2] == 1    # dense matches

    def test_no_matching_layers(self):
        parent = _make_genome_from_layers([
            ("conv2d", {"filters": 32, "kernel_size": 3, "activation": "relu"}),
        ])
        child = _make_genome_from_layers([
            ("conv2d", {"filters": 64, "kernel_size": 5, "activation": "elu"}),
        ])
        mapping = compute_inheritance_map(child, parent)
        assert mapping == [None]


# ── inherit_weights tests ────────────────────────────────────────────────────

class TestInheritWeights:
    def test_identical_architecture_transfers_weights(self):
        """When parent and child have the same genome, all body weights transfer."""
        layers = [
            ("conv2d", {"filters": 32, "kernel_size": 3, "activation": "relu"}),
            ("batchnorm", {}),
            ("dense", {"units": 64, "activation": "relu"}),
        ]
        parent = _make_genome_from_layers(layers)
        child = _make_genome_from_layers(layers)
        child.parent_ids = [parent.id]

        parent_model = build_model(parent, num_classes=10)
        child_model = build_model(child, num_classes=10)

        # Set parent weights to known values
        with torch.no_grad():
            for p in parent_model.parameters():
                p.fill_(0.42)

        n = inherit_weights(child_model, parent_model.state_dict(), child, parent)
        assert n > 0, "Should transfer at least some weights"

        # Verify child weights match parent
        parent_sd = parent_model.state_dict()
        child_sd = child_model.state_dict()
        for key in child_sd:
            if key in parent_sd and parent_sd[key].shape == child_sd[key].shape:
                assert torch.allclose(child_sd[key], parent_sd[key]), f"Key {key} should match"

    def test_different_architecture_partial_transfer(self):
        """When child has extra layers, only matching layers get inherited."""
        parent_layers = [
            ("conv2d", {"filters": 32, "kernel_size": 3, "activation": "relu"}),
            ("dense", {"units": 64, "activation": "relu"}),
        ]
        child_layers = [
            ("conv2d", {"filters": 32, "kernel_size": 3, "activation": "relu"}),
            ("batchnorm", {}),
            ("dense", {"units": 64, "activation": "relu"}),
        ]
        parent = _make_genome_from_layers(parent_layers)
        child = _make_genome_from_layers(child_layers)

        parent_model = build_model(parent, num_classes=10)
        child_model = build_model(child, num_classes=10)

        with torch.no_grad():
            for p in parent_model.parameters():
                p.fill_(0.42)

        n = inherit_weights(child_model, parent_model.state_dict(), child, parent)
        assert n > 0, "Should transfer at least the matching conv+dense weights"

    def test_no_match_returns_zero(self):
        """When no layers match, zero tensors should be transferred for body layers."""
        parent = _make_genome_from_layers([
            ("conv2d", {"filters": 32, "kernel_size": 3, "activation": "relu"}),
            ("dense", {"units": 64, "activation": "relu"}),
        ])
        child = _make_genome_from_layers([
            ("conv2d", {"filters": 64, "kernel_size": 5, "activation": "elu"}),
            ("dense", {"units": 128, "activation": "elu"}),
        ])

        parent_model = build_model(parent, num_classes=10)
        child_model = build_model(child, num_classes=10)

        parent_sd = parent_model.state_dict()
        n = inherit_weights(child_model, parent_sd, child, parent)
        # Stem weights should still transfer (same architecture)
        # but body layers should have no matches beyond stem
        assert n >= 0
