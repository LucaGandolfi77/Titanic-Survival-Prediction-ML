"""Tests for genome encoder — encode, decode, repair, hash."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from search_space.mlp_space import MLP_GENE_SPECS, random_mlp_genome
from search_space.cnn_space import CNN_GENE_SPECS, random_cnn_genome
from search_space.genome_encoder import (
    encode, decode, repair_mlp, repair_cnn, hash_genome, describe,
)
from search_space.constraints import is_valid

rng = np.random.default_rng(42)


class TestMLPGenome:
    def test_random_length(self):
        g = random_mlp_genome(rng)
        assert len(g) == len(MLP_GENE_SPECS)

    def test_encode_decode_roundtrip(self):
        g = random_mlp_genome(rng)
        cfg = decode(g, "mlp")
        assert "n_layers" in cfg
        assert "hidden_sizes" in cfg
        assert isinstance(cfg["hidden_sizes"], list)

    def test_repair_clamps_values(self):
        g = random_mlp_genome(rng)
        # Force out-of-range
        g[0] = 99.0  # n_layers
        repaired = repair_mlp(g)
        spec = MLP_GENE_SPECS[0]
        assert spec.low <= repaired[0] <= spec.high

    def test_repair_idempotent(self):
        g = random_mlp_genome(rng)
        r1 = repair_mlp(g)
        r2 = repair_mlp(r1)
        assert r1 == r2

    def test_hash_deterministic(self):
        g = random_mlp_genome(rng)
        h1 = hash_genome(g, "MNIST")
        h2 = hash_genome(g, "MNIST")
        assert h1 == h2

    def test_hash_differs_by_dataset(self):
        g = random_mlp_genome(rng)
        h1 = hash_genome(g, "MNIST")
        h2 = hash_genome(g, "CIFAR10")
        assert h1 != h2

    def test_describe_string(self):
        g = random_mlp_genome(rng)
        d = describe(g, "mlp")
        assert isinstance(d, str)
        assert len(d) > 0


class TestCNNGenome:
    def test_random_length(self):
        g = random_cnn_genome(rng)
        assert len(g) == len(CNN_GENE_SPECS)

    def test_encode_decode_roundtrip(self):
        g = random_cnn_genome(rng)
        cfg = decode(g, "cnn")
        assert "n_conv_blocks" in cfg
        assert "filters" in cfg
        assert "kernel_size" in cfg

    def test_repair_clamps(self):
        g = random_cnn_genome(rng)
        g[0] = -5.0
        repaired = repair_cnn(g)
        spec = CNN_GENE_SPECS[0]
        assert spec.low <= repaired[0] <= spec.high

    def test_repair_idempotent(self):
        g = random_cnn_genome(rng)
        r1 = repair_cnn(g)
        r2 = repair_cnn(r1)
        assert r1 == r2


class TestConstraints:
    def test_valid_mlp(self):
        for _ in range(20):
            g = random_mlp_genome(rng)
            g = repair_mlp(g)
            assert is_valid(g, "mlp")

    def test_valid_cnn(self):
        for _ in range(20):
            g = random_cnn_genome(rng)
            g = repair_cnn(g)
            assert is_valid(g, "cnn")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
