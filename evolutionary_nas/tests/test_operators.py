"""Tests for evolutionary operators — crossover and mutation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from search_space.mlp_space import MLP_GENE_SPECS, random_mlp_genome
from search_space.cnn_space import CNN_GENE_SPECS, random_cnn_genome
from search_space.genome_encoder import repair_mlp, repair_cnn
from search_space.constraints import is_valid
from evolution.operators import cx_two_point_typed, mut_mixed_type

rng = np.random.default_rng(42)


class TestCrossover:
    def test_mlp_crossover_lengths(self):
        g1 = repair_mlp(random_mlp_genome(rng))
        g2 = repair_mlp(random_mlp_genome(rng))
        c1, c2 = cx_two_point_typed(list(g1), list(g2), "mlp")
        assert len(c1) == len(MLP_GENE_SPECS)
        assert len(c2) == len(MLP_GENE_SPECS)

    def test_cnn_crossover_lengths(self):
        g1 = repair_cnn(random_cnn_genome(rng))
        g2 = repair_cnn(random_cnn_genome(rng))
        c1, c2 = cx_two_point_typed(list(g1), list(g2), "cnn")
        assert len(c1) == len(CNN_GENE_SPECS)
        assert len(c2) == len(CNN_GENE_SPECS)

    def test_crossover_produces_valid_mlp(self):
        for _ in range(20):
            g1 = repair_mlp(random_mlp_genome(rng))
            g2 = repair_mlp(random_mlp_genome(rng))
            c1, c2 = cx_two_point_typed(list(g1), list(g2), "mlp")
            assert is_valid(c1, "mlp"), f"Invalid child1: {c1}"
            assert is_valid(c2, "mlp"), f"Invalid child2: {c2}"

    def test_crossover_produces_valid_cnn(self):
        for _ in range(20):
            g1 = repair_cnn(random_cnn_genome(rng))
            g2 = repair_cnn(random_cnn_genome(rng))
            c1, c2 = cx_two_point_typed(list(g1), list(g2), "cnn")
            assert is_valid(c1, "cnn"), f"Invalid child1: {c1}"
            assert is_valid(c2, "cnn"), f"Invalid child2: {c2}"


class TestMutation:
    def test_mlp_mutation_length(self):
        g = repair_mlp(random_mlp_genome(rng))
        (m,) = mut_mixed_type(list(g), "mlp", indpb=1.0)
        assert len(m) == len(MLP_GENE_SPECS)

    def test_cnn_mutation_length(self):
        g = repair_cnn(random_cnn_genome(rng))
        (m,) = mut_mixed_type(list(g), "cnn", indpb=1.0)
        assert len(m) == len(CNN_GENE_SPECS)

    def test_mutation_produces_valid_mlp(self):
        for _ in range(20):
            g = repair_mlp(random_mlp_genome(rng))
            (m,) = mut_mixed_type(list(g), "mlp", indpb=0.3)
            assert is_valid(m, "mlp"), f"Invalid mutant: {m}"

    def test_mutation_produces_valid_cnn(self):
        for _ in range(20):
            g = repair_cnn(random_cnn_genome(rng))
            (m,) = mut_mixed_type(list(g), "cnn", indpb=0.3)
            assert is_valid(m, "cnn"), f"Invalid mutant: {m}"

    def test_high_indpb_changes_genes(self):
        g = repair_mlp(random_mlp_genome(rng))
        (m,) = mut_mixed_type(list(g), "mlp", indpb=1.0)
        # With indpb=1.0 every gene should be mutated — at least some differ
        n_changed = sum(1 for a, b in zip(g, m) if a != b)
        assert n_changed > 0

    def test_zero_indpb_no_change(self):
        g = repair_mlp(random_mlp_genome(rng))
        (m,) = mut_mixed_type(list(g), "mlp", indpb=0.0)
        assert g == m


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
