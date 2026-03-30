"""Tests for chromosome encoding, decoding, and hashing."""
from __future__ import annotations

import numpy as np
import pytest

from evolutionary_automl.genome.chromosome import (
    CHROMOSOME_LENGTH,
    chromosome_description,
    chromosome_hash,
    chromosome_to_pipeline,
    random_chromosome,
)


class TestRandomChromosome:
    def test_length(self):
        rng = np.random.default_rng(42)
        chrom = random_chromosome(rng)
        assert len(chrom) == CHROMOSOME_LENGTH

    def test_bounds(self):
        rng = np.random.default_rng(7)
        for _ in range(50):
            chrom = random_chromosome(rng)
            for g in chrom:
                assert 0.0 <= g <= 1.0

    def test_reproducibility(self):
        c1 = random_chromosome(np.random.default_rng(42))
        c2 = random_chromosome(np.random.default_rng(42))
        assert c1 == c2


class TestChromosomeHash:
    def test_deterministic(self):
        chrom = [0.5] * 13
        h1 = chromosome_hash(chrom, "iris")
        h2 = chromosome_hash(chrom, "iris")
        assert h1 == h2

    def test_dataset_aware(self):
        chrom = [0.5] * 13
        h1 = chromosome_hash(chrom, "iris")
        h2 = chromosome_hash(chrom, "wine")
        assert h1 != h2

    def test_different_chromosomes(self):
        c1 = [0.5] * 13
        c2 = [0.6] * 13
        assert chromosome_hash(c1, "iris") != chromosome_hash(c2, "iris")


class TestChromosomeToPipeline:
    def test_returns_pipeline(self):
        chrom = [0.33, 0.0, 0.5, 0.0, 0.17, 0.5, 0.27, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5]
        pipe = chromosome_to_pipeline(chrom, n_features=30)
        assert hasattr(pipe, "fit")
        assert hasattr(pipe, "predict")

    def test_all_classifier_types(self):
        for clf_gene in [0.0, 0.17, 0.33, 0.5, 0.67, 0.83, 1.0]:
            chrom = [0.5, 0.0, 0.5, 0.0, clf_gene] + [0.5] * 8
            pipe = chromosome_to_pipeline(chrom, n_features=10)
            assert pipe.steps[-1][0] == "classifier"


class TestChromosomeDescription:
    def test_returns_string(self):
        chrom = [0.5] * 13
        desc = chromosome_description(chrom, 30)
        assert isinstance(desc, str)
        assert "Clf=" in desc
