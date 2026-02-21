"""Tests for the PNAS-style architecture predictor."""

import pytest
import torch

from src.genome import Genome, LayerGene
from src.predictor import (
    ArchPredictor,
    PredictorTrainer,
    _GENE_DIM,
    _MAX_LAYERS,
    encode_gene,
    encode_genome,
)
from src.search_space import SearchSpace


# ── helpers ──────────────────────────────────────────────────────────────────

def _default_space() -> SearchSpace:
    return SearchSpace({
        "min_depth": 3,
        "max_depth": 6,
        "layer_types": ["conv2d", "maxpool", "batchnorm", "dropout", "dense"],
        "conv": {"kernel_sizes": [3], "filters": [32, 64], "activations": ["relu"]},
        "dense": {"units": [64, 128], "activations": ["relu"]},
        "skip_connections": {"enabled": False},
    })


def _make_genome(depth: int = 4, fitness: float = 0.5) -> Genome:
    """Create a small genome for testing."""
    space = _default_space()
    g = space.random_genome(generation=0)
    g.fitness = fitness
    return g


# ── encode tests ─────────────────────────────────────────────────────────────

class TestEncoding:
    def test_encode_gene_length(self):
        gene = LayerGene("conv2d", {"filters": 64, "kernel_size": 3, "activation": "relu"})
        vec = encode_gene(gene)
        assert len(vec) == _GENE_DIM

    def test_encode_gene_conv_onehot(self):
        gene = LayerGene("conv2d", {"filters": 64, "kernel_size": 3, "activation": "relu"})
        vec = encode_gene(gene)
        # conv2d is index 0
        assert vec[0] == 1.0
        assert sum(vec[:6]) == 1.0, "Exactly one type should be hot"

    def test_encode_gene_dense_onehot(self):
        gene = LayerGene("dense", {"units": 128, "activation": "relu"})
        vec = encode_gene(gene)
        # dense is index 5
        assert vec[5] == 1.0

    def test_encode_gene_normalised_values(self):
        gene = LayerGene("conv2d", {"filters": 128, "kernel_size": 7, "activation": "relu"})
        vec = encode_gene(gene)
        assert abs(vec[6] - 1.0) < 1e-5, "filters/128 should be 1.0"
        assert abs(vec[7] - 1.0) < 1e-5, "kernel_size/7 should be 1.0"

    def test_encode_genome_shape(self):
        g = _make_genome()
        t = encode_genome(g)
        assert t.shape == (_MAX_LAYERS, _GENE_DIM)

    def test_encode_genome_padding(self):
        g = _make_genome(depth=4)
        t = encode_genome(g)
        # Rows beyond actual depth should be all zeros (padding)
        actual_depth = len(g.layers)
        for i in range(actual_depth, _MAX_LAYERS):
            assert t[i].sum().item() == 0.0


# ── ArchPredictor model tests ───────────────────────────────────────────────

class TestArchPredictor:
    def test_forward_shape(self):
        model = ArchPredictor(hidden=32)
        x = torch.randn(3, _MAX_LAYERS, _GENE_DIM)
        out = model(x)
        assert out.shape == (3,)

    def test_output_range(self):
        """Sigmoid output should be in [0, 1]."""
        model = ArchPredictor(hidden=32)
        x = torch.randn(5, _MAX_LAYERS, _GENE_DIM)
        out = model(x)
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_gradient_flow(self):
        model = ArchPredictor(hidden=32)
        x = torch.randn(2, _MAX_LAYERS, _GENE_DIM, requires_grad=True)
        out = model(x)
        out.sum().backward()
        assert x.grad is not None


# ── PredictorTrainer tests ───────────────────────────────────────────────────

class TestPredictorTrainer:
    def test_add_observations(self):
        trainer = PredictorTrainer(hidden_size=32)
        genomes = [_make_genome(fitness=f) for f in [0.5, 0.6, 0.7]]
        trainer.add_observations(genomes)
        assert len(trainer._history) == 3

    def test_add_observations_skips_none_fitness(self):
        trainer = PredictorTrainer(hidden_size=32)
        g = _make_genome()
        g.fitness = None
        trainer.add_observations([g])
        assert len(trainer._history) == 0

    def test_fit_requires_minimum_samples(self):
        trainer = PredictorTrainer(hidden_size=32)
        genomes = [_make_genome(fitness=f) for f in [0.5, 0.6]]
        trainer.add_observations(genomes)
        loss = trainer.fit(epochs=5)
        assert loss == float("inf"), "Should refuse to train with < 5 samples"

    def test_fit_produces_finite_loss(self):
        trainer = PredictorTrainer(hidden_size=32)
        genomes = [_make_genome(fitness=f * 0.1) for f in range(1, 11)]
        trainer.add_observations(genomes)
        loss = trainer.fit(epochs=10)
        assert loss < float("inf"), "Should produce a finite loss"
        assert loss >= 0.0

    def test_predict_returns_list(self):
        trainer = PredictorTrainer(hidden_size=32)
        genomes = [_make_genome() for _ in range(3)]
        preds = trainer.predict(genomes)
        assert isinstance(preds, list)
        assert len(preds) == 3

    def test_rank_and_filter(self):
        trainer = PredictorTrainer(hidden_size=32)
        genomes = [_make_genome() for _ in range(6)]
        filtered = trainer.rank_and_filter(genomes, top_k=3)
        assert len(filtered) == 3
        # Each returned genome should be from the original list
        for g in filtered:
            assert any(g.id == orig.id for orig in genomes)
