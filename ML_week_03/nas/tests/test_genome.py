"""Tests for genome encoding and search-space sampling."""

import json
import pytest

from src.genome import Genome, LayerGene
from src.search_space import SearchSpace


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def space_cfg():
    return {
        "min_depth": 3,
        "max_depth": 8,
        "layer_types": ["conv2d", "maxpool", "batchnorm", "dropout", "dense"],
        "conv": {"kernel_sizes": [3, 5], "filters": [32, 64], "activations": ["relu"]},
        "pool": {"size": 2},
        "dropout": {"rates": [0.2, 0.3]},
        "dense": {"units": [64, 128], "activations": ["relu"]},
        "skip_connections": {"enabled": True, "max_span": 3},
    }


@pytest.fixture
def space(space_cfg):
    return SearchSpace(space_cfg)


# ── LayerGene ────────────────────────────────────────────────────────────────

class TestLayerGene:
    def test_roundtrip_dict(self):
        gene = LayerGene("conv2d", {"filters": 64, "kernel_size": 3, "activation": "relu"})
        d = gene.to_dict()
        restored = LayerGene.from_dict(d)
        assert restored.layer_type == "conv2d"
        assert restored.params["filters"] == 64

    def test_repr(self):
        gene = LayerGene("dropout", {"rate": 0.3})
        assert "dropout" in repr(gene)


# ── Genome ───────────────────────────────────────────────────────────────────

class TestGenome:
    def test_depth(self):
        g = Genome(layers=[LayerGene("conv2d", {"filters": 32})] * 5)
        assert g.depth == 5

    def test_clone_has_new_id(self):
        g = Genome(layers=[LayerGene("conv2d", {"filters": 32})])
        g.fitness = 0.85
        c = g.clone()
        assert c.id != g.id
        assert c.fitness is None

    def test_json_roundtrip(self, tmp_path):
        g = Genome(
            layers=[
                LayerGene("conv2d", {"filters": 64, "kernel_size": 3}),
                LayerGene("maxpool", {"size": 2}),
                LayerGene("dense", {"units": 128}),
            ],
            skip_connections=[(0, 2)],
            fitness=0.75,
            generation=3,
        )
        path = tmp_path / "genome.json"
        g.save(path)
        loaded = Genome.load(path)
        assert loaded.depth == 3
        assert loaded.fitness == 0.75
        assert loaded.skip_connections == [(0, 2)]

    def test_summary(self):
        g = Genome(layers=[LayerGene("conv2d")] * 4, fitness=0.9)
        s = g.summary()
        assert "depth=4" in s
        assert "0.9" in s


# ── SearchSpace ──────────────────────────────────────────────────────────────

class TestSearchSpace:
    def test_random_genome_depth(self, space):
        for _ in range(20):
            g = space.random_genome()
            assert space.min_depth <= g.depth <= space.max_depth + 3  # +3 for structural fixes

    def test_random_genome_starts_with_conv(self, space):
        for _ in range(20):
            g = space.random_genome()
            assert g.layers[0].layer_type == "conv2d"

    def test_no_consecutive_pools(self, space):
        for _ in range(20):
            g = space.random_genome()
            for i in range(len(g.layers) - 1):
                pool_types = ("maxpool", "avgpool")
                if g.layers[i].layer_type in pool_types:
                    assert g.layers[i + 1].layer_type not in pool_types

    def test_random_layer_conv(self, space):
        gene = space.random_layer("conv2d")
        assert gene.layer_type == "conv2d"
        assert gene.params["filters"] in [32, 64]
        assert gene.params["kernel_size"] in [3, 5]

    def test_mutate_layer_changes_something(self, space):
        original = LayerGene("conv2d", {"filters": 32, "kernel_size": 3, "activation": "relu"})
        changed = False
        for _ in range(50):
            mutated = space.mutate_layer(original)
            if mutated.params != original.params:
                changed = True
                break
        assert changed, "Mutation should eventually change at least one param"
