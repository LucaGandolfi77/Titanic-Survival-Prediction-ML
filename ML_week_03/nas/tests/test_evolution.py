"""Tests for evolutionary operators."""

import random

import pytest

from src.evolution import (
    _filter_skips,
    crossover,
    mutate,
    select_parents,
    tournament_select,
)
from src.genome import Genome, LayerGene
from src.search_space import SearchSpace


@pytest.fixture
def space():
    return SearchSpace({
        "min_depth": 3,
        "max_depth": 10,
        "layer_types": ["conv2d", "maxpool", "batchnorm", "dropout", "dense"],
        "conv": {"kernel_sizes": [3, 5], "filters": [32, 64], "activations": ["relu"]},
        "pool": {"size": 2},
        "dropout": {"rates": [0.2, 0.3]},
        "dense": {"units": [64, 128], "activations": ["relu"]},
        "skip_connections": {"enabled": True, "max_span": 3},
    })


@pytest.fixture
def population(space):
    pop = []
    for i in range(10):
        g = space.random_genome(generation=0)
        g.fitness = random.uniform(0.3, 0.9)
        pop.append(g)
    return pop


class TestSelection:
    def test_tournament_returns_genome(self, population):
        winner = tournament_select(population, k=3)
        assert isinstance(winner, Genome)
        assert winner.fitness is not None

    def test_tournament_favours_fitter(self, population):
        # Over many trials the winner should have above-median fitness
        wins = [tournament_select(population, k=5).fitness for _ in range(100)]
        median_pop = sorted(g.fitness for g in population)[len(population) // 2]
        avg_win = sum(wins) / len(wins)
        assert avg_win > median_pop

    def test_select_parents_count(self, population):
        parents = select_parents(population, n_parents=6, tournament_size=3)
        assert len(parents) == 6


class TestCrossover:
    def test_produces_two_children(self, population, space):
        p1, p2 = population[0], population[1]
        c1, c2 = crossover(p1, p2, generation=1, space=space)
        assert isinstance(c1, Genome)
        assert isinstance(c2, Genome)
        assert c1.generation == 1
        assert c2.parent_ids == (p1.id, p2.id)

    def test_children_valid_structure(self, population, space):
        for _ in range(20):
            p1, p2 = random.sample(population, 2)
            c1, c2 = crossover(p1, p2, generation=1, space=space)
            # Must start with conv
            assert c1.layers[0].layer_type == "conv2d"
            assert c2.layers[0].layer_type == "conv2d"
            # Depth within bounds (may exceed slightly due to fixes)
            assert c1.depth <= space.max_depth + 3
            assert c2.depth <= space.max_depth + 3


class TestMutation:
    def test_mutate_returns_genome(self, space):
        g = space.random_genome()
        g.fitness = 0.5
        mutation_cfg = {
            "rate": 1.0,  # always mutate
            "add_layer_prob": 0.25,
            "remove_layer_prob": 0.20,
            "change_param_prob": 0.35,
            "toggle_skip_prob": 0.20,
        }
        result = mutate(g, space, mutation_cfg)
        assert isinstance(result, Genome)

    def test_mutation_changes_genome_over_time(self, space):
        g = space.random_genome()
        g.fitness = 0.5
        original_depth = g.depth
        original_layers = [l.to_dict() for l in g.layers]
        mutation_cfg = {"rate": 1.0, "add_layer_prob": 0.5, "remove_layer_prob": 0.1, "change_param_prob": 0.3, "toggle_skip_prob": 0.1}

        changed = False
        for _ in range(50):
            g = mutate(g, space, mutation_cfg)
            current = [l.to_dict() for l in g.layers]
            if current != original_layers or g.depth != original_depth:
                changed = True
                break
        assert changed

    def test_no_mutation_at_zero_rate(self, space):
        g = space.random_genome()
        original = [l.to_dict() for l in g.layers]
        result = mutate(g, space, {"rate": 0.0})
        assert [l.to_dict() for l in result.layers] == original


class TestFilterSkips:
    def test_out_of_bounds_removed(self):
        skips = [(0, 2), (0, 10), (3, 5)]
        filtered = _filter_skips(skips, depth=5)
        assert (0, 2) in filtered
        assert (3, 4) not in filtered  # (3,5) is out of bounds (depth=5 → max idx=4)
        assert (0, 10) not in filtered
