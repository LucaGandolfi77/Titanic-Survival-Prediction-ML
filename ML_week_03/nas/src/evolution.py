"""Evolutionary operators: selection, crossover, mutation.

These operators work on lists of :class:`Genome` objects and produce the
next generation according to classic GA principles:

1. **Tournament selection** — pick the best out of *k* random candidates.
2. **Single-point crossover** — swap layer segments between two parents.
3. **Mutation** — add/remove layer, tweak hyper-parameter, toggle skip.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Tuple

from src.genome import Genome, LayerGene
from src.search_space import SearchSpace


# ── Selection ────────────────────────────────────────────────────────────────

def tournament_select(
    population: List[Genome],
    k: int = 5,
) -> Genome:
    """Pick *k* random individuals, return the one with highest fitness.

    Genomes without a fitness score (``None``) are treated as fitness 0.
    """
    candidates = random.sample(population, min(k, len(population)))
    return max(candidates, key=lambda g: g.fitness if g.fitness is not None else 0.0)


def select_parents(
    population: List[Genome],
    n_parents: int,
    tournament_size: int = 5,
) -> List[Genome]:
    """Select *n_parents* individuals via repeated tournament selection."""
    return [tournament_select(population, tournament_size) for _ in range(n_parents)]


# ── Crossover ────────────────────────────────────────────────────────────────

def crossover(
    parent_a: Genome,
    parent_b: Genome,
    generation: int,
    space: SearchSpace,
) -> Tuple[Genome, Genome]:
    """Single-point crossover producing two children.

    A random split point is chosen.  Child 1 gets the first segment of
    parent A and the second segment of parent B, and vice versa.

    Skip connections are inherited only if both referenced layer indices
    still exist in the child.
    """
    layers_a = [copy.deepcopy(l) for l in parent_a.layers]
    layers_b = [copy.deepcopy(l) for l in parent_b.layers]

    if len(layers_a) < 2 or len(layers_b) < 2:
        # Can't crossover tiny genomes — just clone
        c1, c2 = parent_a.clone(), parent_b.clone()
        c1.generation = c2.generation = generation
        c1.parent_ids = (parent_a.id, parent_b.id)
        c2.parent_ids = (parent_a.id, parent_b.id)
        return c1, c2

    split_a = random.randint(1, len(layers_a) - 1)
    split_b = random.randint(1, len(layers_b) - 1)

    child1_layers = layers_a[:split_a] + layers_b[split_b:]
    child2_layers = layers_b[:split_b] + layers_a[split_a:]

    # Validate and fix structure
    child1_layers = space._ensure_valid_structure(child1_layers)
    child2_layers = space._ensure_valid_structure(child2_layers)

    # Clamp depth
    child1_layers = child1_layers[: space.max_depth]
    child2_layers = child2_layers[: space.max_depth]

    # Inherit skip connections that are still in-bounds
    c1_skips = _filter_skips(parent_a.skip_connections + parent_b.skip_connections, len(child1_layers))
    c2_skips = _filter_skips(parent_a.skip_connections + parent_b.skip_connections, len(child2_layers))

    child1 = Genome(
        layers=child1_layers,
        skip_connections=c1_skips,
        generation=generation,
        parent_ids=(parent_a.id, parent_b.id),
    )
    child2 = Genome(
        layers=child2_layers,
        skip_connections=c2_skips,
        generation=generation,
        parent_ids=(parent_a.id, parent_b.id),
    )
    return child1, child2


def _filter_skips(
    skips: List[Tuple[int, int]], depth: int
) -> List[Tuple[int, int]]:
    """Keep only skip connections whose indices are in-bounds."""
    seen = set()
    result = []
    for src, dst in skips:
        if src < depth and dst < depth and src < dst and (src, dst) not in seen:
            seen.add((src, dst))
            result.append((src, dst))
    return result


# ── Mutation ─────────────────────────────────────────────────────────────────

def mutate(
    genome: Genome,
    space: SearchSpace,
    mutation_cfg: Dict[str, Any],
) -> Genome:
    """Apply random mutations in-place and return the (possibly modified) genome.

    Mutation types (selected by weighted probability):
    * **add_layer** — insert a random layer at a random position.
    * **remove_layer** — delete a random layer (respecting min depth).
    * **change_param** — pick a layer and tweak one hyper-parameter.
    * **toggle_skip** — add or remove a skip connection.
    """
    rate = mutation_cfg.get("rate", 0.3)
    if random.random() > rate:
        return genome  # no mutation this time

    probs = {
        "add_layer": mutation_cfg.get("add_layer_prob", 0.25),
        "remove_layer": mutation_cfg.get("remove_layer_prob", 0.20),
        "change_param": mutation_cfg.get("change_param_prob", 0.35),
        "toggle_skip": mutation_cfg.get("toggle_skip_prob", 0.20),
    }

    op = _weighted_choice(probs)

    if op == "add_layer":
        _mutate_add_layer(genome, space)
    elif op == "remove_layer":
        _mutate_remove_layer(genome, space)
    elif op == "change_param":
        _mutate_change_param(genome, space)
    elif op == "toggle_skip":
        _mutate_toggle_skip(genome, space)

    # Ensure validity after mutation
    genome.layers = space._ensure_valid_structure(genome.layers)
    genome.layers = genome.layers[: space.max_depth]
    genome.skip_connections = _filter_skips(genome.skip_connections, len(genome.layers))

    return genome


def _weighted_choice(probs: Dict[str, float]) -> str:
    ops = list(probs.keys())
    weights = [probs[o] for o in ops]
    return random.choices(ops, weights=weights, k=1)[0]


def _mutate_add_layer(genome: Genome, space: SearchSpace) -> None:
    """Insert a random layer at a random position."""
    if len(genome.layers) >= space.max_depth:
        return
    idx = random.randint(0, len(genome.layers))
    genome.layers.insert(idx, space.random_layer())


def _mutate_remove_layer(genome: Genome, space: SearchSpace) -> None:
    """Remove a random layer (keep at least *min_depth*)."""
    if len(genome.layers) <= space.min_depth:
        return
    idx = random.randint(0, len(genome.layers) - 1)
    genome.layers.pop(idx)


def _mutate_change_param(genome: Genome, space: SearchSpace) -> None:
    """Mutate one hyper-parameter of a random layer."""
    if not genome.layers:
        return
    idx = random.randint(0, len(genome.layers) - 1)
    genome.layers[idx] = space.mutate_layer(genome.layers[idx])


def _mutate_toggle_skip(genome: Genome, space: SearchSpace) -> None:
    """Add a new skip connection or remove an existing one."""
    if not space.skip_enabled:
        return
    depth = len(genome.layers)
    if depth < 3:
        return

    if genome.skip_connections and random.random() < 0.5:
        # Remove a random existing skip
        genome.skip_connections.pop(random.randint(0, len(genome.skip_connections) - 1))
    else:
        # Add a new skip
        src = random.randint(0, depth - 2)
        max_dst = min(src + space.skip_max_span, depth - 1)
        if max_dst > src:
            dst = random.randint(src + 1, max_dst)
            if (src, dst) not in genome.skip_connections:
                genome.skip_connections.append((src, dst))
