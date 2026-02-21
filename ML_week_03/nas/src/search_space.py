"""Search-space definition and random genome sampling.

The search space is fully defined by the YAML config.  This module reads
those constraints and provides :meth:`SearchSpace.random_genome` to
draw a valid random architecture.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from src.genome import Genome, LayerGene


class SearchSpace:
    """Manages the set of valid layer types and parameter ranges.

    Parameters
    ----------
    cfg : dict
        The ``search_space`` section of the YAML config.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.min_depth: int = cfg.get("min_depth", 5)
        self.max_depth: int = cfg.get("max_depth", 15)
        self.layer_types: List[str] = cfg.get(
            "layer_types",
            ["conv2d", "maxpool", "avgpool", "batchnorm", "dropout", "dense"],
        )

        # Per-type parameter ranges
        conv = cfg.get("conv", {})
        self.kernel_sizes: List[int] = conv.get("kernel_sizes", [3, 5, 7])
        self.filters: List[int] = conv.get("filters", [32, 64, 128])
        self.conv_activations: List[str] = conv.get("activations", ["relu", "elu"])

        pool = cfg.get("pool", {})
        self.pool_size: int = pool.get("size", 2)

        drop = cfg.get("dropout", {})
        self.dropout_rates: List[float] = drop.get("rates", [0.1, 0.2, 0.3, 0.4, 0.5])

        dense = cfg.get("dense", {})
        self.dense_units: List[int] = dense.get("units", [64, 128, 256, 512])
        self.dense_activations: List[str] = dense.get("activations", ["relu", "elu"])

        skip = cfg.get("skip_connections", {})
        self.skip_enabled: bool = skip.get("enabled", True)
        self.skip_max_span: int = skip.get("max_span", 4)

    # ── random layer sampling ────────────────────────────────────────────────

    def random_layer(self, layer_type: Optional[str] = None) -> LayerGene:
        """Sample a random layer gene.

        If *layer_type* is ``None`` a random type is chosen from the
        allowed set.
        """
        if layer_type is None:
            layer_type = random.choice(self.layer_types)

        params: Dict[str, Any] = {}

        if layer_type == "conv2d":
            params["filters"] = random.choice(self.filters)
            params["kernel_size"] = random.choice(self.kernel_sizes)
            params["activation"] = random.choice(self.conv_activations)

        elif layer_type in ("maxpool", "avgpool"):
            params["size"] = self.pool_size

        elif layer_type == "batchnorm":
            pass  # no tuneable params

        elif layer_type == "dropout":
            params["rate"] = random.choice(self.dropout_rates)

        elif layer_type == "dense":
            params["units"] = random.choice(self.dense_units)
            params["activation"] = random.choice(self.dense_activations)

        return LayerGene(layer_type=layer_type, params=params)

    # ── structural helpers ───────────────────────────────────────────────────

    def _ensure_valid_structure(self, layers: List[LayerGene]) -> List[LayerGene]:
        """Apply heuristic rules so the architecture compiles.

        Rules
        -----
        1.  Must start with at least one ``conv2d`` (so the feature map exists).
        2.  Cannot have two consecutive pool layers (spatial dims would vanish).
        3.  ``dense`` layers cannot appear before the last pool/conv — we push
            them to the end.
        4.  At least one ``conv2d`` in the genome.
        """
        conv_layers = [l for l in layers if l.layer_type == "conv2d"]
        non_dense = [l for l in layers if l.layer_type != "dense"]
        dense_layers = [l for l in layers if l.layer_type == "dense"]

        # Guarantee at least one conv at the start
        if not conv_layers:
            non_dense.insert(0, self.random_layer("conv2d"))
        elif non_dense and non_dense[0].layer_type != "conv2d":
            non_dense.insert(0, self.random_layer("conv2d"))

        # Remove consecutive pools
        cleaned: List[LayerGene] = [non_dense[0]]
        for layer in non_dense[1:]:
            prev = cleaned[-1]
            if layer.layer_type in ("maxpool", "avgpool") and prev.layer_type in (
                "maxpool",
                "avgpool",
            ):
                continue  # skip duplicate pool
            cleaned.append(layer)

        # Append dense layers at the end
        cleaned.extend(dense_layers)
        return cleaned

    def _random_skips(
        self, depth: int
    ) -> List[Tuple[int, int]]:
        """Generate a small random set of skip connections."""
        if not self.skip_enabled or depth < 3:
            return []
        n_skips = random.randint(0, max(1, depth // 4))
        skips: List[Tuple[int, int]] = []
        for _ in range(n_skips):
            src = random.randint(0, depth - 2)
            max_dst = min(src + self.skip_max_span, depth - 1)
            if max_dst <= src:
                continue
            dst = random.randint(src + 1, max_dst)
            skips.append((src, dst))
        return list(set(skips))

    # ── public API ───────────────────────────────────────────────────────────

    def random_genome(self, generation: int = 0) -> Genome:
        """Create a fully random but *valid* genome."""
        depth = random.randint(self.min_depth, self.max_depth)
        raw_layers = [self.random_layer() for _ in range(depth)]
        layers = self._ensure_valid_structure(raw_layers)
        skips = self._random_skips(len(layers))
        return Genome(
            layers=layers,
            skip_connections=skips,
            generation=generation,
        )

    def mutate_layer(self, gene: LayerGene) -> LayerGene:
        """Return a mutated copy of a single layer gene (change one param)."""
        new = LayerGene(layer_type=gene.layer_type, params=dict(gene.params))

        if new.layer_type == "conv2d":
            choice = random.choice(["filters", "kernel_size", "activation"])
            if choice == "filters":
                new.params["filters"] = random.choice(self.filters)
            elif choice == "kernel_size":
                new.params["kernel_size"] = random.choice(self.kernel_sizes)
            else:
                new.params["activation"] = random.choice(self.conv_activations)

        elif new.layer_type == "dropout":
            new.params["rate"] = random.choice(self.dropout_rates)

        elif new.layer_type == "dense":
            choice = random.choice(["units", "activation"])
            if choice == "units":
                new.params["units"] = random.choice(self.dense_units)
            else:
                new.params["activation"] = random.choice(self.dense_activations)

        # batchnorm / pool → nothing to mutate, return as-is
        return new
