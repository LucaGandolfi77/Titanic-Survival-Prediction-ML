"""
Genome Encoder
==============
Utilities for encoding, decoding, repairing, and hashing genomes.
Works with both MLP and CNN genome types.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List

from search_space.mlp_space import (
    MLP_GENE_SPECS,
    MLP_GENOME_LENGTH,
    decode_mlp_genome,
    describe_mlp,
)
from search_space.cnn_space import (
    CNN_GENE_SPECS,
    CNN_GENOME_LENGTH,
    decode_cnn_genome,
    describe_cnn,
)


def _clip_gene(value: float, spec: Any) -> float:
    if spec.gene_type in ("cat", "cat_zero"):
        n = len(spec.choices)
        return float(int(value) % n)
    if spec.gene_type == "int":
        return float(max(int(spec.low), min(int(spec.high), int(round(value)))))
    if spec.gene_type == "float":
        return float(max(spec.low, min(spec.high, value)))
    if spec.gene_type == "log_float":
        return float(max(spec.low, min(spec.high, value)))
    return value


def repair_mlp(genome: List[float]) -> List[float]:
    """Fix constraint violations in an MLP genome in place."""
    while len(genome) < MLP_GENOME_LENGTH:
        genome.append(0.0)
    genome = genome[:MLP_GENOME_LENGTH]

    for i, spec in enumerate(MLP_GENE_SPECS):
        genome[i] = _clip_gene(genome[i], spec)

    n_layers = int(genome[0])
    # Zero out inactive layers
    for i in range(n_layers, 6):
        genome[1 + i] = 0.0
    # Ensure first active layer has width >= 16
    if genome[1] < 16:
        genome[1] = 16.0

    return genome


def repair_cnn(genome: List[float]) -> List[float]:
    """Fix constraint violations in a CNN genome in place."""
    while len(genome) < CNN_GENOME_LENGTH:
        genome.append(0.0)
    genome = genome[:CNN_GENOME_LENGTH]

    for i, spec in enumerate(CNN_GENE_SPECS):
        genome[i] = _clip_gene(genome[i], spec)

    n_blocks = int(genome[0])
    for i in range(n_blocks, 5):
        genome[1 + i] = 0.0

    if genome[13] < 1:
        genome[13] = 1.0

    return genome


def repair(genome: List[float], net_type: str) -> List[float]:
    """Dispatch repair to the appropriate genome type."""
    if net_type == "mlp":
        return repair_mlp(genome)
    elif net_type == "cnn":
        return repair_cnn(genome)
    raise ValueError(f"Unknown net_type: {net_type}")


def encode(config_dict: Dict[str, Any], net_type: str) -> List[float]:
    """Encode a config dictionary back to genome (inverse of decode)."""
    if net_type == "mlp":
        from search_space.mlp_space import ACTIVATIONS, OPTIMIZERS, BATCH_SIZES
        genome = [0.0] * MLP_GENOME_LENGTH
        genome[0] = float(config_dict["n_layers"])
        for i, s in enumerate(config_dict["hidden_sizes"]):
            genome[1 + i] = float(s)
        genome[7] = float(ACTIVATIONS.index(config_dict["activation"]))
        genome[8] = config_dict["dropout_rate"]
        genome[9] = float(config_dict["use_batch_norm"])
        genome[10] = float(OPTIMIZERS.index(config_dict["optimizer"]))
        genome[11] = config_dict["learning_rate"]
        genome[12] = config_dict["weight_decay"]
        genome[13] = float(BATCH_SIZES.index(config_dict["batch_size"]))
        return genome
    elif net_type == "cnn":
        from search_space.cnn_space import (
            FILTER_OPTIONS, KERNEL_SIZES, POOLING_TYPES,
            CNN_ACTIVATIONS, CNN_OPTIMIZERS, CNN_BATCH_SIZES,
        )
        genome = [0.0] * CNN_GENOME_LENGTH
        genome[0] = float(config_dict["n_conv_blocks"])
        for i, f in enumerate(config_dict["filters"]):
            if i == 0:
                genome[1] = float(FILTER_OPTIONS.index(f))
            else:
                genome[1 + i] = float(([0] + FILTER_OPTIONS).index(f))
        genome[6] = float(KERNEL_SIZES.index(config_dict["kernel_size"]))
        genome[7] = float(config_dict["use_depthwise"])
        genome[8] = float(config_dict["use_skip_conn"])
        genome[9] = float(POOLING_TYPES.index(config_dict["pooling_type"]))
        genome[10] = float(CNN_ACTIVATIONS.index(config_dict["activation"]))
        genome[11] = config_dict["dropout_rate"]
        genome[12] = float(config_dict["use_batch_norm"])
        genome[13] = float(config_dict["dense_layers"])
        genome[14] = float(config_dict["dense_width"])
        genome[15] = float(CNN_OPTIMIZERS.index(config_dict["optimizer"]))
        genome[16] = config_dict["learning_rate"]
        genome[17] = config_dict["weight_decay"]
        genome[18] = float(CNN_BATCH_SIZES.index(config_dict["batch_size"]))
        return genome
    raise ValueError(f"Unknown net_type: {net_type}")


def decode(genome: List[float], net_type: str) -> Dict[str, Any]:
    """Decode genome into a configuration dictionary."""
    if net_type == "mlp":
        return decode_mlp_genome(genome)
    elif net_type == "cnn":
        return decode_cnn_genome(genome)
    raise ValueError(f"Unknown net_type: {net_type}")


def hash_genome(genome: List[float], dataset: str = "") -> str:
    """Deterministic hash of a genome, optionally including dataset name."""
    rounded = [round(g, 6) for g in genome]
    key = f"{dataset}|{rounded}"
    return hashlib.md5(key.encode()).hexdigest()


def describe(genome: List[float], net_type: str) -> str:
    """Human-readable description of a genome."""
    if net_type == "mlp":
        return describe_mlp(genome)
    elif net_type == "cnn":
        return describe_cnn(genome)
    raise ValueError(f"Unknown net_type: {net_type}")


if __name__ == "__main__":
    import numpy as np
    from search_space.mlp_space import random_mlp_genome
    rng = np.random.default_rng(42)
    g = random_mlp_genome(rng)
    h = hash_genome(g, "MNIST")
    print(f"Hash: {h}")
    d = decode(g, "mlp")
    g2 = encode(d, "mlp")
    print(f"Round-trip decode→encode: {decode(g2, 'mlp')}")
