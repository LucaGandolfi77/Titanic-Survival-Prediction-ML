"""
Feature Extractor
=================
Convert a genome (MLP or CNN) into a fixed-length numerical feature
vector suitable for surrogate model input.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from search_space.mlp_space import (
    MLP_GENOME_LENGTH, MLP_GENE_SPECS, ACTIVATIONS, OPTIMIZERS, BATCH_SIZES,
)
from search_space.cnn_space import (
    CNN_GENOME_LENGTH, CNN_GENE_SPECS, FILTER_OPTIONS, KERNEL_SIZES,
    POOLING_TYPES, CNN_ACTIVATIONS, CNN_OPTIMIZERS, CNN_BATCH_SIZES,
)


def genome_to_features_mlp(genome: List[float]) -> np.ndarray:
    """Extract a numerical feature vector from an MLP genome.

    Categorical genes are one-hot encoded; numerical genes are normalized.
    """
    features: List[float] = []

    # n_layers (normalized to [0, 1])
    features.append(genome[0] / 6.0)

    # hidden sizes (normalized by 512)
    for i in range(6):
        features.append(genome[1 + i] / 512.0)

    # activation (one-hot, 5 categories)
    act_idx = int(genome[7]) % len(ACTIVATIONS)
    oh = [0.0] * len(ACTIVATIONS)
    oh[act_idx] = 1.0
    features.extend(oh)

    # dropout_rate
    features.append(genome[8])

    # use_batch_norm
    features.append(float(int(genome[9])))

    # optimizer (one-hot, 4 categories)
    opt_idx = int(genome[10]) % len(OPTIMIZERS)
    oh = [0.0] * len(OPTIMIZERS)
    oh[opt_idx] = 1.0
    features.extend(oh)

    # learning_rate (log-scaled)
    features.append((math.log10(max(genome[11], 1e-8)) + 4) / 3)

    # weight_decay (log-scaled)
    features.append((math.log10(max(genome[12], 1e-10)) + 6) / 4)

    # batch_size (normalized)
    bs_idx = int(genome[13]) % len(BATCH_SIZES)
    features.append(bs_idx / (len(BATCH_SIZES) - 1))

    return np.array(features, dtype=np.float32)


def genome_to_features_cnn(genome: List[float]) -> np.ndarray:
    """Extract a numerical feature vector from a CNN genome."""
    features: List[float] = []

    # n_conv_blocks
    features.append(genome[0] / 5.0)

    # filter counts (normalized by 128)
    for i in range(5):
        features.append(genome[1 + i] / max(len(FILTER_OPTIONS), 1))

    # kernel_size (one-hot, 2 categories)
    k_idx = int(genome[6]) % len(KERNEL_SIZES)
    oh = [0.0] * len(KERNEL_SIZES)
    oh[k_idx] = 1.0
    features.extend(oh)

    # use_depthwise, use_skip_conn
    features.append(float(int(genome[7])))
    features.append(float(int(genome[8])))

    # pooling_type (one-hot, 3 categories)
    p_idx = int(genome[9]) % len(POOLING_TYPES)
    oh = [0.0] * len(POOLING_TYPES)
    oh[p_idx] = 1.0
    features.extend(oh)

    # activation (one-hot, 3 categories)
    a_idx = int(genome[10]) % len(CNN_ACTIVATIONS)
    oh = [0.0] * len(CNN_ACTIVATIONS)
    oh[a_idx] = 1.0
    features.extend(oh)

    # dropout_rate
    features.append(genome[11])

    # use_batch_norm
    features.append(float(int(genome[12])))

    # dense_layers, dense_width
    features.append(genome[13] / 3.0)
    features.append(genome[14] / 512.0)

    # optimizer (one-hot, 3 categories)
    o_idx = int(genome[15]) % len(CNN_OPTIMIZERS)
    oh = [0.0] * len(CNN_OPTIMIZERS)
    oh[o_idx] = 1.0
    features.extend(oh)

    # learning_rate, weight_decay (log)
    features.append((math.log10(max(genome[16], 1e-8)) + 4) / 3)
    features.append((math.log10(max(genome[17], 1e-10)) + 6) / 4)

    # batch_size
    bs_idx = int(genome[18]) % len(CNN_BATCH_SIZES)
    features.append(bs_idx / max(len(CNN_BATCH_SIZES) - 1, 1))

    return np.array(features, dtype=np.float32)


def genome_to_features(genome: List[float], net_type: str) -> np.ndarray:
    """Dispatch to the appropriate feature extractor."""
    if net_type == "mlp":
        return genome_to_features_mlp(genome)
    elif net_type == "cnn":
        return genome_to_features_cnn(genome)
    raise ValueError(f"Unknown net_type: {net_type}")


if __name__ == "__main__":
    from search_space.mlp_space import random_mlp_genome
    rng = np.random.default_rng(42)
    g = random_mlp_genome(rng)
    f = genome_to_features(g, "mlp")
    print(f"MLP feature vector length: {len(f)}")
    print(f"Values: {f}")
