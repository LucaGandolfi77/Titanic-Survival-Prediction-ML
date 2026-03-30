"""
CNN Search Space
================
Fixed-length genome encoding for Convolutional Neural Network architectures.
Supports depthwise-separable convolutions and skip connections for compactness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

FILTER_OPTIONS = [8, 16, 32, 64, 128]
KERNEL_SIZES = [3, 5]
POOLING_TYPES = ["max", "avg", "none"]
CNN_ACTIVATIONS = ["relu", "elu", "gelu"]
CNN_OPTIMIZERS = ["adam", "sgd", "adamw"]
CNN_BATCH_SIZES = [32, 64, 128]

CNN_GENOME_LENGTH = 19


@dataclass
class GeneSpec:
    name: str
    gene_type: str
    low: float
    high: float
    choices: List[Any] | None = None


CNN_GENE_SPECS: List[GeneSpec] = [
    GeneSpec("n_conv_blocks",  "int",       1,    5),
    GeneSpec("filters_0",      "cat",       0,    len(FILTER_OPTIONS) - 1, FILTER_OPTIONS),
    GeneSpec("filters_1",      "cat_zero",  0,    len(FILTER_OPTIONS),     [0] + FILTER_OPTIONS),
    GeneSpec("filters_2",      "cat_zero",  0,    len(FILTER_OPTIONS),     [0] + FILTER_OPTIONS),
    GeneSpec("filters_3",      "cat_zero",  0,    len(FILTER_OPTIONS),     [0] + FILTER_OPTIONS),
    GeneSpec("filters_4",      "cat_zero",  0,    len(FILTER_OPTIONS),     [0] + FILTER_OPTIONS),
    GeneSpec("kernel_size",    "cat",       0,    len(KERNEL_SIZES) - 1,   KERNEL_SIZES),
    GeneSpec("use_depthwise",  "int",       0,    1),
    GeneSpec("use_skip_conn",  "int",       0,    1),
    GeneSpec("pooling_type",   "cat",       0,    len(POOLING_TYPES) - 1,  POOLING_TYPES),
    GeneSpec("activation",     "cat",       0,    len(CNN_ACTIVATIONS) - 1, CNN_ACTIVATIONS),
    GeneSpec("dropout_rate",   "float",     0.0,  0.5),
    GeneSpec("use_batch_norm", "int",       0,    1),
    GeneSpec("dense_layers",   "int",       1,    3),
    GeneSpec("dense_width",    "int",       64,   512),
    GeneSpec("optimizer",      "cat",       0,    len(CNN_OPTIMIZERS) - 1,  CNN_OPTIMIZERS),
    GeneSpec("learning_rate",  "log_float", 1e-4, 1e-1),
    GeneSpec("weight_decay",   "log_float", 1e-6, 1e-2),
    GeneSpec("batch_size",     "cat",       0,    len(CNN_BATCH_SIZES) - 1, CNN_BATCH_SIZES),
]


def random_cnn_genome(rng: Any) -> List[float]:
    """Generate a random CNN genome."""
    genome: List[float] = []
    for spec in CNN_GENE_SPECS:
        if spec.gene_type == "int":
            genome.append(float(rng.integers(int(spec.low), int(spec.high) + 1)))
        elif spec.gene_type == "float":
            genome.append(float(rng.uniform(spec.low, spec.high)))
        elif spec.gene_type == "log_float":
            log_lo, log_hi = math.log10(spec.low), math.log10(spec.high)
            genome.append(float(10 ** rng.uniform(log_lo, log_hi)))
        elif spec.gene_type in ("cat", "cat_zero"):
            n_choices = len(spec.choices)
            genome.append(float(rng.integers(0, n_choices)))
        else:
            raise ValueError(f"Unknown gene type: {spec.gene_type}")
    return genome


def decode_cnn_genome(genome: List[float]) -> Dict[str, Any]:
    """Convert raw CNN genome list into a human-readable dict."""
    n_blocks = int(genome[0])

    filters_list: List[int] = []
    first_spec = CNN_GENE_SPECS[1]
    f0 = FILTER_OPTIONS[int(genome[1]) % len(FILTER_OPTIONS)]
    filters_list.append(f0)

    for i in range(1, n_blocks):
        spec = CNN_GENE_SPECS[1 + i]
        idx = int(genome[1 + i]) % len(spec.choices)
        val = spec.choices[idx]
        if val == 0:
            val = FILTER_OPTIONS[0]
        filters_list.append(val)

    return {
        "n_conv_blocks": len(filters_list),
        "filters": filters_list,
        "kernel_size": KERNEL_SIZES[int(genome[6]) % len(KERNEL_SIZES)],
        "use_depthwise": bool(int(genome[7])),
        "use_skip_conn": bool(int(genome[8])),
        "pooling_type": POOLING_TYPES[int(genome[9]) % len(POOLING_TYPES)],
        "activation": CNN_ACTIVATIONS[int(genome[10]) % len(CNN_ACTIVATIONS)],
        "dropout_rate": float(genome[11]),
        "use_batch_norm": bool(int(genome[12])),
        "dense_layers": int(genome[13]),
        "dense_width": int(genome[14]),
        "optimizer": CNN_OPTIMIZERS[int(genome[15]) % len(CNN_OPTIMIZERS)],
        "learning_rate": float(genome[16]),
        "weight_decay": float(genome[17]),
        "batch_size": CNN_BATCH_SIZES[int(genome[18]) % len(CNN_BATCH_SIZES)],
    }


def describe_cnn(genome: List[float]) -> str:
    """Return a one-line human-readable description of a CNN genome."""
    d = decode_cnn_genome(genome)
    f_str = "→".join(str(f) for f in d["filters"])
    dw = "DW" if d["use_depthwise"] else "Std"
    skip = "+skip" if d["use_skip_conn"] else ""
    return (
        f"CNN[{f_str}] k={d['kernel_size']} {dw}{skip} "
        f"pool={d['pooling_type']} act={d['activation']} "
        f"bn={d['use_batch_norm']} do={d['dropout_rate']:.2f} "
        f"dense={d['dense_layers']}×{d['dense_width']} "
        f"opt={d['optimizer']} lr={d['learning_rate']:.1e} bs={d['batch_size']}"
    )


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(42)
    g = random_cnn_genome(rng)
    print(describe_cnn(g))
    print(decode_cnn_genome(g))
