"""
MLP Search Space
================
Fixed-length genome encoding for Multi-Layer Perceptron architectures.
Each gene has a type (int, float, categorical) and valid range.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

ACTIVATIONS = ["relu", "tanh", "elu", "selu", "gelu"]
OPTIMIZERS = ["adam", "sgd", "adamw", "rmsprop"]
BATCH_SIZES = [32, 64, 128, 256]

MLP_GENOME_LENGTH = 14


@dataclass
class GeneSpec:
    name: str
    gene_type: str          # "int", "float", "cat", "log_float"
    low: float
    high: float
    choices: List[Any] | None = None


MLP_GENE_SPECS: List[GeneSpec] = [
    GeneSpec("n_layers",       "int",       1,    6),
    GeneSpec("hidden_size_0",  "int",       16,   512),
    GeneSpec("hidden_size_1",  "int",       0,    512),
    GeneSpec("hidden_size_2",  "int",       0,    512),
    GeneSpec("hidden_size_3",  "int",       0,    512),
    GeneSpec("hidden_size_4",  "int",       0,    512),
    GeneSpec("hidden_size_5",  "int",       0,    512),
    GeneSpec("activation",     "cat",       0,    len(ACTIVATIONS) - 1, ACTIVATIONS),
    GeneSpec("dropout_rate",   "float",     0.0,  0.6),
    GeneSpec("use_batch_norm", "int",       0,    1),
    GeneSpec("optimizer",      "cat",       0,    len(OPTIMIZERS) - 1, OPTIMIZERS),
    GeneSpec("learning_rate",  "log_float", 1e-4, 1e-1),
    GeneSpec("weight_decay",   "log_float", 1e-6, 1e-2),
    GeneSpec("batch_size",     "cat",       0,    len(BATCH_SIZES) - 1, BATCH_SIZES),
]


def random_mlp_genome(rng: Any) -> List[float]:
    """Generate a random MLP genome using the provided numpy RNG."""
    import math
    genome: List[float] = []
    for spec in MLP_GENE_SPECS:
        if spec.gene_type == "int":
            genome.append(float(rng.integers(int(spec.low), int(spec.high) + 1)))
        elif spec.gene_type == "float":
            genome.append(float(rng.uniform(spec.low, spec.high)))
        elif spec.gene_type == "log_float":
            log_low, log_high = math.log10(spec.low), math.log10(spec.high)
            genome.append(float(10 ** rng.uniform(log_low, log_high)))
        elif spec.gene_type == "cat":
            genome.append(float(rng.integers(0, len(spec.choices))))
        else:
            raise ValueError(f"Unknown gene type: {spec.gene_type}")
    return genome


def decode_mlp_genome(genome: List[float]) -> Dict[str, Any]:
    """Convert raw genome list into a human-readable dict."""
    n_layers = int(genome[0])
    hidden_sizes = []
    for i in range(n_layers):
        w = int(genome[1 + i])
        if i == 0 and w == 0:
            w = 16
        if w > 0:
            hidden_sizes.append(w)

    return {
        "n_layers": len(hidden_sizes) if hidden_sizes else 1,
        "hidden_sizes": hidden_sizes if hidden_sizes else [16],
        "activation": ACTIVATIONS[int(genome[7]) % len(ACTIVATIONS)],
        "dropout_rate": float(genome[8]),
        "use_batch_norm": bool(int(genome[9])),
        "optimizer": OPTIMIZERS[int(genome[10]) % len(OPTIMIZERS)],
        "learning_rate": float(genome[11]),
        "weight_decay": float(genome[12]),
        "batch_size": BATCH_SIZES[int(genome[13]) % len(BATCH_SIZES)],
    }


def describe_mlp(genome: List[float]) -> str:
    """Return a one-line human-readable description of an MLP genome."""
    d = decode_mlp_genome(genome)
    sizes = "→".join(str(s) for s in d["hidden_sizes"])
    return (
        f"MLP[{sizes}] act={d['activation']} bn={d['use_batch_norm']} "
        f"do={d['dropout_rate']:.2f} opt={d['optimizer']} "
        f"lr={d['learning_rate']:.1e} wd={d['weight_decay']:.1e} "
        f"bs={d['batch_size']}"
    )


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(42)
    g = random_mlp_genome(rng)
    print(describe_mlp(g))
    print(decode_mlp_genome(g))
