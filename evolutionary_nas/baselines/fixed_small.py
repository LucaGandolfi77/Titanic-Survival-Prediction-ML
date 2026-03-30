"""
Fixed Small Baselines
=====================
Hand-designed small MLP and CNN architectures for comparison.
"""

from __future__ import annotations

from typing import Any, Dict, List


def get_fixed_mlp_configs() -> List[Dict[str, Any]]:
    """Return a list of hand-designed small MLP configurations."""
    return [
        {
            "name": "MLP-Tiny",
            "hidden_sizes": [64, 32],
            "activation": "relu",
            "dropout_rate": 0.2,
            "use_batch_norm": True,
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 64,
        },
        {
            "name": "MLP-Small",
            "hidden_sizes": [128, 64],
            "activation": "relu",
            "dropout_rate": 0.3,
            "use_batch_norm": True,
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 128,
        },
        {
            "name": "MLP-Medium",
            "hidden_sizes": [256, 128, 64],
            "activation": "gelu",
            "dropout_rate": 0.3,
            "use_batch_norm": True,
            "optimizer": "adamw",
            "learning_rate": 5e-4,
            "weight_decay": 1e-3,
            "batch_size": 128,
        },
    ]


def get_fixed_cnn_configs() -> List[Dict[str, Any]]:
    """Return a list of hand-designed small CNN configurations."""
    return [
        {
            "name": "CNN-Tiny",
            "filters": [16, 32],
            "kernel_size": 3,
            "use_depthwise": False,
            "use_skip_conn": False,
            "pooling_type": "max",
            "activation": "relu",
            "dropout_rate": 0.2,
            "use_batch_norm": True,
            "dense_layers": 1,
            "dense_width": 64,
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 64,
        },
        {
            "name": "CNN-Small",
            "filters": [32, 64],
            "kernel_size": 3,
            "use_depthwise": False,
            "use_skip_conn": True,
            "pooling_type": "max",
            "activation": "relu",
            "dropout_rate": 0.25,
            "use_batch_norm": True,
            "dense_layers": 1,
            "dense_width": 128,
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 64,
        },
        {
            "name": "CNN-Medium",
            "filters": [32, 64, 128],
            "kernel_size": 3,
            "use_depthwise": True,
            "use_skip_conn": True,
            "pooling_type": "max",
            "activation": "gelu",
            "dropout_rate": 0.3,
            "use_batch_norm": True,
            "dense_layers": 2,
            "dense_width": 256,
            "optimizer": "adamw",
            "learning_rate": 5e-4,
            "weight_decay": 1e-3,
            "batch_size": 128,
        },
    ]


if __name__ == "__main__":
    for cfg in get_fixed_mlp_configs():
        print(f"{cfg['name']}: {cfg['hidden_sizes']}")
    for cfg in get_fixed_cnn_configs():
        print(f"{cfg['name']}: {cfg['filters']}")
