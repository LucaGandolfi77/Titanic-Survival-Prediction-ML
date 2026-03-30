"""
Constraints
============
Validate architecture constraints: parameter budget, minimum depth, valid combos.
"""

from __future__ import annotations

from typing import List

from config import CFG


def check_param_budget(param_count: int) -> bool:
    """Return True if param_count is within the allowed budget."""
    return param_count <= CFG.MAX_PARAMS


def check_mlp_constraints(genome: List[float]) -> bool:
    """Validate MLP genome constraints beyond simple clipping."""
    n_layers = int(genome[0])
    if n_layers < 1 or n_layers > CFG.MLP_MAX_LAYERS:
        return False
    if int(genome[1]) < 16:
        return False
    return True


def check_cnn_constraints(genome: List[float]) -> bool:
    """Validate CNN genome constraints."""
    n_blocks = int(genome[0])
    if n_blocks < 1 or n_blocks > CFG.CNN_MAX_BLOCKS:
        return False
    dense_layers = int(genome[13])
    if dense_layers < 1 or dense_layers > 3:
        return False
    return True


def is_valid(genome: List[float], net_type: str) -> bool:
    """Check if a genome satisfies all constraints for its type."""
    if net_type == "mlp":
        return check_mlp_constraints(genome)
    elif net_type == "cnn":
        return check_cnn_constraints(genome)
    return False


if __name__ == "__main__":
    print(f"Max params: {CFG.MAX_PARAMS}")
    print(f"check_param_budget(100_000) = {check_param_budget(100_000)}")
    print(f"check_param_budget(1_000_000) = {check_param_budget(1_000_000)}")
