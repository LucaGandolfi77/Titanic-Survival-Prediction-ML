"""Weight inheritance — transfer trained weights from parent to child.

When a child genome shares identical layer configurations with a parent,
the corresponding weights can be copied instead of re-initializing
randomly.  This dramatically speeds up convergence because the child
starts from a warm state.

Strategy
--------
1. Compare each layer gene in the child with the parent's layers.
2. For layers that are *identical* (same type + same params), copy weights.
3. For new / modified layers, keep the random initialization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger

from src.genome import Genome, LayerGene


def genes_match(a: LayerGene, b: LayerGene) -> bool:
    """Return True if two layer genes are functionally identical."""
    return a.layer_type == b.layer_type and a.params == b.params


def compute_inheritance_map(
    child: Genome,
    parent: Genome,
) -> List[Optional[int]]:
    """For each layer in *child*, find the matching parent layer index.

    Uses a greedy forward scan — if child layer *i* matches parent layer
    *j*, record ``map[i] = j`` and advance *j*.  This preserves ordering.

    Returns
    -------
    list[int | None]
        ``map[i]`` is the parent layer index whose weights should be
        copied into child layer *i*, or ``None`` if no match.
    """
    mapping: List[Optional[int]] = [None] * len(child.layers)
    j = 0
    for i, c_gene in enumerate(child.layers):
        saved_j = j
        found = False
        while j < len(parent.layers):
            if genes_match(c_gene, parent.layers[j]):
                mapping[i] = j
                j += 1
                found = True
                break
            j += 1
        if not found:
            j = saved_j  # restore scan position so later layers can still match
    return mapping


def inherit_weights(
    child_model: nn.Module,
    parent_state_dict: Dict[str, Any],
    child_genome: Genome,
    parent_genome: Genome,
) -> int:
    """Copy matching weights from parent into child model.

    Parameters
    ----------
    child_model : nn.Module
        The freshly-built (randomly initialised) child model.
    parent_state_dict : dict
        ``state_dict`` from the trained parent model.
    child_genome, parent_genome : Genome
        Used to compute the layer mapping.

    Returns
    -------
    int
        Number of parameter tensors transferred.
    """
    mapping = compute_inheritance_map(child_genome, parent_genome)

    child_sd = child_model.state_dict()
    n_transferred = 0

    # Build a name→index map for body_layers in both models
    # NASModel stores layers in self.body_layers (ModuleList)
    for child_idx, parent_idx in enumerate(mapping):
        if parent_idx is None:
            continue

        # Keys look like "body_layers.{idx}.0.weight" etc.
        child_prefix = f"body_layers.{child_idx}."
        parent_prefix = f"body_layers.{parent_idx}."

        for ckey in list(child_sd.keys()):
            if not ckey.startswith(child_prefix):
                continue
            pkey = ckey.replace(child_prefix, parent_prefix, 1)
            if pkey in parent_state_dict and parent_state_dict[pkey].shape == child_sd[ckey].shape:
                child_sd[ckey] = parent_state_dict[pkey].clone()
                n_transferred += 1

    # Also try to copy stem and head weights (always the same architecture)
    for key in child_sd:
        if (key.startswith("stem.") or key.startswith("head.")) and key in parent_state_dict:
            if parent_state_dict[key].shape == child_sd[key].shape:
                child_sd[key] = parent_state_dict[key].clone()
                n_transferred += 1

    child_model.load_state_dict(child_sd)
    logger.debug(f"Weight inheritance: transferred {n_transferred} tensors")
    return n_transferred
