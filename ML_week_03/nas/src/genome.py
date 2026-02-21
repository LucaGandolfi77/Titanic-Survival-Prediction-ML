"""Genome encoding for neural architectures.

A *genome* is an ordered list of ``LayerGene`` objects that fully describe a
CNN architecture.  Each gene stores the layer type and its hyper-parameters.
Skip connections are stored separately as ``(src_idx, dst_idx)`` pairs.

The genome can be serialised to / loaded from JSON so that the best
architecture discovered by the search can be saved and reproduced.
"""

from __future__ import annotations

import copy
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Layer gene ───────────────────────────────────────────────────────────────

@dataclass
class LayerGene:
    """Single layer specification inside a genome.

    Parameters
    ----------
    layer_type : str
        One of ``conv2d``, ``maxpool``, ``avgpool``, ``batchnorm``,
        ``dropout``, ``dense``.
    params : dict
        Type-specific hyper-parameters.  Examples::

            conv2d   → {"filters": 64, "kernel_size": 3, "activation": "relu"}
            maxpool  → {"size": 2}
            avgpool  → {"size": 2}
            batchnorm → {}
            dropout  → {"rate": 0.3}
            dense    → {"units": 128, "activation": "relu"}
    """

    layer_type: str
    params: Dict[str, Any] = field(default_factory=dict)

    # -- serialisation helpers ------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {"layer_type": self.layer_type, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LayerGene":
        return cls(layer_type=d["layer_type"], params=dict(d.get("params", {})))

    def __repr__(self) -> str:  # noqa: D105
        p = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.layer_type}({p})"


# ── Genome ───────────────────────────────────────────────────────────────────

@dataclass
class Genome:
    """Complete architecture description.

    Attributes
    ----------
    layers : list[LayerGene]
        Ordered layers that will be compiled into a ``nn.Module``.
    skip_connections : list[tuple[int, int]]
        Residual connections ``(src_layer_idx, dst_layer_idx)``.
    fitness : float | None
        Validation accuracy obtained during evaluation (``None`` if not yet
        evaluated).
    generation : int
        Generation in which this genome was created.
    parent_ids : tuple[str, str] | None
        IDs of the two parents (``None`` for random-init genomes).
    id : str
        Unique identifier (short UUID).
    """

    layers: List[LayerGene] = field(default_factory=list)
    skip_connections: List[Tuple[int, int]] = field(default_factory=list)
    fitness: Optional[float] = None
    generation: int = 0
    parent_ids: Optional[Tuple[str, str]] = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    # -- helpers --------------------------------------------------------------

    @property
    def depth(self) -> int:
        """Number of evolvable layers."""
        return len(self.layers)

    def clone(self) -> "Genome":
        """Deep-copy (new id, fitness reset)."""
        g = copy.deepcopy(self)
        g.id = uuid.uuid4().hex[:8]
        g.fitness = None
        return g

    # -- serialisation --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "generation": self.generation,
            "fitness": self.fitness,
            "parent_ids": list(self.parent_ids) if self.parent_ids else None,
            "layers": [l.to_dict() for l in self.layers],
            "skip_connections": [list(sc) for sc in self.skip_connections],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Genome":
        return cls(
            id=d.get("id", uuid.uuid4().hex[:8]),
            generation=d.get("generation", 0),
            fitness=d.get("fitness"),
            parent_ids=tuple(d["parent_ids"]) if d.get("parent_ids") else None,
            layers=[LayerGene.from_dict(ld) for ld in d["layers"]],
            skip_connections=[tuple(sc) for sc in d.get("skip_connections", [])],
        )

    def save(self, path: str | Path) -> None:
        """Write genome to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Genome":
        """Load genome from a JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text()))

    def summary(self) -> str:
        """Human-readable one-liner."""
        skips = len(self.skip_connections)
        fit = f"{self.fitness:.4f}" if self.fitness is not None else "?"
        return (
            f"Genome({self.id}) depth={self.depth} skips={skips} "
            f"fitness={fit} gen={self.generation}"
        )

    def __repr__(self) -> str:  # noqa: D105
        return self.summary()
