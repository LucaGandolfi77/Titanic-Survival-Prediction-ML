"""
operator_base.py – Abstract Operator class, Port, and the global operator registry.

Every concrete operator inherits from :class:`Operator` and registers itself
via the :func:`register_operator` helper so the GUI palette can discover it.
"""
from __future__ import annotations

import abc
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

logger = logging.getLogger(__name__)

# ── Port‑type enum ──────────────────────────────────────────────────────────

class PortType(Enum):
    """Semantic type that flows through a port / wire."""
    EXAMPLE_SET = "ExampleSet"
    MODEL = "Model"
    PERFORMANCE = "Performance"
    ANY = "Any"


# Wire colour look‑up used by the canvas
PORT_COLOURS: Dict[PortType, str] = {
    PortType.EXAMPLE_SET: "#e8a838",   # orange
    PortType.MODEL:       "#3a8ee6",   # blue
    PortType.PERFORMANCE: "#4caf50",   # green
    PortType.ANY:         "#9e9e9e",   # gray
}

# ── Category enum (for the colour‑coded dot on the operator node) ───────

class OpCategory(Enum):
    DATA = "Data"
    TRANSFORM = "Transform"
    FEATURE = "Feature"
    MODEL = "Model"
    EVALUATION = "Evaluation"
    VISUALIZATION = "Visualization"
    UTILITY = "Utility"


CATEGORY_COLOURS: Dict[OpCategory, str] = {
    OpCategory.DATA:          "#42a5f5",
    OpCategory.TRANSFORM:     "#66bb6a",
    OpCategory.FEATURE:       "#ab47bc",
    OpCategory.MODEL:         "#ef5350",
    OpCategory.EVALUATION:    "#ffa726",
    OpCategory.VISUALIZATION: "#26c6da",
    OpCategory.UTILITY:       "#78909c",
}


# ── Port data‑class ────────────────────────────────────────────────────────

@dataclass
class Port:
    """A named input or output port on an operator."""
    name: str
    port_type: PortType = PortType.EXAMPLE_SET
    optional: bool = False
    value: Any = None            # filled during execution
    connected: bool = False      # set by the canvas


# ── ParamSpec – describes one user‑configurable parameter ───────────────

class ParamKind(Enum):
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    BOOL = "bool"
    CHOICE = "choice"
    FILEPATH = "filepath"
    COLUMN = "column"            # populated from connected input schema
    COLUMN_LIST = "column_list"
    TEXT = "text"


@dataclass
class ParamSpec:
    """Metadata for a single operator parameter (drives the right panel)."""
    name: str
    kind: ParamKind
    default: Any = None
    choices: Optional[List[str]] = None
    description: str = ""
    min_val: Optional[float] = None
    max_val: Optional[float] = None


# ── Abstract Operator ───────────────────────────────────────────────────────

class Operator(abc.ABC):
    """Base class for every operator in RapidMiner‑Lite.

    Subclasses MUST define:
      - ``op_type``:  a human‑readable string (e.g. "Read CSV")
      - ``category``: an OpCategory enum value
      - ``_build_ports()``: populates ``self.inputs`` / ``self.outputs``
      - ``_build_params()``: populates ``self.param_specs``
      - ``execute(inputs)``: runs the actual computation

    Subclasses register themselves by calling ``register_operator(cls)``
    at module level so the palette can find them automatically.
    """

    op_type: str = "AbstractOperator"
    category: OpCategory = OpCategory.UTILITY
    description: str = ""

    def __init__(self) -> None:
        self.op_id: str = f"op_{uuid.uuid4().hex[:8]}"
        self.display_name: str = self.op_type
        self.x: float = 100.0
        self.y: float = 100.0

        # Ports
        self.inputs: Dict[str, Port] = {}
        self.outputs: Dict[str, Port] = {}
        self._build_ports()

        # Parameters
        self.param_specs: List[ParamSpec] = []
        self._build_params()
        self.params: Dict[str, Any] = {
            ps.name: ps.default for ps in self.param_specs
        }

        # Execution state
        self.executed: bool = False
        self.error: Optional[str] = None
        self.exec_time: float = 0.0

    # ── abstract hooks ──────────────────────────────────────────────────

    @abc.abstractmethod
    def _build_ports(self) -> None:
        """Populate ``self.inputs`` and ``self.outputs``."""

    @abc.abstractmethod
    def _build_params(self) -> None:
        """Populate ``self.param_specs``."""

    @abc.abstractmethod
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the operator logic.

        Args:
            inputs: mapping port‑name → value for every connected input.

        Returns:
            mapping port‑name → value for each output port.
        """

    # ── helpers ─────────────────────────────────────────────────────────

    def get_param(self, name: str) -> Any:
        """Return current value of a parameter (with fall‑back to spec default)."""
        return self.params.get(name)

    def set_param(self, name: str, value: Any) -> None:
        self.params[name] = value

    def reset_execution(self) -> None:
        """Clear run state so the operator can be re‑executed."""
        self.executed = False
        self.error = None
        self.exec_time = 0.0
        for p in self.outputs.values():
            p.value = None

    # ── serialisation ───────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.op_id,
            "type": self.op_type,
            "display_name": self.display_name,
            "position": [self.x, self.y],
            "params": {k: _serialise_val(v) for k, v in self.params.items()},
        }

    def load_params(self, data: Dict[str, Any]) -> None:
        """Restore params from a dict (loaded from JSON)."""
        for k, v in data.items():
            if k in self.params:
                self.params[k] = v

    def __repr__(self) -> str:
        return f"<{self.op_type} id={self.op_id}>"


def _serialise_val(v: Any) -> Any:
    """Make a value JSON‑safe."""
    if v is None or isinstance(v, (int, float, str, bool)):
        return v
    if isinstance(v, (list, tuple)):
        return [_serialise_val(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _serialise_val(val) for k, val in v.items()}
    return str(v)


# ── Global operator registry ───────────────────────────────────────────────

_OPERATOR_REGISTRY: Dict[str, Type[Operator]] = {}


def register_operator(cls: Type[Operator]) -> Type[Operator]:
    """Class decorator – adds *cls* to the global registry."""
    _OPERATOR_REGISTRY[cls.op_type] = cls
    return cls


def get_operator_class(op_type: str) -> Type[Operator]:
    """Retrieve a registered operator class by its ``op_type`` string."""
    if op_type not in _OPERATOR_REGISTRY:
        raise KeyError(f"Unknown operator type: {op_type!r}")
    return _OPERATOR_REGISTRY[op_type]


def list_operator_types() -> List[str]:
    """Return all registered op_type strings."""
    return list(_OPERATOR_REGISTRY.keys())


def operators_by_category() -> Dict[OpCategory, List[Type[Operator]]]:
    """Group registered operators by category for the palette."""
    groups: Dict[OpCategory, List[Type[Operator]]] = {}
    for cls in _OPERATOR_REGISTRY.values():
        groups.setdefault(cls.category, []).append(cls)
    return groups


# ── Connection data class ──────────────────────────────────────────────────

@dataclass
class Connection:
    """Describes a wire between two operator ports."""
    from_op_id: str
    from_port: str
    to_op_id: str
    to_port: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "from": self.from_op_id,
            "from_port": self.from_port,
            "to": self.to_op_id,
            "to_port": self.to_port,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> "Connection":
        return cls(
            from_op_id=d["from"],
            from_port=d["from_port"],
            to_op_id=d["to"],
            to_port=d["to_port"],
        )
