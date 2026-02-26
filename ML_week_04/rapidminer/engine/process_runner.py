"""
process_runner.py – DAG builder + topological executor.

Builds a directed acyclic graph from operators and connections, performs a
topological sort, and executes each operator in order while pushing outputs
through connected wires.  Execution runs in a background thread.
"""
from __future__ import annotations

import logging
import time
import threading
import traceback
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from engine.operator_base import Connection, Operator

logger = logging.getLogger(__name__)


# ── Utility operators (registered here to keep operator_base lean) ─────

from engine.operator_base import (
    OpCategory,
    ParamKind,
    ParamSpec,
    Port,
    PortType,
    register_operator,
)


@register_operator
class LogToConsole(Operator):
    op_type = "Log to Console"
    category = OpCategory.UTILITY
    description = "Log a summary of the input ExampleSet to the execution log."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = []

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import pandas as pd
        df = inputs["in"]
        if isinstance(df, pd.DataFrame):
            logger.info("ExampleSet: %d rows × %d cols  Memory: %s",
                        len(df), len(df.columns),
                        f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        return {"out": df}


@register_operator
class SetMacro(Operator):
    op_type = "Set Macro"
    category = OpCategory.UTILITY
    description = "Set a global macro variable (name → value)."

    _macros: Dict[str, str] = {}

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.ANY, optional=True)
        self.outputs["out"] = Port("out", PortType.ANY)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("name", ParamKind.STRING, default="my_macro", description="Macro name."),
            ParamSpec("value", ParamKind.STRING, default="", description="Macro value."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        name = self.get_param("name")
        value = self.get_param("value")
        self._macros[name] = value
        logger.info("Macro set: %s = %s", name, value)
        passthrough = inputs.get("in")
        return {"out": passthrough}


@register_operator
class BranchOp(Operator):
    op_type = "Branch"
    category = OpCategory.UTILITY
    description = "Conditional routing: if macro condition is true → out_true, else → out_false."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["out_true"] = Port("out_true", PortType.EXAMPLE_SET)
        self.outputs["out_false"] = Port("out_false", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("condition", ParamKind.TEXT, default="True", description="Python expression evaluated as bool."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        df = inputs["in"]
        cond = self.get_param("condition") or "True"
        macros = SetMacro._macros
        try:
            result = bool(eval(cond, {"__builtins__": {}}, macros))  # noqa: S307
        except Exception:
            result = True
        if result:
            return {"out_true": df, "out_false": None}
        else:
            return {"out_true": None, "out_false": df}


# ═══════════════════════════════════════════════════════════════════════════
# Process — the "document" that holds operators + connections
# ═══════════════════════════════════════════════════════════════════════════

class Process:
    """A RapidMiner‑Lite process: a collection of operators linked by connections."""

    def __init__(self, name: str = "New Process") -> None:
        self.name: str = name
        self.version: str = "1.0"
        self.operators: Dict[str, Operator] = {}
        self.connections: List[Connection] = []

    # ── mutators ────────────────────────────────────────────────────────

    def add_operator(self, op: Operator) -> None:
        self.operators[op.op_id] = op

    def remove_operator(self, op_id: str) -> None:
        self.operators.pop(op_id, None)
        self.connections = [
            c for c in self.connections
            if c.from_op_id != op_id and c.to_op_id != op_id
        ]

    def add_connection(self, conn: Connection) -> None:
        self.connections.append(conn)

    def remove_connection(self, conn: Connection) -> None:
        self.connections = [
            c for c in self.connections
            if not (c.from_op_id == conn.from_op_id and c.from_port == conn.from_port
                    and c.to_op_id == conn.to_op_id and c.to_port == conn.to_port)
        ]

    # ── serialisation ───────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "operators": [op.to_dict() for op in self.operators.values()],
            "connections": [c.to_dict() for c in self.connections],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Process":
        from engine.operator_base import get_operator_class
        proc = cls(name=data.get("name", "Loaded Process"))
        proc.version = data.get("version", "1.0")
        for od in data.get("operators", []):
            op_cls = get_operator_class(od["type"])
            op = op_cls()
            op.op_id = od["id"]
            op.display_name = od.get("display_name", od["type"])
            pos = od.get("position", [100, 100])
            op.x, op.y = pos[0], pos[1]
            op.load_params(od.get("params", {}))
            proc.add_operator(op)
        for cd in data.get("connections", []):
            proc.add_connection(Connection.from_dict(cd))
        return proc


# ═══════════════════════════════════════════════════════════════════════════
# DAG builder + topological sort
# ═══════════════════════════════════════════════════════════════════════════

def _build_dag(process: Process) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:
    """Return adjacency list (children) and in‑degree map."""
    adj: Dict[str, Set[str]] = defaultdict(set)
    in_deg: Dict[str, int] = {oid: 0 for oid in process.operators}
    for c in process.connections:
        if c.from_op_id in process.operators and c.to_op_id in process.operators:
            if c.to_op_id not in adj[c.from_op_id]:
                adj[c.from_op_id].add(c.to_op_id)
                in_deg[c.to_op_id] = in_deg.get(c.to_op_id, 0) + 1
    return dict(adj), in_deg


def topological_sort(process: Process) -> List[str]:
    """Kahn's algorithm — returns operator IDs in execution order."""
    adj, in_deg = _build_dag(process)
    queue: deque[str] = deque()
    for oid in process.operators:
        if in_deg.get(oid, 0) == 0:
            queue.append(oid)

    order: List[str] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in adj.get(node, set()):
            in_deg[child] -= 1
            if in_deg[child] == 0:
                queue.append(child)

    if len(order) != len(process.operators):
        raise RuntimeError("Process graph contains a cycle!")
    return order


# ═══════════════════════════════════════════════════════════════════════════
# ProcessRunner – executes a Process end‑to‑end
# ═══════════════════════════════════════════════════════════════════════════

class ProcessRunner:
    """Execute an entire Process in topological order.

    Execution emits callbacks so the GUI can update:
      - on_start(total_operators)
      - on_operator_start(op_id, op_type)
      - on_operator_done(op_id, op_type, elapsed)
      - on_operator_error(op_id, op_type, error_str)
      - on_complete(total_time, results)
      - on_log(message)
    """

    def __init__(self) -> None:
        self.on_start: Optional[Callable] = None
        self.on_operator_start: Optional[Callable] = None
        self.on_operator_done: Optional[Callable] = None
        self.on_operator_error: Optional[Callable] = None
        self.on_complete: Optional[Callable] = None
        self.on_log: Optional[Callable] = None
        self._stop_flag = threading.Event()

    # ── public API ──────────────────────────────────────────────────────

    def run(self, process: Process) -> Dict[str, Any]:
        """Run synchronously.  Returns final output dict {op_id: outputs}."""
        self._stop_flag.clear()
        order = topological_sort(process)
        total = len(order)
        self._emit(self.on_start, total)
        self._emit(self.on_log, f"▸ Process '{process.name}' started  ({total} operators)")

        # Reset all operators
        for op in process.operators.values():
            op.reset_execution()

        results: Dict[str, Dict[str, Any]] = {}
        t0 = time.perf_counter()

        for idx, op_id in enumerate(order):
            if self._stop_flag.is_set():
                self._emit(self.on_log, "⚠ Process stopped by user.")
                break

            op = process.operators[op_id]
            self._emit(self.on_operator_start, op_id, op.display_name)
            self._emit(self.on_log, f"  [{idx+1}/{total}] Running {op.display_name}…")

            # Gather inputs from upstream connections
            input_data: Dict[str, Any] = {}
            for conn in process.connections:
                if conn.to_op_id == op_id:
                    src_outputs = results.get(conn.from_op_id, {})
                    input_data[conn.to_port] = src_outputs.get(conn.from_port)

            ts = time.perf_counter()
            try:
                outputs = op.execute(input_data)
                op.executed = True
                elapsed = time.perf_counter() - ts
                op.exec_time = elapsed
                results[op_id] = outputs
                # Store values in output ports for inspection
                for port_name, val in outputs.items():
                    if port_name in op.outputs:
                        op.outputs[port_name].value = val
                self._emit(self.on_operator_done, op_id, op.display_name, elapsed)
                self._emit(self.on_log, f"    ✓ {op.display_name}  ({elapsed:.3f}s)")
            except Exception as exc:
                elapsed = time.perf_counter() - ts
                op.exec_time = elapsed
                op.error = traceback.format_exc()
                results[op_id] = {}
                self._emit(self.on_operator_error, op_id, op.display_name, str(exc))
                self._emit(self.on_log, f"    ✗ {op.display_name}: {exc}")
                logger.exception("Operator %s failed", op.display_name)

        total_time = time.perf_counter() - t0
        self._emit(self.on_log, f"▸ Process completed in {total_time:.3f}s")
        self._emit(self.on_complete, total_time, results)
        return results

    def run_async(self, process: Process) -> threading.Thread:
        """Run in a background thread."""
        t = threading.Thread(target=self.run, args=(process,), daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        """Signal the runner to stop after the current operator."""
        self._stop_flag.set()

    # ── internal ────────────────────────────────────────────────────────

    @staticmethod
    def _emit(cb: Optional[Callable], *args: Any) -> None:
        if cb is not None:
            try:
                cb(*args)
            except Exception:
                pass
