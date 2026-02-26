"""
NotebookManager – Jupyter-like code cells with a shared namespace.

Each cell has: id, source code, output text, execution order.
Cells share a global dict so variables persist between runs.
"""
from __future__ import annotations

import io
import sys
import traceback
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Cell data
# ---------------------------------------------------------------------------

class NotebookCell:
    """One code cell inside the notebook."""

    def __init__(
        self,
        source: str = "",
        cell_id: Optional[str] = None,
    ) -> None:
        self.cell_id: str = cell_id or uuid.uuid4().hex[:8]
        self.source: str = source
        self.output: str = ""
        self.execution_count: int = 0
        self.error: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "source": self.source,
            "output": self.output,
            "execution_count": self.execution_count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NotebookCell":
        c = cls(source=d.get("source", ""), cell_id=d.get("cell_id"))
        c.output = d.get("output", "")
        c.execution_count = d.get("execution_count", 0)
        return c


# ---------------------------------------------------------------------------
# NotebookManager
# ---------------------------------------------------------------------------

class NotebookManager:
    """Manages a list of code cells with a shared execution namespace.

    The namespace always contains ``pd``, ``np``, and can be seeded
    with DataFrames via :meth:`inject`.
    """

    def __init__(self) -> None:
        self.cells: List[NotebookCell] = []
        self._namespace: Dict[str, Any] = {
            "pd": pd,
            "np": np,
            "__builtins__": __builtins__,
        }
        self._exec_counter: int = 0

    # -- cell management -----------------------------------------------------

    def add_cell(self, source: str = "") -> NotebookCell:
        """Append a new cell and return it."""
        cell = NotebookCell(source=source)
        self.cells.append(cell)
        return cell

    def insert_cell(self, index: int, source: str = "") -> NotebookCell:
        """Insert a cell at *index*."""
        cell = NotebookCell(source=source)
        self.cells.insert(index, cell)
        return cell

    def delete_cell(self, cell_id: str) -> None:
        """Remove a cell by id."""
        self.cells = [c for c in self.cells if c.cell_id != cell_id]

    def move_cell(self, cell_id: str, direction: int) -> None:
        """Move a cell up (-1) or down (+1)."""
        idx = self._index_of(cell_id)
        if idx is None:
            return
        new_idx = idx + direction
        if 0 <= new_idx < len(self.cells):
            self.cells[idx], self.cells[new_idx] = self.cells[new_idx], self.cells[idx]

    def _index_of(self, cell_id: str) -> Optional[int]:
        for i, c in enumerate(self.cells):
            if c.cell_id == cell_id:
                return i
        return None

    # -- execution -----------------------------------------------------------

    def run_cell(self, cell_id: str) -> str:
        """Execute a single cell and return its captured output.

        Stdout/stderr are captured; exceptions are caught and displayed
        as the cell output.
        """
        cell = self._get(cell_id)
        if cell is None:
            return "Cell not found."

        self._exec_counter += 1
        cell.execution_count = self._exec_counter

        old_stdout, old_stderr = sys.stdout, sys.stderr
        buf = io.StringIO()
        sys.stdout = buf
        sys.stderr = buf

        try:
            # compile + exec to allow both expressions and statements
            code = compile(cell.source, f"<cell {cell.cell_id}>", "exec")
            exec(code, self._namespace)
            cell.error = False
        except Exception:
            traceback.print_exc(file=buf)
            cell.error = True
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        cell.output = buf.getvalue()
        return cell.output

    def run_all(self) -> None:
        """Execute every cell in order."""
        for cell in self.cells:
            self.run_cell(cell.cell_id)

    # -- namespace -----------------------------------------------------------

    def inject(self, name: str, obj: Any) -> None:
        """Inject a variable into the shared namespace."""
        self._namespace[name] = obj

    def reset_namespace(self) -> None:
        """Clear the namespace (keep pd/np)."""
        self._namespace = {
            "pd": pd,
            "np": np,
            "__builtins__": __builtins__,
        }
        self._exec_counter = 0

    # -- export --------------------------------------------------------------

    def export_as_py(self) -> str:
        """Return all cell sources concatenated as a .py script."""
        lines = ["#!/usr/bin/env python3", "# Exported from DataikuLite Notebook", ""]
        for i, cell in enumerate(self.cells):
            lines.append(f"# --- Cell {i + 1} ---")
            lines.append(cell.source)
            lines.append("")
        return "\n".join(lines)

    # -- serialisation -------------------------------------------------------

    def to_list(self) -> List[Dict[str, Any]]:
        return [c.to_dict() for c in self.cells]

    def load_from_list(self, data: List[Dict[str, Any]]) -> None:
        self.cells = [NotebookCell.from_dict(d) for d in data]

    # -- internal ------------------------------------------------------------

    def _get(self, cell_id: str) -> Optional[NotebookCell]:
        for c in self.cells:
            if c.cell_id == cell_id:
                return c
        return None
