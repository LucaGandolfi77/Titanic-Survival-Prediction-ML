"""Tests for core/notebook.py"""
from __future__ import annotations

import pytest

from core.notebook import NotebookCell, NotebookManager


# ---------------------------------------------------------------------------
# NotebookCell
# ---------------------------------------------------------------------------

class TestNotebookCell:
    def test_defaults(self) -> None:
        cell = NotebookCell()
        assert cell.source == ""
        assert cell.output == ""
        assert cell.execution_count == 0
        assert len(cell.cell_id) > 0

    def test_to_from_dict(self) -> None:
        cell = NotebookCell(source="x = 1")
        cell.output = "done"
        cell.execution_count = 3
        d = cell.to_dict()
        cell2 = NotebookCell.from_dict(d)
        assert cell2.source == "x = 1"
        assert cell2.output == "done"
        assert cell2.execution_count == 3
        assert cell2.cell_id == cell.cell_id


# ---------------------------------------------------------------------------
# NotebookManager – cell management
# ---------------------------------------------------------------------------

class TestCellManagement:
    def test_add_cell(self) -> None:
        nb = NotebookManager()
        c = nb.add_cell("print('hi')")
        assert c.source == "print('hi')"
        assert len(nb.cells) == 1

    def test_insert_cell(self) -> None:
        nb = NotebookManager()
        nb.add_cell("first")
        nb.add_cell("third")
        nb.insert_cell(1, "second")
        assert nb.cells[1].source == "second"
        assert len(nb.cells) == 3

    def test_delete_cell(self) -> None:
        nb = NotebookManager()
        c = nb.add_cell("x")
        nb.add_cell("y")
        nb.delete_cell(c.cell_id)
        assert len(nb.cells) == 1
        assert nb.cells[0].source == "y"

    def test_delete_nonexistent(self) -> None:
        nb = NotebookManager()
        nb.add_cell("a")
        nb.delete_cell("nonexistent")
        assert len(nb.cells) == 1  # no change

    def test_move_cell_down(self) -> None:
        nb = NotebookManager()
        c1 = nb.add_cell("first")
        nb.add_cell("second")
        nb.move_cell(c1.cell_id, 1)
        assert nb.cells[0].source == "second"
        assert nb.cells[1].source == "first"

    def test_move_cell_up(self) -> None:
        nb = NotebookManager()
        nb.add_cell("first")
        c2 = nb.add_cell("second")
        nb.move_cell(c2.cell_id, -1)
        assert nb.cells[0].source == "second"

    def test_move_out_of_bounds(self) -> None:
        nb = NotebookManager()
        c = nb.add_cell("only")
        nb.move_cell(c.cell_id, -1)  # should not crash
        assert nb.cells[0].source == "only"


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

class TestExecution:
    def test_run_cell_captures_stdout(self) -> None:
        nb = NotebookManager()
        c = nb.add_cell("print('hello world')")
        output = nb.run_cell(c.cell_id)
        assert "hello world" in output
        assert c.execution_count == 1

    def test_run_cell_handles_error(self) -> None:
        nb = NotebookManager()
        c = nb.add_cell("1 / 0")
        output = nb.run_cell(c.cell_id)
        assert "ZeroDivisionError" in output
        assert c.error is True

    def test_shared_namespace(self) -> None:
        nb = NotebookManager()
        c1 = nb.add_cell("my_var = 42")
        c2 = nb.add_cell("print(my_var * 2)")
        nb.run_cell(c1.cell_id)
        output = nb.run_cell(c2.cell_id)
        assert "84" in output

    def test_run_all(self) -> None:
        nb = NotebookManager()
        nb.add_cell("a = 10")
        nb.add_cell("b = a + 5")
        nb.add_cell("print(b)")
        nb.run_all()
        assert "15" in nb.cells[2].output

    def test_execution_count_increments(self) -> None:
        nb = NotebookManager()
        c = nb.add_cell("x = 1")
        nb.run_cell(c.cell_id)
        assert c.execution_count == 1
        nb.run_cell(c.cell_id)
        assert c.execution_count == 2

    def test_nonexistent_cell(self) -> None:
        nb = NotebookManager()
        output = nb.run_cell("fake_id")
        assert "not found" in output.lower()

    def test_pandas_numpy_available(self) -> None:
        nb = NotebookManager()
        c = nb.add_cell("import pandas as pd; print(pd.__version__)")
        output = nb.run_cell(c.cell_id)
        assert c.error is False
        assert output.strip() != ""


# ---------------------------------------------------------------------------
# Namespace injection
# ---------------------------------------------------------------------------

class TestNamespace:
    def test_inject(self) -> None:
        nb = NotebookManager()
        nb.inject("my_list", [1, 2, 3])
        c = nb.add_cell("print(sum(my_list))")
        output = nb.run_cell(c.cell_id)
        assert "6" in output

    def test_reset_namespace(self) -> None:
        nb = NotebookManager()
        nb.inject("my_list", [1, 2, 3])
        nb.reset_namespace()
        c = nb.add_cell("print(my_list)")
        output = nb.run_cell(c.cell_id)
        assert "NameError" in output


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_as_py(self) -> None:
        nb = NotebookManager()
        nb.add_cell("x = 1")
        nb.add_cell("y = x + 2")
        script = nb.export_as_py()
        assert "x = 1" in script
        assert "y = x + 2" in script
        assert "Cell 1" in script
        assert "Cell 2" in script


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestSerialisation:
    def test_to_from_list(self) -> None:
        nb = NotebookManager()
        nb.add_cell("a = 1")
        nb.add_cell("b = 2")
        data = nb.to_list()
        assert len(data) == 2

        nb2 = NotebookManager()
        nb2.load_from_list(data)
        assert len(nb2.cells) == 2
        assert nb2.cells[0].source == "a = 1"
        assert nb2.cells[1].source == "b = 2"
