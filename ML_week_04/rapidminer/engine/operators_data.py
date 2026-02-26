"""
operators_data.py – Data‑access operators: Read CSV / Excel / JSON / DB,
Write CSV, Retrieve, Store.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from engine.operator_base import (
    Operator,
    OpCategory,
    ParamKind,
    ParamSpec,
    Port,
    PortType,
    register_operator,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Read CSV
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class ReadCSV(Operator):
    """Read a CSV file into an ExampleSet."""

    op_type = "Read CSV"
    category = OpCategory.DATA
    description = "Reads a CSV file and outputs an ExampleSet."

    def _build_ports(self) -> None:
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("filepath", ParamKind.FILEPATH, default="", description="Path to the CSV file."),
            ParamSpec("separator", ParamKind.CHOICE, default=",", choices=[",", ";", "\\t", "|"], description="Column separator."),
            ParamSpec("header", ParamKind.BOOL, default=True, description="First row is header."),
            ParamSpec("encoding", ParamKind.CHOICE, default="utf-8", choices=["utf-8", "latin-1", "cp1252"], description="File encoding."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        fp = Path(self.get_param("filepath"))
        if not fp.exists():
            raise FileNotFoundError(f"CSV file not found: {fp}")
        sep = self.get_param("separator")
        if sep == "\\t":
            sep = "\t"
        header = 0 if self.get_param("header") else None
        encoding = self.get_param("encoding")
        df = pd.read_csv(fp, sep=sep, header=header, encoding=encoding)
        logger.info("Read CSV: %s  (%d rows, %d cols)", fp.name, len(df), len(df.columns))
        return {"out": df}


# ═══════════════════════════════════════════════════════════════════════════
# Read Excel
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class ReadExcel(Operator):
    """Read an Excel file (.xlsx/.xls) into an ExampleSet."""

    op_type = "Read Excel"
    category = OpCategory.DATA
    description = "Reads an Excel worksheet and outputs an ExampleSet."

    def _build_ports(self) -> None:
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("filepath", ParamKind.FILEPATH, default="", description="Path to the Excel file."),
            ParamSpec("sheet", ParamKind.STRING, default="Sheet1", description="Sheet name or index."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        fp = Path(self.get_param("filepath"))
        if not fp.exists():
            raise FileNotFoundError(f"Excel file not found: {fp}")
        sheet = self.get_param("sheet")
        try:
            sheet_idx = int(sheet)
            df = pd.read_excel(fp, sheet_name=sheet_idx)
        except (ValueError, TypeError):
            df = pd.read_excel(fp, sheet_name=sheet)
        logger.info("Read Excel: %s [%s]  (%d rows)", fp.name, sheet, len(df))
        return {"out": df}


# ═══════════════════════════════════════════════════════════════════════════
# Read JSON
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class ReadJSON(Operator):
    """Read a JSON file into an ExampleSet."""

    op_type = "Read JSON"
    category = OpCategory.DATA
    description = "Reads a JSON file and outputs an ExampleSet."

    def _build_ports(self) -> None:
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("filepath", ParamKind.FILEPATH, default="", description="Path to the JSON file."),
            ParamSpec("orient", ParamKind.CHOICE, default="records", choices=["records", "columns", "index", "split"], description="JSON orientation."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        fp = Path(self.get_param("filepath"))
        if not fp.exists():
            raise FileNotFoundError(f"JSON file not found: {fp}")
        orient = self.get_param("orient")
        df = pd.read_json(fp, orient=orient)
        logger.info("Read JSON: %s  (%d rows)", fp.name, len(df))
        return {"out": df}


# ═══════════════════════════════════════════════════════════════════════════
# Read Database (SQLite)
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class ReadDatabase(Operator):
    """Execute a SQL query against an SQLite database."""

    op_type = "Read Database"
    category = OpCategory.DATA
    description = "Reads from an SQLite database using a SQL query."

    def _build_ports(self) -> None:
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("db_path", ParamKind.FILEPATH, default="", description="Path to the SQLite .db file."),
            ParamSpec("query", ParamKind.TEXT, default="SELECT * FROM table_name", description="SQL query to execute."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        db = Path(self.get_param("db_path"))
        if not db.exists():
            raise FileNotFoundError(f"Database not found: {db}")
        query = self.get_param("query")
        conn = sqlite3.connect(str(db))
        try:
            df = pd.read_sql_query(query, conn)
        finally:
            conn.close()
        logger.info("Read DB: %s  (%d rows)", db.name, len(df))
        return {"out": df}


# ═══════════════════════════════════════════════════════════════════════════
# Write CSV
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class WriteCSV(Operator):
    """Write an ExampleSet to a CSV file (pass‑through)."""

    op_type = "Write CSV"
    category = OpCategory.DATA
    description = "Writes the input ExampleSet to a CSV file and passes it through."

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("filepath", ParamKind.FILEPATH, default="output.csv", description="Destination CSV path."),
            ParamSpec("separator", ParamKind.CHOICE, default=",", choices=[",", ";", "\\t", "|"], description="Column separator."),
            ParamSpec("index", ParamKind.BOOL, default=False, description="Write row index."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = inputs["in"]
        fp = Path(self.get_param("filepath"))
        fp.parent.mkdir(parents=True, exist_ok=True)
        sep = self.get_param("separator")
        if sep == "\\t":
            sep = "\t"
        df.to_csv(fp, sep=sep, index=self.get_param("index"))
        logger.info("Wrote CSV: %s  (%d rows)", fp.name, len(df))
        return {"out": df}


# ═══════════════════════════════════════════════════════════════════════════
# Retrieve (from repository)
# ═══════════════════════════════════════════════════════════════════════════

# Shared in‑memory repository for Store / Retrieve operators
_shared_repository: Dict[str, Any] = {}


@register_operator
class Retrieve(Operator):
    """Load an ExampleSet that was previously stored in the repository."""

    op_type = "Retrieve"
    category = OpCategory.DATA
    description = "Retrieves a stored ExampleSet from the repository by name."

    _repository = _shared_repository

    def _build_ports(self) -> None:
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("name", ParamKind.STRING, default="", description="Variable name in repository."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        name = self.get_param("name")
        if name not in self._repository:
            raise KeyError(f"Repository has no entry called '{name}'")
        obj = self._repository[name]
        logger.info("Retrieved '%s' from repository", name)
        return {"out": obj}


# ═══════════════════════════════════════════════════════════════════════════
# Store (to repository)
# ═══════════════════════════════════════════════════════════════════════════

@register_operator
class Store(Operator):
    """Save an ExampleSet into the repository for later use."""

    op_type = "Store"
    category = OpCategory.DATA
    description = "Stores the input ExampleSet in the repository under a name."

    _repository = _shared_repository

    def _build_ports(self) -> None:
        self.inputs["in"] = Port("in", PortType.EXAMPLE_SET)
        self.outputs["out"] = Port("out", PortType.EXAMPLE_SET)

    def _build_params(self) -> None:
        self.param_specs = [
            ParamSpec("name", ParamKind.STRING, default="stored_data", description="Repository variable name."),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = inputs["in"]
        name = self.get_param("name")
        _shared_repository[name] = df.copy()
        logger.info("Stored '%s' in repository  (%d rows)", name, len(df))
        return {"out": df}
