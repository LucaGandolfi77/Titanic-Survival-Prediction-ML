"""
test_operators_data.py – Tests for data‑access operators.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from engine.operator_base import get_operator_class


# ═══════════════════════════════════════════════════════════════════════════
# Read CSV
# ═══════════════════════════════════════════════════════════════════════════

class TestReadCSV:
    def test_basic_read(self, iris_csv):
        op = get_operator_class("Read CSV")()
        op.set_param("filepath", str(iris_csv))
        out = op.execute({})
        df = out["out"]
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 30
        assert "species" in df.columns

    def test_missing_file_raises(self, tmp_dir):
        op = get_operator_class("Read CSV")()
        op.set_param("filepath", str(tmp_dir / "nope.csv"))
        with pytest.raises(FileNotFoundError):
            op.execute({})

    def test_semicolon_separator(self, tmp_dir):
        p = tmp_dir / "semi.csv"
        p.write_text("a;b\n1;2\n3;4\n")
        op = get_operator_class("Read CSV")()
        op.set_param("filepath", str(p))
        op.set_param("separator", ";")
        df = op.execute({})["out"]
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 2

    def test_no_header(self, tmp_dir):
        p = tmp_dir / "nohead.csv"
        p.write_text("1,2,3\n4,5,6\n")
        op = get_operator_class("Read CSV")()
        op.set_param("filepath", str(p))
        op.set_param("header", False)
        df = op.execute({})["out"]
        assert len(df) == 2
        assert len(df.columns) == 3


# ═══════════════════════════════════════════════════════════════════════════
# Read JSON
# ═══════════════════════════════════════════════════════════════════════════

class TestReadJSON:
    def test_basic_read(self, tmp_dir, iris_df):
        p = tmp_dir / "data.json"
        iris_df.to_json(p, orient="records")
        op = get_operator_class("Read JSON")()
        op.set_param("filepath", str(p))
        df = op.execute({})["out"]
        assert len(df) == 30

    def test_missing_file(self, tmp_dir):
        op = get_operator_class("Read JSON")()
        op.set_param("filepath", str(tmp_dir / "missing.json"))
        with pytest.raises(FileNotFoundError):
            op.execute({})


# ═══════════════════════════════════════════════════════════════════════════
# Read Database
# ═══════════════════════════════════════════════════════════════════════════

class TestReadDatabase:
    def test_sqlite_read(self, tmp_dir, iris_df):
        db_path = tmp_dir / "test.db"
        conn = sqlite3.connect(str(db_path))
        iris_df.to_sql("iris", conn, index=False)
        conn.close()

        op = get_operator_class("Read Database")()
        op.set_param("db_path", str(db_path))
        op.set_param("query", "SELECT * FROM iris")
        df = op.execute({})["out"]
        assert len(df) == 30

    def test_missing_db(self, tmp_dir):
        op = get_operator_class("Read Database")()
        op.set_param("db_path", str(tmp_dir / "nope.db"))
        op.set_param("query", "SELECT 1")
        with pytest.raises(FileNotFoundError):
            op.execute({})


# ═══════════════════════════════════════════════════════════════════════════
# Write CSV
# ═══════════════════════════════════════════════════════════════════════════

class TestWriteCSV:
    def test_write_and_passthrough(self, tmp_dir, iris_df):
        out_path = tmp_dir / "written.csv"
        op = get_operator_class("Write CSV")()
        op.set_param("filepath", str(out_path))
        result = op.execute({"in": iris_df})
        assert "out" in result
        assert out_path.exists()
        reloaded = pd.read_csv(out_path)
        assert len(reloaded) == 30


# ═══════════════════════════════════════════════════════════════════════════
# Store / Retrieve
# ═══════════════════════════════════════════════════════════════════════════

class TestStoreRetrieve:
    def test_roundtrip(self, iris_df):
        store = get_operator_class("Store")()
        store.set_param("name", "test_data")
        store.execute({"in": iris_df})

        retrieve = get_operator_class("Retrieve")()
        retrieve.set_param("name", "test_data")
        # Share the same class‑level repository
        df = retrieve.execute({})["out"]
        assert len(df) == 30

    def test_retrieve_missing_key(self):
        retrieve = get_operator_class("Retrieve")()
        retrieve.set_param("name", "nonexistent_key_xyz")
        with pytest.raises(KeyError):
            retrieve.execute({})
