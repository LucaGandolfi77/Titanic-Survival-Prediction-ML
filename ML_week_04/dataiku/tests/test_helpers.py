"""Tests for utils/helpers.py"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.helpers import (
    NumpyEncoder,
    column_stats,
    dataframe_memory,
    detect_column_type,
    format_bytes,
    load_json,
    read_dataframe,
    save_json,
    timestamp_str,
    write_dataframe,
)


# ---------------------------------------------------------------------------
# format_bytes
# ---------------------------------------------------------------------------

class TestFormatBytes:
    def test_bytes(self) -> None:
        assert format_bytes(500) == "500.0 B"

    def test_kilobytes(self) -> None:
        assert format_bytes(2048) == "2.0 KB"

    def test_megabytes(self) -> None:
        assert format_bytes(1_048_576) == "1.0 MB"

    def test_zero(self) -> None:
        assert format_bytes(0) == "0.0 B"


# ---------------------------------------------------------------------------
# detect_column_type
# ---------------------------------------------------------------------------

class TestDetectColumnType:
    def test_numeric(self) -> None:
        s = pd.Series([1, 2, 3, 4, 5])
        assert detect_column_type(s) == "numeric"

    def test_boolean(self) -> None:
        s = pd.Series([True, False, True])
        assert detect_column_type(s) == "boolean"

    def test_categorical(self) -> None:
        s = pd.Series(["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"])
        assert detect_column_type(s) == "categorical"

    def test_text(self) -> None:
        s = pd.Series(["hello world", "foo bar baz", "unique string 123"])
        assert detect_column_type(s) == "text"

    def test_datetime(self) -> None:
        s = pd.Series(pd.to_datetime(["2021-01-01", "2021-06-15", "2022-12-31"]))
        assert detect_column_type(s) == "datetime"


# ---------------------------------------------------------------------------
# column_stats
# ---------------------------------------------------------------------------

class TestColumnStats:
    def test_numeric_stats(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, None, 5.0]})
        stats = column_stats(df, "x")
        assert stats["dtype"] == "float64"
        assert stats["null_count"] == 1
        assert stats["unique_count"] == 4
        assert stats["count"] == 4
        assert pytest.approx(stats["min"]) == 1.0
        assert pytest.approx(stats["max"]) == 5.0

    def test_string_stats(self) -> None:
        df = pd.DataFrame({"c": ["a", "b", "a", None]})
        stats = column_stats(df, "c")
        assert stats["null_count"] == 1
        assert stats["unique_count"] == 2
        assert "mean" not in stats  # not numeric


# ---------------------------------------------------------------------------
# dataframe_memory
# ---------------------------------------------------------------------------

class TestDataframeMemory:
    def test_returns_string(self) -> None:
        df = pd.DataFrame({"a": range(100)})
        mem = dataframe_memory(df)
        assert isinstance(mem, str)
        assert any(u in mem for u in ("B", "KB", "MB"))


# ---------------------------------------------------------------------------
# read / write dataframe round-trip
# ---------------------------------------------------------------------------

class TestReadWriteDataframe:
    def _roundtrip(self, ext: str) -> None:
        df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"data{ext}"
            write_dataframe(df, path)
            assert path.exists()
            df2 = read_dataframe(path)
            assert list(df2.columns) == list(df.columns)
            assert len(df2) == len(df)

    def test_csv(self) -> None:
        self._roundtrip(".csv")

    def test_json(self) -> None:
        self._roundtrip(".json")

    def test_parquet(self) -> None:
        self._roundtrip(".parquet")

    def test_tsv(self) -> None:
        self._roundtrip(".tsv")

    def test_unsupported_ext(self, tmp_path: str) -> None:
        f = Path(tmp_path) / "data.xyz"
        f.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported"):
            read_dataframe(f)

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            read_dataframe(Path("/tmp/nonexistent_file.csv"))


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

class TestJsonHelpers:
    def test_save_load_roundtrip(self) -> None:
        data = {"key": "value", "nums": [1, 2, 3]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.json"
            save_json(data, path)
            loaded = load_json(path)
            assert loaded == data

    def test_numpy_encoder(self) -> None:
        data = {"a": np.int64(42), "b": np.float64(3.14), "c": np.array([1, 2])}
        result = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(result)
        assert parsed["a"] == 42
        assert pytest.approx(parsed["b"]) == 3.14
        assert parsed["c"] == [1, 2]


# ---------------------------------------------------------------------------
# timestamp_str
# ---------------------------------------------------------------------------

class TestTimestamp:
    def test_format(self) -> None:
        ts = timestamp_str()
        assert "T" in ts
        assert len(ts) >= 19
