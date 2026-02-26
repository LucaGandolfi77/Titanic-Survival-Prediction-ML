"""
Utility helpers for DataikuLite DSS.

Provides common functions used across the application: file I/O,
memory formatting, type detection, and thread-safe GUI callbacks.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS: Dict[str, str] = {
    ".csv": "CSV files",
    ".tsv": "TSV files",
    ".xlsx": "Excel files",
    ".xls": "Excel files",
    ".json": "JSON files",
    ".parquet": "Parquet files",
    ".pq": "Parquet files",
}


def read_dataframe(path: Union[str, Path]) -> pd.DataFrame:
    """Read a file into a DataFrame, auto-detecting format from extension.

    Args:
        path: Path to the data file.

    Returns:
        A pandas DataFrame.

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext in (".csv", ".tsv"):
        sep = "\t" if ext == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    elif ext == ".json":
        return pd.read_json(path)
    elif ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def write_dataframe(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """Write a DataFrame to disk, auto-detecting format from extension.

    Args:
        df: The DataFrame to write.
        path: Destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext in (".csv", ".tsv"):
        sep = "\t" if ext == ".tsv" else ","
        df.to_csv(path, sep=sep, index=False)
    elif ext in (".xlsx", ".xls"):
        df.to_excel(path, index=False)
    elif ext == ".json":
        df.to_json(path, orient="records", indent=2)
    elif ext in (".parquet", ".pq"):
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


# ---------------------------------------------------------------------------
# Column type detection
# ---------------------------------------------------------------------------

def detect_column_type(series: pd.Series) -> str:
    """Detect the semantic type of a pandas Series.

    Returns one of: 'numeric', 'categorical', 'datetime', 'text', 'boolean'.
    """
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    # Attempt to parse as datetime
    try:
        pd.to_datetime(series.dropna().head(20))
        return "datetime"
    except (ValueError, TypeError):
        pass
    # Heuristic: if nunique / count < 0.5 → categorical
    nunique = series.nunique()
    count = series.count()
    if count > 0 and nunique / count < 0.5:
        return "categorical"
    return "text"


def column_stats(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """Compute summary statistics for a single column.

    Args:
        df: Source DataFrame.
        col: Column name.

    Returns:
        A dict with dtype, null_count, unique_count, and optional min/max/mean.
    """
    s = df[col]
    stats: Dict[str, Any] = {
        "dtype": str(s.dtype),
        "semantic_type": detect_column_type(s),
        "null_count": int(s.isna().sum()),
        "unique_count": int(s.nunique()),
        "count": int(s.count()),
    }
    if pd.api.types.is_numeric_dtype(s):
        stats["min"] = float(s.min()) if s.count() > 0 else None
        stats["max"] = float(s.max()) if s.count() > 0 else None
        stats["mean"] = float(s.mean()) if s.count() > 0 else None
        stats["std"] = float(s.std()) if s.count() > 0 else None
        stats["median"] = float(s.median()) if s.count() > 0 else None
    return stats


# ---------------------------------------------------------------------------
# Memory / formatting helpers
# ---------------------------------------------------------------------------

def format_bytes(n_bytes: int) -> str:
    """Return a human-readable string for *n_bytes*."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n_bytes) < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024  # type: ignore[assignment]
    return f"{n_bytes:.1f} PB"


def dataframe_memory(df: pd.DataFrame) -> str:
    """Return the deep memory usage of a DataFrame as a readable string."""
    return format_bytes(df.memory_usage(deep=True).sum())


# ---------------------------------------------------------------------------
# Background thread runner
# ---------------------------------------------------------------------------

def run_in_background(
    func: Callable[..., Any],
    args: Tuple[Any, ...] = (),
    kwargs: Optional[Dict[str, Any]] = None,
    on_done: Optional[Callable[[Any], None]] = None,
    on_error: Optional[Callable[[Exception], None]] = None,
) -> threading.Thread:
    """Run *func* in a daemon thread, invoking callbacks when finished.

    Args:
        func: The callable to execute.
        args: Positional arguments for *func*.
        kwargs: Keyword arguments for *func*.
        on_done: Called with the result on success.
        on_error: Called with the exception on failure.

    Returns:
        The started Thread object.
    """
    kwargs = kwargs or {}

    def _target() -> None:
        try:
            result = func(*args, **kwargs)
            if on_done:
                on_done(result)
        except Exception as exc:
            traceback.print_exc()
            if on_error:
                on_error(exc)

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def save_json(data: Any, path: Union[str, Path]) -> None:
    """Save *data* as pretty-printed JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, cls=NumpyEncoder)


def load_json(path: Union[str, Path]) -> Any:
    """Load and return parsed JSON from *path*."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Timestamp helper
# ---------------------------------------------------------------------------

def timestamp_str() -> str:
    """Return an ISO-formatted timestamp string (no microseconds)."""
    return datetime.now().isoformat(timespec="seconds")
