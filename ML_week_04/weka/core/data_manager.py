"""
data_manager.py – Dataset loading, inspection, and state management.
Supports CSV, TSV, Excel, and ARFF files.
"""

from __future__ import annotations

import pathlib
import re
from typing import Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────
# ARFF mini-parser (covers most Weka files)
# ──────────────────────────────────────────────────────────────────────
def _parse_arff(path: str | pathlib.Path) -> pd.DataFrame:
    """Read an ARFF file into a DataFrame."""
    attrs: list[tuple[str, str]] = []
    data_lines: list[str] = []
    in_data = False
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            low = line.lower()
            if low.startswith("@data"):
                in_data = True
                continue
            if in_data:
                data_lines.append(line)
            elif low.startswith("@attribute"):
                parts = line.split(None, 2)
                name = parts[1].strip("'\"")
                atype = parts[2] if len(parts) > 2 else "string"
                attrs.append((name, atype))

    cols = [a[0] for a in attrs]
    rows: list[list] = []
    for dl in data_lines:
        vals = [v.strip().strip("'\"") for v in dl.split(",")]
        rows.append(vals)

    df = pd.DataFrame(rows, columns=cols)
    # Try to convert numeric columns
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass
    df.replace("?", np.nan, inplace=True)
    return df


# ──────────────────────────────────────────────────────────────────────
# DataManager – single source of truth for the active dataset
# ──────────────────────────────────────────────────────────────────────
class DataManager:
    """Manages the currently loaded dataset and keeps an undo history."""

    def __init__(self) -> None:
        self.df: Optional[pd.DataFrame] = None
        self.filepath: Optional[str] = None
        self.filename: str = ""
        self._history: list[pd.DataFrame] = []
        self._listeners: list = []

    # ── I/O ───────────────────────────────────────────────────────────
    def load(self, path: str) -> pd.DataFrame:
        p = pathlib.Path(path)
        ext = p.suffix.lower()
        if ext == ".arff":
            df = _parse_arff(p)
        elif ext in (".xls", ".xlsx"):
            df = pd.read_excel(p)
        elif ext == ".tsv":
            df = pd.read_csv(p, sep="\t")
        else:
            # Try common CSV encodings
            for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
                try:
                    df = pd.read_csv(p, encoding=enc)
                    break
                except (UnicodeDecodeError, Exception):
                    continue
            else:
                raise ValueError(f"Cannot read {p.name}")

        self.df = df
        self.filepath = str(p)
        self.filename = p.name
        self._history = [df.copy()]
        self._notify()
        return df

    def save(self, path: str) -> None:
        if self.df is None:
            return
        p = pathlib.Path(path)
        ext = p.suffix.lower()
        if ext in (".xls", ".xlsx"):
            self.df.to_excel(p, index=False)
        elif ext == ".tsv":
            self.df.to_csv(p, sep="\t", index=False)
        else:
            self.df.to_csv(p, index=False)

    # ── Undo ──────────────────────────────────────────────────────────
    def checkpoint(self) -> None:
        if self.df is not None:
            self._history.append(self.df.copy())

    def undo(self) -> bool:
        if len(self._history) > 1:
            self._history.pop()
            self.df = self._history[-1].copy()
            self._notify()
            return True
        return False

    # ── Introspection helpers ─────────────────────────────────────────
    def summary(self) -> dict:
        if self.df is None:
            return {}
        df = self.df
        return {
            "filename": self.filename,
            "rows": len(df),
            "cols": len(df.columns),
            "numeric": list(df.select_dtypes("number").columns),
            "categorical": list(df.select_dtypes(exclude="number").columns),
            "missing": int(df.isnull().sum().sum()),
            "mem_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        }

    def column_stats(self, col: str) -> dict:
        if self.df is None or col not in self.df.columns:
            return {}
        s = self.df[col]
        info: dict = {
            "name": col,
            "dtype": str(s.dtype),
            "count": int(s.count()),
            "missing": int(s.isnull().sum()),
            "unique": int(s.nunique()),
        }
        if pd.api.types.is_numeric_dtype(s):
            info.update({
                "mean": round(float(s.mean()), 4),
                "std": round(float(s.std()), 4),
                "min": float(s.min()),
                "25%": float(s.quantile(0.25)),
                "50%": float(s.quantile(0.50)),
                "75%": float(s.quantile(0.75)),
                "max": float(s.max()),
            })
        else:
            vc = s.value_counts()
            info["top_values"] = vc.head(10).to_dict()
        return info

    # ── Observer pattern (UI refresh) ─────────────────────────────────
    def add_listener(self, cb) -> None:
        self._listeners.append(cb)

    def _notify(self) -> None:
        for cb in self._listeners:
            try:
                cb()
            except Exception:
                pass

    def set_df(self, df: pd.DataFrame) -> None:
        """Replace current dataframe and notify listeners."""
        self.checkpoint()
        self.df = df
        self._notify()
