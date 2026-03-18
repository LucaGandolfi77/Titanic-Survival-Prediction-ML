"""data_loader.py — ARFF parser and synthetic dataset generator.

Provides:
- load_arff(path) → pandas DataFrame with named columns + 'label'
- generate_circle_data() → creates circletrain/circletest/circleall .arff files
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# ARFF parser
# ------------------------------------------------------------------

def load_arff(filepath: str | Path) -> pd.DataFrame:
    """Parse an ARFF file and return a pandas DataFrame.

    - Numeric / real attributes become float columns.
    - Nominal attributes become string columns.
    - The last attribute is renamed to ``label``.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"ARFF file not found: {filepath}")

    attr_names: list[str] = []
    attr_types: list[str] = []          # 'numeric' or list of nominals
    in_data = False
    rows: list[list[str]] = []

    with open(filepath, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue

            lower = line.lower()
            if lower.startswith("@relation"):
                continue
            elif lower.startswith("@attribute"):
                parts = re.split(r"\s+", line, maxsplit=2)
                name = parts[1]
                type_str = parts[2].strip()
                attr_names.append(name)
                if type_str.lower() in ("real", "numeric", "integer"):
                    attr_types.append("numeric")
                else:
                    # Nominal: e.g. {q,c}
                    attr_types.append("nominal")
            elif lower.startswith("@data"):
                in_data = True
                continue

            if in_data and line:
                fields = [v.strip() for v in line.split(",")]
                # Trim to expected number of attributes (some ARFF files
                # have trailing commas or extra fields on a few rows).
                if len(fields) > len(attr_names):
                    fields = fields[: len(attr_names)]
                rows.append(fields)

    df = pd.DataFrame(rows, columns=attr_names)

    # Cast numeric columns to float
    for name, atype in zip(attr_names, attr_types):
        if atype == "numeric":
            df[name] = pd.to_numeric(df[name], errors="coerce")

    # Rename last column → label
    if attr_names:
        df = df.rename(columns={attr_names[-1]: "label"})

    return df


def arff_features_and_label(df: pd.DataFrame):
    """Split a loaded ARFF DataFrame into X (numpy) and y (numpy str)."""
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(float)
    y = df["label"].values.astype(str)
    return X, y


# ------------------------------------------------------------------
# Synthetic circle/square data generator
# ------------------------------------------------------------------

def _label_circle(x: float, y: float) -> str:
    """Return 'c' (inside unit circle) or 'q' (outside, in square)."""
    return "c" if x * x + y * y <= 1.0 else "q"


def write_arff(df: pd.DataFrame, relation_name: str, filepath: str | Path) -> None:
    """Write a DataFrame to a valid ARFF file.

    Conventions:
    - float columns → @attribute … real
    - string / object columns → @attribute … {val1,val2,…}
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"@relation {relation_name}\n")
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                f.write(f"@attribute {col} real\n")
            else:
                vals = sorted(df[col].unique())
                f.write(f"@attribute {col} {{{','.join(vals)}}}\n")
        f.write("\n@data\n")
        for _, row in df.iterrows():
            vals = []
            for col in df.columns:
                v = row[col]
                if pd.api.types.is_numeric_dtype(df[col]):
                    vals.append(f"{v:.7f}")
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")


def generate_circle_data(data_dir: str | Path = "data",
                         seed: int = 42) -> None:
    """Generate circletrain, circletest, circleall .arff files if missing."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    for name in ("circletrain", "circletest"):
        path = data_dir / f"{name}.arff"
        if path.exists():
            continue
        xs = rng.uniform(-1.5, 1.5, 100)
        ys = rng.uniform(-1.5, 1.5, 100)
        labels = [_label_circle(x, y) for x, y in zip(xs, ys)]
        df = pd.DataFrame({"x": xs, "y": ys, "label": labels})
        write_arff(df, "circle", path)
        print(f"  [gen] Created {path}")

    # Dense grid
    path_all = data_dir / "circleall.arff"
    if not path_all.exists():
        grid = np.arange(-1.5, 1.501, 0.05)
        xx, yy = np.meshgrid(grid, grid)
        xs = xx.ravel()
        ys = yy.ravel()
        labels = [_label_circle(x, y) for x, y in zip(xs, ys)]
        df = pd.DataFrame({"x": xs, "y": ys, "label": labels})
        write_arff(df, "circle", path_all)
        print(f"  [gen] Created {path_all}")
