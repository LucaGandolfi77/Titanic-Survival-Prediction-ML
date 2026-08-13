"""arff_parser.py — Lightweight ARFF file parser (no Weka / Java).

Handles both numeric/real and nominal attributes.
Returns numpy feature array, string labels, and attribute names.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np


class ArffData(NamedTuple):
    """Container returned by :func:`load_arff`."""
    X: np.ndarray          # (n_samples, n_features) float64
    y: np.ndarray          # (n_samples,) object (string labels)
    attr_names: list[str]  # feature attribute names (excluding the class)
    class_name: str        # name of the class attribute


def load_arff(path: str | Path) -> ArffData:
    """Parse an ARFF file and return features, labels, and metadata.

    Args:
        path: path to the ``.arff`` file.

    Returns:
        An :class:`ArffData` named-tuple.
    """
    path = Path(path)
    attr_names: list[str] = []
    attr_types: list[str] = []   # "numeric" or "nominal"
    data_rows: list[list[str]] = []
    in_data = False

    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            # Skip blanks and comments
            if not line or line.startswith("%"):
                continue

            lower = line.lower()

            if lower.startswith("@data"):
                in_data = True
                continue

            if lower.startswith("@attribute"):
                parts = line.split()
                name = parts[1]
                rest = " ".join(parts[2:]).strip()
                if rest.lower() in ("real", "numeric", "integer"):
                    atype = "numeric"
                elif rest.startswith("{"):
                    atype = "nominal"
                else:
                    atype = "numeric"
                attr_names.append(name)
                attr_types.append(atype)
                continue

            if in_data:
                # Handle both comma-separated and space-separated data
                row = [v.strip() for v in line.split(",")]
                if len(row) == 1:
                    row = line.split()
                data_rows.append(row)

    if not data_rows:
        raise ValueError(f"No data rows found in {path}")

    # The last attribute is the class label
    class_name = attr_names.pop()
    attr_types.pop()

    n_features = len(attr_names)
    n_samples = len(data_rows)

    X = np.zeros((n_samples, n_features), dtype=np.float64)
    y = np.empty(n_samples, dtype=object)

    for i, row in enumerate(data_rows):
        for j in range(n_features):
            X[i, j] = float(row[j])
        y[i] = row[n_features]

    return ArffData(X=X, y=y, attr_names=attr_names, class_name=class_name)
