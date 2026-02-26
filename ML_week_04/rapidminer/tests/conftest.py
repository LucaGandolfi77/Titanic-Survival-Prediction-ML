"""
conftest.py – Shared fixtures for RapidMiner‑Lite test suite.
"""
from __future__ import annotations

import os
import sys
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force‑import all operator modules so the registry is populated
import engine.operators_data        # noqa: F401
import engine.operators_transform   # noqa: F401
import engine.operators_feature     # noqa: F401
import engine.operators_model       # noqa: F401
import engine.operators_eval        # noqa: F401
import engine.operators_viz         # noqa: F401
import engine.process_runner        # noqa: F401


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def iris_df() -> pd.DataFrame:
    """Small synthetic Iris‑like DataFrame (30 rows, 3 classes)."""
    rng = np.random.RandomState(42)
    n = 30
    classes = ["setosa"] * 10 + ["versicolor"] * 10 + ["virginica"] * 10
    df = pd.DataFrame({
        "sepal_length": rng.uniform(4, 8, n).round(1),
        "sepal_width": rng.uniform(2, 5, n).round(1),
        "petal_length": rng.uniform(1, 7, n).round(1),
        "petal_width": rng.uniform(0.1, 2.5, n).round(1),
        "species": classes,
    })
    df.attrs["_roles"] = {"species": "label"}
    return df


@pytest.fixture()
def regression_df() -> pd.DataFrame:
    """Small regression DataFrame."""
    rng = np.random.RandomState(0)
    n = 40
    x = rng.uniform(0, 10, n)
    df = pd.DataFrame({
        "x1": x,
        "x2": rng.uniform(0, 5, n).round(2),
        "target": (3 * x + rng.normal(0, 1, n)).round(2),
    })
    df.attrs["_roles"] = {"target": "label"}
    return df


@pytest.fixture()
def numeric_df() -> pd.DataFrame:
    """Simple numeric DataFrame (no label)."""
    rng = np.random.RandomState(1)
    return pd.DataFrame({
        "a": rng.randn(50),
        "b": rng.randn(50),
        "c": rng.randn(50),
    })


@pytest.fixture()
def mixed_df() -> pd.DataFrame:
    """DataFrame with numeric + categorical columns and some NaN."""
    df = pd.DataFrame({
        "age": [25, 30, None, 45, 50, 22, None, 38, 60, 28],
        "salary": [30000, 50000, 45000, None, 80000, 28000, 55000, None, 90000, 35000],
        "city": ["NY", "LA", "NY", "SF", "LA", "NY", "SF", "LA", "SF", "NY"],
        "employed": [True, True, False, True, True, False, True, True, True, False],
    })
    return df


@pytest.fixture()
def tmp_dir(tmp_path):
    """Provide a temporary directory and clean up."""
    yield tmp_path
    # tmp_path auto-cleaned by pytest


@pytest.fixture()
def iris_csv(tmp_dir, iris_df) -> Path:
    """Write iris_df to a CSV and return the path."""
    p = tmp_dir / "iris_test.csv"
    iris_df.to_csv(p, index=False)
    return p


@pytest.fixture()
def sample_csv(tmp_dir, mixed_df) -> Path:
    """Write mixed_df to a CSV."""
    p = tmp_dir / "mixed.csv"
    mixed_df.to_csv(p, index=False)
    return p
