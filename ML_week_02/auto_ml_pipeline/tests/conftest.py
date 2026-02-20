"""
Shared fixtures for AutoML-Lite tests.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def binary_clf_df():
    """Small binary-classification DataFrame (Titanic-like)."""
    rng = np.random.RandomState(42)
    n = 200
    return pd.DataFrame({
        "Age": rng.randint(1, 80, n).astype(float),
        "Fare": rng.uniform(5, 500, n),
        "Pclass": rng.choice([1, 2, 3], n),
        "Sex": rng.choice(["male", "female"], n),
        "Embarked": rng.choice(["S", "C", "Q", np.nan], n),
        "SibSp": rng.randint(0, 5, n),
        "Survived": rng.choice([0, 1], n),
    })


@pytest.fixture
def regression_df():
    """Small regression DataFrame."""
    rng = np.random.RandomState(42)
    n = 200
    x1 = rng.uniform(0, 100, n)
    x2 = rng.uniform(0, 50, n)
    cat = rng.choice(["A", "B", "C"], n)
    noise = rng.normal(0, 5, n)
    y = 3 * x1 + 2 * x2 + noise
    return pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "cat": cat,
        "target": y,
    })


@pytest.fixture
def multiclass_df():
    """Small multiclass DataFrame (iris-like)."""
    rng = np.random.RandomState(42)
    n = 150
    return pd.DataFrame({
        "f1": rng.normal(5, 1, n),
        "f2": rng.normal(3, 0.5, n),
        "f3": rng.normal(4, 1.2, n),
        "f4": rng.normal(1, 0.3, n),
        "species": rng.choice(["setosa", "versicolor", "virginica"], n),
    })


@pytest.fixture
def config():
    """Minimal test config (AttrDict)."""
    from automl.utils.config import load_config
    return load_config(None)  # loads default.yaml
