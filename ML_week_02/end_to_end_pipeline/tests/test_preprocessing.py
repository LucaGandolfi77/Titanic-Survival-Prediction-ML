"""
test_preprocessing.py — Unit Tests for Feature Engineering
==========================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import (
    build_preprocessor,
    engineer_features,
    extract_title,
)


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal DataFrame mimicking raw Titanic data."""
    return pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4],
            "Survived": [0, 1, 1, 0],
            "Pclass": [3, 1, 3, 1],
            "Name": [
                "Braund, Mr. Owen Harris",
                "Cumings, Mrs. John Bradley",
                "Heikkinen, Miss. Laina",
                "Futrelle, Mrs. Jacques Heath",
            ],
            "Sex": ["male", "female", "female", "female"],
            "Age": [22.0, 38.0, 26.0, 35.0],
            "SibSp": [1, 1, 0, 1],
            "Parch": [0, 0, 0, 0],
            "Ticket": ["A/5 21171", "PC 17599", "STON/O2. 3101282", "113803"],
            "Fare": [7.25, 71.28, 7.92, 53.10],
            "Cabin": [None, "C85", None, "C123"],
            "Embarked": ["S", "C", "S", "S"],
        }
    )


# ── Title extraction ────────────────────────────────────────

class TestExtractTitle:
    def test_mr(self):
        assert extract_title("Braund, Mr. Owen Harris") == "Mr"

    def test_mrs(self):
        assert extract_title("Cumings, Mrs. John Bradley") == "Mrs"

    def test_miss(self):
        assert extract_title("Heikkinen, Miss. Laina") == "Miss"

    def test_rare_title(self):
        assert extract_title("Rothes, Countess. of") == "Rare"

    def test_mlle_maps_to_miss(self):
        assert extract_title("Someone, Mlle. Foo") == "Miss"

    def test_ms_maps_to_mrs(self):
        assert extract_title("Someone, Ms. Bar") == "Mrs"


# ── Feature engineering ─────────────────────────────────────

class TestEngineerFeatures:
    def test_family_size(self, sample_df: pd.DataFrame):
        result = engineer_features(sample_df)
        expected = sample_df["SibSp"] + sample_df["Parch"] + 1
        pd.testing.assert_series_equal(
            result["FamilySize"], expected, check_names=False
        )

    def test_is_alone(self, sample_df: pd.DataFrame):
        result = engineer_features(sample_df)
        assert result.loc[2, "IsAlone"] == 1  # SibSp=0, Parch=0
        assert result.loc[0, "IsAlone"] == 0  # SibSp=1

    def test_title_column_created(self, sample_df: pd.DataFrame):
        result = engineer_features(sample_df)
        assert "Title" in result.columns
        assert list(result["Title"]) == ["Mr", "Mrs", "Miss", "Mrs"]

    def test_original_not_mutated(self, sample_df: pd.DataFrame):
        original_cols = set(sample_df.columns)
        engineer_features(sample_df)
        assert set(sample_df.columns) == original_cols


# ── Preprocessor pipeline ───────────────────────────────────

class TestBuildPreprocessor:
    def test_output_shape(self, sample_df: pd.DataFrame):
        df = engineer_features(sample_df)
        numeric = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "IsAlone"]
        categorical = ["Pclass", "Sex", "Embarked", "Title"]

        preprocessor = build_preprocessor(numeric, categorical)
        X = preprocessor.fit_transform(df)

        assert X.shape[0] == 4
        # At least numeric_count + one-hot columns
        assert X.shape[1] > len(numeric)

    def test_no_nans_after_transform(self, sample_df: pd.DataFrame):
        df = engineer_features(sample_df)
        numeric = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "IsAlone"]
        categorical = ["Pclass", "Sex", "Embarked", "Title"]

        preprocessor = build_preprocessor(numeric, categorical)
        X = preprocessor.fit_transform(df)
        assert not np.isnan(X).any()

    def test_handles_missing_values(self):
        df = pd.DataFrame(
            {
                "Age": [22.0, None, None],
                "Fare": [7.25, None, 53.0],
                "SibSp": [1, 0, 1],
                "Parch": [0, 0, 0],
                "FamilySize": [2, 1, 2],
                "IsAlone": [0, 1, 0],
                "Pclass": [3, 1, 2],
                "Sex": ["male", "female", None],
                "Embarked": ["S", None, "C"],
                "Title": ["Mr", "Mrs", "Miss"],
            }
        )
        numeric = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "IsAlone"]
        categorical = ["Pclass", "Sex", "Embarked", "Title"]

        preprocessor = build_preprocessor(numeric, categorical)
        X = preprocessor.fit_transform(df)
        assert not np.isnan(X).any()
