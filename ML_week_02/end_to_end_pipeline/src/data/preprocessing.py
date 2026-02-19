"""
preprocessing.py — Feature Engineering Pipeline
================================================
Builds a reproducible sklearn ColumnTransformer that handles:
  • Missing-value imputation
  • Feature engineering (FamilySize, IsAlone, Title extraction)
  • Scaling for numeric features
  • One-hot encoding for categorical features

The pipeline is serialised to disk so that the exact same
transformations are applied at inference time.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger("titanic_mlops.preprocessing")


# ── Feature engineering helpers ──────────────────────────────


def extract_title(name: str) -> str:
    """Extract the social title from a passenger name string."""
    title = name.split(",")[1].split(".")[0].strip()
    # Group rare titles
    rare = {
        "Lady", "Countess", "Capt", "Col", "Don", "Dr",
        "Major", "Rev", "Sir", "Jonkheer", "Dona",
    }
    if title in rare:
        return "Rare"
    if title == "Mlle":
        return "Miss"
    if title in ("Ms", "Mme"):
        return "Mrs"
    return title


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features **before** the sklearn pipeline runs.

    New columns
    -----------
    FamilySize : SibSp + Parch + 1
    IsAlone    : 1 if FamilySize == 1 else 0
    Title      : extracted from Name
    """
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    if "Name" in df.columns:
        df["Title"] = df["Name"].apply(extract_title)
    return df


# ── Sklearn preprocessing pipeline ─────────────────────────


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that processes numeric and categorical
    features in parallel.

    Parameters
    ----------
    numeric_features : list[str]
        Column names to treat as numeric.
    categorical_features : list[str]
        Column names to one-hot encode.

    Returns
    -------
    ColumnTransformer
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


# ── Public API ──────────────────────────────────────────────


def load_and_split(
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load the raw CSV, engineer features, and split into train/val.

    Parameters
    ----------
    cfg : dict
        Project configuration dictionary.

    Returns
    -------
    X_train, X_val, y_train, y_val
    """
    prep_cfg = cfg["preprocessing"]
    data_path: Path = cfg["paths"]["data_dir"] / "train.csv"

    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)

    # Feature engineering
    df = engineer_features(df)

    # Separate target
    target = prep_cfg["target_column"]
    y = df[target]
    X = df.drop(columns=[target] + prep_cfg["drop_columns"], errors="ignore")

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=prep_cfg["test_size"],
        random_state=cfg["project"]["random_seed"],
        stratify=y,
    )
    logger.info(
        "Train/val split: %d / %d rows", len(X_train), len(X_val)
    )
    return X_train, X_val, y_train, y_val


def fit_and_save_preprocessor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, ColumnTransformer]:
    """
    Fit the preprocessor on training data, transform, and persist.

    Parameters
    ----------
    X_train : DataFrame
        Raw training features.
    y_train : Series
        Training target (passed through unchanged).
    cfg : dict
        Project configuration.

    Returns
    -------
    X_train_processed : ndarray
        Transformed training matrix.
    preprocessor : ColumnTransformer
        Fitted transformer (also saved to disk).
    """
    prep_cfg = cfg["preprocessing"]
    processed_dir: Path = cfg["paths"]["processed_dir"]

    numeric_features = prep_cfg["numeric_features"] + ["FamilySize", "IsAlone"]
    categorical_features = prep_cfg["categorical_features"] + ["Title"]

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_processed = preprocessor.fit_transform(X_train)

    # Persist artefacts
    _save_pickle(X_train_processed, processed_dir / "X_train.pkl")
    _save_pickle(y_train.values, processed_dir / "y_train.pkl")
    _save_pickle(preprocessor, processed_dir / "preprocessor.pkl")

    logger.info(
        "Preprocessor fitted — output shape %s — saved to %s",
        X_train_processed.shape, processed_dir,
    )
    return X_train_processed, preprocessor


def transform_validation(
    X_val: pd.DataFrame,
    preprocessor: ColumnTransformer,
    cfg: Dict[str, Any],
) -> np.ndarray:
    """
    Transform validation data using the already-fitted preprocessor.

    Parameters
    ----------
    X_val : DataFrame
        Raw validation features.
    preprocessor : ColumnTransformer
        Fitted transformer.
    cfg : dict
        Project configuration.

    Returns
    -------
    X_val_processed : ndarray
    """
    processed_dir: Path = cfg["paths"]["processed_dir"]
    X_val_processed = preprocessor.transform(X_val)
    _save_pickle(X_val_processed, processed_dir / "X_test.pkl")
    logger.info("Validation data transformed — shape %s", X_val_processed.shape)
    return X_val_processed


# ── Helpers ─────────────────────────────────────────────────


def _save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)
    logger.debug("Saved %s", path)


def load_pickle(path: Path) -> Any:
    """Load a pickled object from *path*."""
    with open(path, "rb") as fh:
        return pickle.load(fh)
