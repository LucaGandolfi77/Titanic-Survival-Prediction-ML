"""ant_train.py — Train supervised models on recorded ant game data.

Supports three model types:
  'dt'  — scikit-learn DecisionTreeClassifier
  'mlp' — scikit-learn MLPClassifier
  'gp'  — DEAP-based Genetic Programming (delegated to gp_player.py)
"""

from __future__ import annotations

import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


# -----------------------------------------------------------------------
# Data loading & augmentation
# -----------------------------------------------------------------------

def load_data(filename: str) -> pd.DataFrame:
    """Load a recording CSV into a DataFrame.

    The last column is the direction label; all preceding columns are
    neighbourhood features.

    Args:
        filename: path to the CSV file.

    Returns:
        DataFrame with numeric feature columns and a 'direction' column.
    """
    df = pd.read_csv(filename, header=None)
    n_cols = df.shape[1]
    feature_cols = [f"f{i}" for i in range(n_cols - 1)]
    df.columns = feature_cols + ["direction"]
    # Ensure features are numeric
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    return df


def augment_data(df: pd.DataFrame) -> pd.DataFrame:
    """Quadruple the dataset via 90°/180°/270° rotations.

    For each state-action pair the (2m+1)x(2m+1) neighbourhood grid is
    rotated and the direction label is rotated correspondingly.

    Args:
        df: original DataFrame from load_data().

    Returns:
        Augmented DataFrame (4× larger).
    """
    feature_cols = [c for c in df.columns if c != "direction"]
    side = int(np.sqrt(len(feature_cols)))
    if side * side != len(feature_cols):
        raise ValueError("Feature count is not a perfect square — cannot reshape to grid.")

    # Direction rotation mapping (90° clockwise each step)
    rot_dir = {"up": "right", "right": "down", "down": "left", "left": "up"}

    all_rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        grid = np.array([row[c] for c in feature_cols]).reshape(side, side)
        direction = row["direction"]

        for _ in range(4):
            flat = grid.flatten()
            entry = {f"f{i}": flat[i] for i in range(len(flat))}
            entry["direction"] = direction
            all_rows.append(entry)
            # Rotate 90° clockwise
            grid = np.rot90(grid, k=-1)
            direction = rot_dir[direction]

    return pd.DataFrame(all_rows)


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def ant_train(
    filename: str,
    model_type: str = "dt",
    augment: bool = True,
    verbose: bool = True,
    **gp_kwargs: Any,
) -> Any:
    """Train a model on recorded ant game data.

    Args:
        filename: path to the recording CSV.
        model_type: 'dt' (Decision Tree), 'mlp' (MLP), or 'gp' (Genetic Program).
        augment: if True, apply symmetry augmentation before training.
        verbose: print training info.
        **gp_kwargs: extra keyword arguments forwarded to gp_player.train_gp().

    Returns:
        The trained model object.
    """
    df = load_data(filename)
    if augment:
        df = augment_data(df)
        if verbose:
            print(f"  Augmented dataset: {len(df)} rows")

    feature_cols = [c for c in df.columns if c != "direction"]
    X = df[feature_cols].values.astype(float)
    y = df["direction"].values

    if model_type == "gp":
        from gp_player import train_gp
        model = train_gp(X, y, verbose=verbose, **gp_kwargs)
        return model

    # Encode labels for sklearn
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    if model_type == "dt":
        model = DecisionTreeClassifier(max_depth=10, random_state=42)
    elif model_type == "mlp":
        n_features = X.shape[1]
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=500,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'dt', 'mlp', or 'gp'.")

    model.fit(X, y_enc)

    # Attach label encoder so we can decode predictions later
    model._label_encoder = le  # type: ignore[attr-defined]

    if verbose:
        score = model.score(X, y_enc)
        print(f"  [{model_type.upper()}] Training accuracy: {score:.4f}  "
              f"({len(X)} samples, {X.shape[1]} features)")

    return model


def save_model(model: Any, path: str) -> None:
    """Pickle a trained model to disk.

    Args:
        model: any pickle-able model object.
        path: destination file path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  [save] {path}")


def load_model(path: str) -> Any:
    """Load a pickled model from disk.

    Args:
        path: file path to the pickle.

    Returns:
        The model object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


# -----------------------------------------------------------------------
# Standalone
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train an ant-trail model")
    parser.add_argument("--file", type=str, required=True, help="Training CSV")
    parser.add_argument("--model", type=str, default="dt", choices=["dt", "mlp", "gp"])
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    model = ant_train(args.file, model_type=args.model, augment=not args.no_augment)
    print(f"  Trained model: {type(model)}")
