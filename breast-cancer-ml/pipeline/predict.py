"""Prediction module — load a saved model and predict on new data.

Usage:
    python -m pipeline.predict --model random_forest \
        --input "[17.99,10.38,122.8,1001,...]"
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loader import load_config, load_data  # noqa: E402


def load_model(model_name: str, config: Optional[dict] = None) -> object:
    """Load a serialised model from the outputs directory.

    Args:
        model_name: Model slug, e.g. ``"random_forest"``.
        config: Configuration dictionary.

    Returns:
        Loaded sklearn-compatible estimator.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if config is None:
        config = load_config()

    models_dir = ROOT / config["paths"]["models"]
    pkl_path = models_dir / f"{model_name}.pkl"

    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {pkl_path}\n"
            f"Available: {[p.stem for p in models_dir.glob('*.pkl')]}"
        )

    return joblib.load(pkl_path)


def predict(
    model_name: str,
    input_data: np.ndarray | pd.DataFrame | list[float],
    config: Optional[dict] = None,
) -> dict:
    """Run inference on new sample(s).

    Args:
        model_name: Model slug, e.g. ``"random_forest"``.
        input_data: Feature vector(s).  A flat list is treated as a single
            sample with 30 features.
        config: Configuration dictionary.

    Returns:
        Dictionary with ``prediction``, ``probability``,
        ``class_label``, and ``model`` keys.
    """
    if config is None:
        config = load_config()

    estimator = load_model(model_name, config)

    # Normalise input shape
    if isinstance(input_data, list):
        input_data = np.array(input_data).reshape(1, -1)
    elif isinstance(input_data, pd.Series):
        input_data = input_data.values.reshape(1, -1)
    elif isinstance(input_data, pd.DataFrame):
        input_data = input_data.values
    elif input_data.ndim == 1:
        input_data = input_data.reshape(1, -1)

    # Scale the input using the same scaler
    bundle = load_data(config, scale=True, export_csv=False)
    input_scaled = bundle.scaler.transform(input_data)

    # Predict
    pred_class: np.ndarray = estimator.predict(input_scaled)
    pred_proba: np.ndarray = estimator.predict_proba(input_scaled)

    target_names = bundle.target_names

    results = []
    for i in range(len(pred_class)):
        label = target_names[int(pred_class[i])]
        results.append({
            "model": model_name,
            "prediction": int(pred_class[i]),
            "class_label": label,
            "probability_benign": float(pred_proba[i, 1]),
            "probability_malignant": float(pred_proba[i, 0]),
        })

    if len(results) == 1:
        return results[0]
    return {"predictions": results}


# ── CLI ──────────────────────────────────────────────────────────
def main() -> None:
    """Parse CLI arguments and run inference."""
    parser = argparse.ArgumentParser(
        description="Predict breast cancer class from feature vector."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="logistic_regression",
        help="Model slug (e.g. logistic_regression, random_forest, svm, xgboost)",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help='Feature vector as a Python list string, e.g. "[17.99, 10.38, ...]"',
    )
    args = parser.parse_args()

    try:
        features = ast.literal_eval(args.input)
    except (ValueError, SyntaxError) as exc:
        print(f"Error parsing input: {exc}")
        sys.exit(1)

    if not isinstance(features, list) or len(features) != 30:
        print(f"Expected a list of 30 floats, got {type(features).__name__} of length {len(features) if isinstance(features, list) else '?'}")
        sys.exit(1)

    result = predict(args.model, features)
    print(f"\n── Prediction ({''.join(args.model.split('_')).title()}) ──")
    print(f"  Class     : {result['class_label']} ({result['prediction']})")
    print(f"  P(benign) : {result['probability_benign']:.4f}")
    print(f"  P(malig.) : {result['probability_malignant']:.4f}")


if __name__ == "__main__":
    main()
