"""Evaluation pipeline — detailed metrics, confusion matrices, ROC curves.

Usage:
    python -m pipeline.evaluate
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loader import load_config, load_data, DataBundle  # noqa: E402
from models.base_model import BaseModel  # noqa: E402
from models.logistic_regression import LogisticRegressionModel  # noqa: E402
from models.random_forest import RandomForestModel  # noqa: E402
from models.svm import SVMModel  # noqa: E402
from models.xgboost_model import XGBoostModel  # noqa: E402


DPI = 150
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)


# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────
def _load_trained_models(config: dict) -> List[BaseModel]:
    """Load pre-trained model wrappers.

    For each model wrapper class, load the saved ``.pkl`` estimator so
    that evaluation does not require re-training.

    Args:
        config: Parsed configuration.

    Returns:
        List of :class:`BaseModel` subclasses with loaded estimators.
    """
    models_dir = ROOT / config["paths"]["models"]
    model_cfgs = config.get("models", {})
    rs = config.get("random_state", 42)

    registry: List[tuple] = [
        ("logistic_regression", LogisticRegressionModel),
        ("random_forest", RandomForestModel),
        ("svm", SVMModel),
        ("xgboost", XGBoostModel),
    ]

    loaded: List[BaseModel] = []
    for slug, cls in registry:
        params = {**model_cfgs.get(slug, {}), "random_state": rs}
        wrapper = cls(params)
        pkl_path = models_dir / f"{wrapper.name.lower().replace(' ', '_')}.pkl"
        if pkl_path.exists():
            wrapper.estimator = BaseModel.load(pkl_path)
            loaded.append(wrapper)
            print(f"  ✓ Loaded {wrapper.name} from {pkl_path.name}")
        else:
            print(f"  ⚠ Model file not found: {pkl_path} — skipping")

    return loaded


# ─────────────────────────────────────────────────────────────────
#  Confusion matrix
# ─────────────────────────────────────────────────────────────────
def plot_confusion_matrix(
    model: BaseModel,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    target_names: list[str],
    save_dir: Path,
) -> None:
    """Plot and save a confusion-matrix heatmap for a single model.

    Args:
        model: Fitted model wrapper.
        X_test: Test features.
        y_test: True labels.
        target_names: Human-readable class labels.
        save_dir: Directory for the PNG.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model.name}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    slug = model.name.lower().replace(" ", "_")
    fig.savefig(save_dir / f"confusion_matrix_{slug}.png", dpi=DPI)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
#  ROC curves
# ─────────────────────────────────────────────────────────────────
def plot_roc_curves(
    models: List[BaseModel],
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    save_dir: Path,
) -> None:
    """Plot combined ROC curves for all models.

    Args:
        models: List of fitted model wrappers.
        X_test: Test features.
        y_test: True labels.
        save_dir: Directory for the PNG.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colours = sns.color_palette("husl", n_colors=len(models))

    for model, colour in zip(models, colours):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, color=colour, lw=2, label=f"{model.name} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    sns.despine()
    fig.tight_layout()
    fig.savefig(save_dir / "roc_curves.png", dpi=DPI)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
#  Full evaluation
# ─────────────────────────────────────────────────────────────────
def evaluate_all(config: dict | None = None) -> pd.DataFrame:
    """Run full evaluation on the test set and generate all artefacts.

    Args:
        config: Configuration dictionary.

    Returns:
        DataFrame with per-model metrics.
    """
    if config is None:
        config = load_config()

    bundle: DataBundle = load_data(config, scale=True, export_csv=False)
    plots_dir = ROOT / config["paths"]["plots"]
    reports_dir = ROOT / config["paths"]["reports"]
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    models = _load_trained_models(config)
    if not models:
        print("No trained models found.  Run `python -m pipeline.train` first.")
        return pd.DataFrame()

    target_names = bundle.target_names

    print("\n── Evaluation Pipeline ───────────────────────────")
    rows: list[dict] = []

    for model in models:
        print(f"\n▸ {model.name}")
        metrics = model.evaluate(bundle.X_test, bundle.y_test)
        for k, v in metrics.items():
            print(f"    {k:<12s}: {v:.4f}")

        # Classification report
        y_pred = model.predict(bundle.X_test)
        report = classification_report(bundle.y_test, y_pred, target_names=target_names)
        slug = model.name.lower().replace(" ", "_")
        report_path = reports_dir / f"classification_report_{slug}.txt"
        report_path.write_text(report, encoding="utf-8")

        # Confusion matrix
        plot_confusion_matrix(model, bundle.X_test, bundle.y_test, target_names, plots_dir)

        rows.append({"model": model.name, **metrics})

    # Combined ROC
    plot_roc_curves(models, bundle.X_test, bundle.y_test, plots_dir)

    results = pd.DataFrame(rows)
    results.to_csv(reports_dir / "evaluation_results.csv", index=False)

    print(f"\n  ✓ Plots  → {plots_dir}")
    print(f"  ✓ Reports → {reports_dir}")
    print("── Done ──────────────────────────────────────────\n")

    return results


# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    evaluate_all()
