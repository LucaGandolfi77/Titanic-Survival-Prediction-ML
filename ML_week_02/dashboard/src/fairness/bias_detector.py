"""
bias_detector.py – Automatic bias detection across protected attributes.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .metrics import compute_all_metrics, subgroup_metrics


def detect_bias(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    sensitive: np.ndarray | pd.Series,
    attribute_name: str = "protected",
    thresholds: Dict[str, float] | None = None,
) -> pd.DataFrame:
    """Scan one protected attribute for bias using pre-computed predictions.

    Returns a summary DataFrame with one row per metric,
    flagged with a traffic-light status: ✅ PASS, ⚠️ WARNING, ❌ FAIL.
    """
    thresholds = thresholds or {
        "demographic_parity_difference": 0.1,
        "equal_opportunity_difference": 0.1,
        "disparate_impact_ratio": 0.8,     # minimum acceptable
        "equalized_odds_difference": 0.1,
    }

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)

    metrics = compute_all_metrics(y_true, y_pred, sensitive)

    records = []
    for metric_name, value in metrics.items():
        threshold = thresholds.get(metric_name)
        if threshold is None:
            status = "⚠️ NO THRESHOLD"
        elif metric_name == "disparate_impact_ratio":
            # Higher is better (≥ 0.8)
            if value >= threshold:
                status = "✅ PASS"
            elif value >= threshold * 0.9:
                status = "⚠️ WARNING"
            else:
                status = "❌ FAIL"
        else:
            # Lower is better (< threshold)
            if abs(value) <= threshold:
                status = "✅ PASS"
            elif abs(value) <= threshold * 1.5:
                status = "⚠️ WARNING"
            else:
                status = "❌ FAIL"

        records.append({
            "protected_attribute": attribute_name,
            "metric": metric_name,
            "value": value,
            "threshold": threshold,
            "status": status,
        })

    return pd.DataFrame(records)


def subgroup_analysis(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    sensitive: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Detailed per-group performance for a single protected attribute."""
    return subgroup_metrics(np.asarray(y_true), np.asarray(y_pred), np.asarray(sensitive))


def bias_summary_text(bias_df: pd.DataFrame) -> str:
    """Generate a plain-English bias summary from the detection DataFrame."""
    n_fail = (bias_df["status"] == "❌ FAIL").sum()
    n_warn = (bias_df["status"] == "⚠️ WARNING").sum()
    n_pass = (bias_df["status"] == "✅ PASS").sum()

    lines = [f"Bias Scan Results: {n_pass} passed, {n_warn} warnings, {n_fail} failures.\n"]

    if n_fail > 0:
        lines.append("⚠️ CRITICAL FINDINGS:")
        for _, row in bias_df[bias_df["status"] == "❌ FAIL"].iterrows():
            lines.append(
                f"  • {row['metric']} on '{row['protected_attribute']}' "
                f"= {row['value']:.4f} (threshold: {row['threshold']})"
            )

    if n_warn > 0:
        lines.append("\n⚠️ WARNINGS:")
        for _, row in bias_df[bias_df["status"] == "⚠️ WARNING"].iterrows():
            lines.append(
                f"  • {row['metric']} on '{row['protected_attribute']}' "
                f"= {row['value']:.4f} (threshold: {row['threshold']})"
            )

    if n_fail == 0 and n_warn == 0:
        lines.append("✅ No significant bias detected across all tested attributes.")

    return "\n".join(lines)
