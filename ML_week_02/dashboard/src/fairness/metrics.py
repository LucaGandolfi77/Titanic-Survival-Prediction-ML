"""
metrics.py – Fairness metric computation.

Computes group-level fairness metrics following the Fairlearn taxonomy:
demographic parity, equalised odds, disparate impact, etc.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ── Core metric functions ─────────────────────────────────────

def demographic_parity_difference(
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> float:
    """Max |P(ŷ=1|A=a) - P(ŷ=1|A=b)| across all subgroup pairs."""
    groups = np.unique(sensitive)
    rates = {g: y_pred[sensitive == g].mean() for g in groups}
    vals = list(rates.values())
    return round(float(max(vals) - min(vals)), 4)


def equal_opportunity_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> float:
    """Max difference in true-positive rate across groups."""
    groups = np.unique(sensitive)
    tpr = {}
    for g in groups:
        mask = (sensitive == g) & (y_true == 1)
        if mask.sum() == 0:
            tpr[g] = 0.0
        else:
            tpr[g] = y_pred[mask].mean()
    vals = list(tpr.values())
    return round(float(max(vals) - min(vals)), 4)


def disparate_impact_ratio(
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    reference_group: str | int | None = None,
) -> float:
    """P(ŷ=1|A=unprivileged) / P(ŷ=1|A=privileged).

    The 80 % rule: a ratio ≥ 0.8 is considered fair.
    """
    groups = np.unique(sensitive)
    rates = {g: y_pred[sensitive == g].mean() for g in groups}

    if reference_group is not None and reference_group in rates:
        ref_rate = rates[reference_group]
    else:
        ref_rate = max(rates.values())

    if ref_rate == 0:
        return 0.0
    min_rate = min(rates.values())
    return round(float(min_rate / ref_rate), 4)


def equalized_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> float:
    """Max difference in both TPR and FPR across groups."""
    groups = np.unique(sensitive)
    tpr, fpr = {}, {}

    for g in groups:
        pos = (sensitive == g) & (y_true == 1)
        neg = (sensitive == g) & (y_true == 0)
        tpr[g] = y_pred[pos].mean() if pos.sum() > 0 else 0.0
        fpr[g] = y_pred[neg].mean() if neg.sum() > 0 else 0.0

    tpr_diff = max(tpr.values()) - min(tpr.values())
    fpr_diff = max(fpr.values()) - min(fpr.values())
    return round(float(max(tpr_diff, fpr_diff)), 4)


# ── Subgroup performance ─────────────────────────────────────

def subgroup_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> pd.DataFrame:
    """Accuracy, precision-like proxy, selection rate per group."""
    from sklearn.metrics import accuracy_score, f1_score

    groups = np.unique(sensitive)
    records = []
    for g in groups:
        mask = sensitive == g
        yt, yp = y_true[mask], y_pred[mask]
        n = int(mask.sum())
        records.append({
            "group": g,
            "n_samples": n,
            "selection_rate": round(float(yp.mean()), 4),
            "accuracy": round(float(accuracy_score(yt, yp)), 4) if n > 0 else None,
            "f1": round(float(f1_score(yt, yp, zero_division=0)), 4) if n > 0 else None,
        })
    return pd.DataFrame(records)


# ── Dispatcher ────────────────────────────────────────────────

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    reference_group: str | int | None = None,
) -> Dict[str, float]:
    """Compute all fairness metrics at once."""
    return {
        "demographic_parity_difference": demographic_parity_difference(y_pred, sensitive),
        "equal_opportunity_difference": equal_opportunity_difference(y_true, y_pred, sensitive),
        "disparate_impact_ratio": disparate_impact_ratio(y_pred, sensitive, reference_group),
        "equalized_odds_difference": equalized_odds_difference(y_true, y_pred, sensitive),
    }
