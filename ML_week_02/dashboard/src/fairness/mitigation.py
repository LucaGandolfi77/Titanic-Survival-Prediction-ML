"""
mitigation.py – Bias mitigation strategies and recommendations.

Provides both actionable code-level mitigations (re-weighting, threshold
optimisation) and business-level recommendations.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


# ── Pre-processing: re-weighting ──────────────────────────────

def compute_sample_weights(
    y: np.ndarray,
    sensitive: np.ndarray,
) -> np.ndarray:
    """Compute sample weights to equalise selection rates across groups.

    Uses the Kamiran & Calders (2012) re-weighting method.
    """
    n = len(y)
    groups = np.unique(sensitive)
    classes = np.unique(y)
    weights = np.ones(n, dtype=float)

    for g in groups:
        for c in classes:
            mask = (sensitive == g) & (y == c)
            expected = (sensitive == g).sum() * (y == c).sum() / n
            actual = mask.sum()
            if actual > 0:
                weights[mask] = expected / actual

    return weights


# ── Post-processing: threshold optimisation ───────────────────

def find_equalised_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sensitive: np.ndarray,
    metric: str = "demographic_parity",
    n_thresholds: int = 100,
) -> Dict[str, float]:
    """Find per-group classification thresholds that minimise unfairness.

    Returns dict {group: optimal_threshold}.
    """
    groups = np.unique(sensitive)
    thresholds: Dict[str, float] = {}

    for g in groups:
        mask = sensitive == g
        probs_g = y_prob[mask]
        y_g = y_true[mask]

        best_t = 0.5
        best_score = float("inf")

        for t in np.linspace(0.01, 0.99, n_thresholds):
            preds_g = (probs_g >= t).astype(int)
            rate = preds_g.mean()
            global_rate = (y_prob >= 0.5).astype(int).mean()

            if metric == "demographic_parity":
                score = abs(rate - global_rate)
            else:
                # Equalised odds: minimize TPR difference
                pos = y_g == 1
                tpr = preds_g[pos].mean() if pos.sum() > 0 else 0
                global_tpr = ((y_prob >= 0.5).astype(int)[y_true == 1]).mean()
                score = abs(tpr - global_tpr)

            if score < best_score:
                best_score = score
                best_t = t

        thresholds[str(g)] = round(best_t, 3)

    return thresholds


# ── Recommendations ───────────────────────────────────────────

def generate_recommendations(bias_df: pd.DataFrame) -> List[Dict[str, str]]:
    """Based on bias scan results, generate actionable recommendations."""
    recs: List[Dict[str, str]] = []

    fails = bias_df[bias_df["status"] == "❌ FAIL"]
    warns = bias_df[bias_df["status"] == "⚠️ WARNING"]

    # Demographic parity issues
    dp_issues = fails[fails["metric"] == "demographic_parity_difference"]
    if not dp_issues.empty:
        recs.append({
            "severity": "HIGH",
            "category": "Selection Rate Disparity",
            "finding": (
                f"Significant differences in positive prediction rates across "
                f"groups in: {', '.join(dp_issues['protected_attribute'].unique())}."
            ),
            "recommendation": (
                "1. Apply re-weighting during training to balance group representation.\n"
                "2. Use post-processing threshold adjustment per group.\n"
                "3. Consider removing or transforming the correlated proxy features."
            ),
            "regulation": "May violate Equal Credit Opportunity Act / EU AI Act Article 10.",
        })

    # Disparate impact issues
    di_issues = fails[fails["metric"] == "disparate_impact_ratio"]
    if not di_issues.empty:
        for _, row in di_issues.iterrows():
            recs.append({
                "severity": "CRITICAL",
                "category": "Disparate Impact (80% Rule)",
                "finding": (
                    f"Disparate impact ratio for '{row['protected_attribute']}' "
                    f"is {row['value']:.2f} (< 0.80 threshold)."
                ),
                "recommendation": (
                    "1. Immediately review feature set for proxy discrimination.\n"
                    "2. Apply adversarial de-biasing or re-weighting.\n"
                    "3. Document justification if disparity is business-necessary."
                ),
                "regulation": "Fails the EEOC 80% rule. Review under GDPR Art. 22.",
            })

    # Equal opportunity
    eo_issues = fails[fails["metric"] == "equal_opportunity_difference"]
    if not eo_issues.empty:
        recs.append({
            "severity": "HIGH",
            "category": "Equal Opportunity Violation",
            "finding": "True positive rates differ significantly across groups.",
            "recommendation": (
                "1. Apply equalised-odds post-processing.\n"
                "2. Audit training data for label bias.\n"
                "3. Consider collecting more balanced training samples."
            ),
            "regulation": "Relevant to Fair Housing Act and EU AI Act.",
        })

    # Warnings
    if not warns.empty and not recs:
        recs.append({
            "severity": "MEDIUM",
            "category": "Near-threshold Warnings",
            "finding": f"{len(warns)} metrics are approaching unfairness thresholds.",
            "recommendation": (
                "1. Monitor these metrics closely in production.\n"
                "2. Set up automated fairness alerts.\n"
                "3. Re-evaluate at next model retraining cycle."
            ),
            "regulation": "Proactive compliance with EU AI Act risk management.",
        })

    # All pass
    if not recs:
        recs.append({
            "severity": "LOW",
            "category": "No Issues Detected",
            "finding": "All fairness metrics are within acceptable thresholds.",
            "recommendation": (
                "Continue monitoring. Re-run analysis when:\n"
                "• Model is retrained\n"
                "• Data distribution shifts\n"
                "• Regulatory requirements change"
            ),
            "regulation": "Compliant with current standards.",
        })

    return recs
