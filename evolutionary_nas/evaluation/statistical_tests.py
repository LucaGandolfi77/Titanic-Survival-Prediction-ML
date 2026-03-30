"""
Statistical Tests
=================
Wilcoxon signed-rank, Friedman test, effect size (Cohen's d),
and summary statistics for multi-seed experiment comparison.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


def compute_summary_stats(values: List[float]) -> Dict[str, float]:
    """Mean, std, median, IQR, min, max for a list of values."""
    a = np.array(values)
    q1, q3 = float(np.percentile(a, 25)), float(np.percentile(a, 75))
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "median": float(np.median(a)),
        "iqr": q3 - q1,
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "n": len(a),
    }


def wilcoxon_signed_rank(
    x: List[float], y: List[float]
) -> Dict[str, float]:
    """Wilcoxon signed-rank test for paired samples."""
    x_arr, y_arr = np.array(x), np.array(y)
    diff = x_arr - y_arr
    if np.all(diff == 0):
        return {"statistic": 0.0, "p_value": 1.0}
    stat, p = stats.wilcoxon(diff)
    return {"statistic": float(stat), "p_value": float(p)}


def friedman_test(
    *groups: List[float],
) -> Dict[str, float]:
    """Friedman test for comparing >2 algorithms across datasets.

    Each group is a list of scores (one per dataset or seed).
    """
    if len(groups) < 3:
        return {"statistic": 0.0, "p_value": 1.0}
    stat, p = stats.friedmanchisquare(*groups)
    return {"statistic": float(stat), "p_value": float(p)}


def cohens_d(x: List[float], y: List[float]) -> float:
    """Cohen's d effect size between two samples."""
    x_arr, y_arr = np.array(x), np.array(y)
    nx, ny = len(x_arr), len(y_arr)
    pooled_std = np.sqrt(
        ((nx - 1) * np.var(x_arr, ddof=1) + (ny - 1) * np.var(y_arr, ddof=1))
        / (nx + ny - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float((np.mean(x_arr) - np.mean(y_arr)) / pooled_std)


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    if d_abs < 0.5:
        return "small"
    if d_abs < 0.8:
        return "medium"
    return "large"


if __name__ == "__main__":
    a = [0.85, 0.87, 0.86, 0.88, 0.84, 0.89, 0.87, 0.86, 0.85, 0.88]
    b = [0.82, 0.83, 0.81, 0.84, 0.80, 0.83, 0.82, 0.81, 0.82, 0.84]
    print(f"Summary A: {compute_summary_stats(a)}")
    w = wilcoxon_signed_rank(a, b)
    print(f"Wilcoxon: W={w['statistic']:.1f}, p={w['p_value']:.4f}")
    d = cohens_d(a, b)
    print(f"Cohen's d: {d:.3f} ({interpret_effect_size(d)})")
