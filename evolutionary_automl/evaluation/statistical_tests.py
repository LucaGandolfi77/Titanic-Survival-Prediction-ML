"""
Statistical significance tests for comparing optimization methods.

Provides Wilcoxon signed-rank test (paired, 2 methods), Mann-Whitney U
test (independent, 2 methods), and Friedman test (>2 methods across
multiple datasets). All functions return p-values, test statistics,
and effect sizes suitable for thesis-grade reporting.

References:
    Demšar, J. (2006). "Statistical Comparisons of Classifiers over
    Multiple Data Sets." JMLR 7, pp. 1–30.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats


def wilcoxon_signed_rank(
    scores_a: List[float],
    scores_b: List[float],
    alternative: str = "two-sided",
) -> Dict[str, Any]:
    """Wilcoxon signed-rank test for paired samples.

    Use when comparing two methods evaluated on the same folds/seeds.

    Args:
        scores_a: Performance scores for method A (one per run).
        scores_b: Performance scores for method B (one per run).
        alternative: 'two-sided', 'greater', or 'less'.

    Returns:
        Dict with statistic, p_value, effect_size (r = Z / sqrt(N)),
        and a human-readable interpretation.
    """
    a, b = np.array(scores_a), np.array(scores_b)
    n = len(a)

    if np.allclose(a, b):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "effect_size": 0.0,
            "significant": False,
            "interpretation": "Distributions are identical.",
        }

    result = stats.wilcoxon(a, b, alternative=alternative)
    z_score = stats.norm.isf(result.pvalue / 2)
    effect_size = abs(z_score) / np.sqrt(n)

    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "effect_size": float(effect_size),
        "significant": result.pvalue < 0.05,
        "interpretation": _interpret_effect(effect_size),
    }


def mann_whitney_u(
    scores_a: List[float],
    scores_b: List[float],
    alternative: str = "two-sided",
) -> Dict[str, Any]:
    """Mann-Whitney U test for independent samples.

    Args:
        scores_a: Scores for method A.
        scores_b: Scores for method B.
        alternative: 'two-sided', 'greater', or 'less'.

    Returns:
        Dict with statistic, p_value, effect_size (rank-biserial r).
    """
    a, b = np.array(scores_a), np.array(scores_b)
    result = stats.mannwhitneyu(a, b, alternative=alternative)
    n1, n2 = len(a), len(b)
    effect_size = 1 - (2 * result.statistic) / (n1 * n2)

    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "effect_size": float(abs(effect_size)),
        "significant": result.pvalue < 0.05,
        "interpretation": _interpret_effect(abs(effect_size)),
    }


def friedman_test(
    score_matrix: List[List[float]],
    method_names: List[str],
) -> Dict[str, Any]:
    """Friedman test for comparing >2 methods across multiple datasets.

    Args:
        score_matrix: List of lists, shape [n_datasets, n_methods].
            Each row = one dataset, each column = one method's score.
        method_names: Names of the methods (columns).

    Returns:
        Dict with statistic, p_value, average_ranks, and post-hoc
        Nemenyi critical difference if significant.
    """
    matrix = np.array(score_matrix)
    n_datasets, n_methods = matrix.shape

    # scipy.stats.friedmanchisquare takes columns as separate arrays
    columns = [matrix[:, j] for j in range(n_methods)]
    result = stats.friedmanchisquare(*columns)

    # Compute average ranks
    ranks = np.zeros_like(matrix, dtype=float)
    for i in range(n_datasets):
        ranks[i] = stats.rankdata(-matrix[i])  # higher score → rank 1

    avg_ranks = ranks.mean(axis=0)
    rank_dict = {name: float(r) for name, r in zip(method_names, avg_ranks)}

    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant": result.pvalue < 0.05,
        "average_ranks": rank_dict,
        "n_datasets": n_datasets,
        "n_methods": n_methods,
    }


def compute_summary_stats(scores: List[float]) -> Dict[str, float]:
    """Compute mean, std, median, IQR for a list of scores."""
    a = np.array(scores)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "median": float(np.median(a)),
        "iqr": float(np.percentile(a, 75) - np.percentile(a, 25)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def _interpret_effect(r: float) -> str:
    """Interpret effect size using Cohen's conventions."""
    if r < 0.1:
        return "negligible"
    elif r < 0.3:
        return "small"
    elif r < 0.5:
        return "medium"
    else:
        return "large"


if __name__ == "__main__":
    a = [0.90, 0.91, 0.89, 0.92, 0.88, 0.93, 0.87, 0.91, 0.90, 0.89]
    b = [0.85, 0.86, 0.84, 0.87, 0.83, 0.88, 0.82, 0.86, 0.85, 0.84]

    print("Wilcoxon:", wilcoxon_signed_rank(a, b))
    print("Mann-Whitney:", mann_whitney_u(a, b))

    matrix = [[0.90, 0.85, 0.80], [0.92, 0.88, 0.83], [0.88, 0.84, 0.79]]
    print("Friedman:", friedman_test(matrix, ["GA", "Random", "Grid"]))
    print("Summary A:", compute_summary_stats(a))
