"""
Statistical Tests
==================
Paired Wilcoxon signed-rank, Friedman + Nemenyi post-hoc,
Cohen's d effect size, and Spearman / Pearson correlations.
All functions return plain dicts / DataFrames for easy serialisation.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from config import CFG


# ── Effect size ───────────────────────────────────────────────────────


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for two independent samples (pooled SD)."""
    na, nb = len(a), len(b)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


# ── Paired Wilcoxon ──────────────────────────────────────────────────


def wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float | None = None,
) -> Dict:
    """Two-sided Wilcoxon signed-rank test on paired accuracy vectors."""
    alpha = alpha or CFG.SIGNIFICANCE_LEVEL
    diff = np.array(scores_a) - np.array(scores_b)
    if np.all(diff == 0):
        return {"statistic": 0.0, "p_value": 1.0, "significant": False, "cohens_d": 0.0}
    stat, p = stats.wilcoxon(scores_a, scores_b, alternative="two-sided")
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "significant": p < alpha,
        "cohens_d": cohens_d(np.asarray(scores_a), np.asarray(scores_b)),
    }


# ── Friedman + Nemenyi ───────────────────────────────────────────────


def friedman_test(
    score_matrix: np.ndarray,
    group_names: List[str] | None = None,
    alpha: float | None = None,
) -> Dict:
    """
    Friedman test on k related samples.

    Parameters
    ----------
    score_matrix : ndarray of shape (n_samples, k)
        Each column is a different treatment (e.g. pruning strategy).
    group_names  : list of str, optional
    alpha        : significance level

    Returns
    -------
    dict with statistic, p_value, significant, avg_ranks
    """
    alpha = alpha or CFG.SIGNIFICANCE_LEVEL
    k = score_matrix.shape[1]
    if group_names is None:
        group_names = [f"group_{i}" for i in range(k)]

    stat, p = stats.friedmanchisquare(*[score_matrix[:, i] for i in range(k)])

    # Average ranks  (rank 1 = best)
    ranks = np.zeros_like(score_matrix)
    for i in range(score_matrix.shape[0]):
        ranks[i] = stats.rankdata(-score_matrix[i])  # negative → higher is better → rank 1
    avg_ranks = {name: float(r) for name, r in zip(group_names, ranks.mean(axis=0))}

    return {
        "statistic": float(stat),
        "p_value": float(p),
        "significant": p < alpha,
        "avg_ranks": avg_ranks,
    }


def nemenyi_critical_difference(
    n_datasets: int,
    k_groups: int,
    alpha: float | None = None,
) -> float:
    """
    Approximate Nemenyi critical difference.

    CD = q_alpha * sqrt(k * (k+1) / (6 * N))

    q_alpha values for alpha=0.05 (studentised range / sqrt(2)):
    We use scipy to compute them from the studentised range distribution.
    """
    alpha = alpha or CFG.SIGNIFICANCE_LEVEL
    # Approximate q_{alpha,k} using tabulated values for common k
    # For a general approach we use the formula with the chi2 approximation
    from scipy.stats import chi2

    q_alpha_table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
    }
    q = q_alpha_table.get(k_groups, 2.728)  # fallback to k=5 value
    cd = q * np.sqrt(k_groups * (k_groups + 1) / (6 * n_datasets))
    return float(cd)


# ── Correlations ─────────────────────────────────────────────────────


def correlation_analysis(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> Dict:
    """Pearson and Spearman correlation between two columns."""
    x = df[x_col].values.astype(float)
    y = df[y_col].values.astype(float)
    r_pearson, p_pearson = stats.pearsonr(x, y)
    r_spearman, p_spearman = stats.spearmanr(x, y)
    return {
        "pearson_r": float(r_pearson),
        "pearson_p": float(p_pearson),
        "spearman_r": float(r_spearman),
        "spearman_p": float(p_spearman),
    }


# ── Pairwise comparison table ────────────────────────────────────────


def pairwise_wilcoxon(
    df: pd.DataFrame,
    score_col: str = "test_accuracy",
    strategy_col: str = "strategy",
    group_col: str = "seed",
) -> pd.DataFrame:
    """
    Produce a pairwise Wilcoxon comparison table for all strategies.

    The DataFrame must have one row per (strategy, seed) with the score.
    Returns a DataFrame with columns: strategy_a, strategy_b, stat, p, significant, cohens_d.
    """
    strategies = sorted(df[strategy_col].unique())
    results = []
    for i, sa in enumerate(strategies):
        for sb in strategies[i + 1:]:
            va = df.loc[df[strategy_col] == sa].sort_values(group_col)[score_col].values
            vb = df.loc[df[strategy_col] == sb].sort_values(group_col)[score_col].values
            n = min(len(va), len(vb))
            res = wilcoxon_test(va[:n], vb[:n])
            results.append({
                "strategy_a": sa,
                "strategy_b": sb,
                **res,
            })
    return pd.DataFrame(results)
