"""
Statistical Tests
==================
Paired Wilcoxon signed-rank, Friedman + Nemenyi post-hoc,
Cohen's d, Wilcoxon effect size r, and Spearman correlations.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from config import CFG


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    var_a = np.var(a, ddof=1) if na > 1 else 0.0
    var_b = np.var(b, ddof=1) if nb > 1 else 0.0
    pooled = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / max(na + nb - 2, 1))
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def wilcoxon_test(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float | None = None,
) -> Dict:
    alpha = alpha or CFG.SIGNIFICANCE_LEVEL
    diff = np.asarray(a) - np.asarray(b)
    if np.all(diff == 0):
        return {"statistic": 0.0, "p_value": 1.0, "significant": False,
                "effect_size_r": 0.0, "cohens_d": 0.0}
    stat, p = stats.wilcoxon(a, b, alternative="two-sided")
    n = len(diff[diff != 0])
    z = stats.norm.ppf(1 - p / 2) if p > 0 else 0.0
    r = z / np.sqrt(n) if n > 0 else 0.0
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "significant": bool(p < alpha),
        "effect_size_r": float(r),
        "cohens_d": cohens_d(np.asarray(a), np.asarray(b)),
    }


def friedman_test(
    score_matrix: np.ndarray,
    group_names: List[str] | None = None,
    alpha: float | None = None,
) -> Dict:
    """Friedman test on k related samples (n_samples × k matrix)."""
    alpha = alpha or CFG.SIGNIFICANCE_LEVEL
    k = score_matrix.shape[1]
    group_names = group_names or [f"g{i}" for i in range(k)]

    stat, p = stats.friedmanchisquare(*[score_matrix[:, i] for i in range(k)])

    ranks = np.zeros_like(score_matrix)
    for i in range(score_matrix.shape[0]):
        ranks[i] = stats.rankdata(-score_matrix[i])
    avg_ranks = {name: float(r) for name, r in zip(group_names, ranks.mean(axis=0))}

    return {
        "statistic": float(stat),
        "p_value": float(p),
        "significant": bool(p < alpha),
        "avg_ranks": avg_ranks,
    }


def nemenyi_critical_difference(n_datasets: int, k_groups: int, alpha: float = 0.05) -> float:
    q_table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
               6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    q = q_table.get(k_groups, 2.728)
    return float(q * np.sqrt(k_groups * (k_groups + 1) / (6 * n_datasets)))


def correlation_analysis(df: pd.DataFrame, x_col: str, y_col: str) -> Dict:
    x = df[x_col].values.astype(float)
    y = df[y_col].values.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {"spearman_r": 0.0, "spearman_p": 1.0, "pearson_r": 0.0, "pearson_p": 1.0}
    r_s, p_s = stats.spearmanr(x, y)
    r_p, p_p = stats.pearsonr(x, y)
    return {
        "spearman_r": float(r_s), "spearman_p": float(p_s),
        "pearson_r": float(r_p), "pearson_p": float(p_p),
    }


def pairwise_wilcoxon(
    df: pd.DataFrame,
    score_col: str = "test_accuracy",
    method_col: str = "method",
    group_col: str = "seed",
) -> pd.DataFrame:
    methods = sorted(df[method_col].unique())
    results = []
    for i, ma in enumerate(methods):
        for mb in methods[i + 1:]:
            va = df.loc[df[method_col] == ma].sort_values(group_col)[score_col].values
            vb = df.loc[df[method_col] == mb].sort_values(group_col)[score_col].values
            n = min(len(va), len(vb))
            res = wilcoxon_test(va[:n], vb[:n])
            results.append({"method_a": ma, "method_b": mb, **res})
    return pd.DataFrame(results)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    a = rng.normal(0.85, 0.03, 15)
    b = rng.normal(0.80, 0.04, 15)
    print("Wilcoxon:", wilcoxon_test(a, b))
