"""Non-parametric statistical tests for clustering comparison."""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def friedman_posthoc(
    df: pd.DataFrame,
    score_col: str,
    method_col: str = "method",
    block_col: str = "seed",
) -> Dict:
    methods = sorted(df[method_col].unique())
    blocks = sorted(df[block_col].unique())

    matrix = np.full((len(blocks), len(methods)), np.nan)
    for j, m in enumerate(methods):
        for i, b in enumerate(blocks):
            vals = df[(df[method_col] == m) & (df[block_col] == b)][score_col]
            if len(vals) > 0:
                matrix[i, j] = vals.mean()

    valid_rows = ~np.isnan(matrix).any(axis=1)
    matrix = matrix[valid_rows]

    if matrix.shape[0] < 3 or matrix.shape[1] < 2:
        return {
            "statistic": np.nan,
            "pvalue": np.nan,
            "reject": False,
            "posthoc": pd.DataFrame(),
        }

    stat, p = stats.friedmanchisquare(*matrix.T)

    n_methods = len(methods)
    n_blocks = matrix.shape[0]
    ranks = np.zeros_like(matrix)
    for i in range(n_blocks):
        ranks[i] = stats.rankdata(-matrix[i])
    mean_ranks = ranks.mean(axis=0)

    cd = np.sqrt((n_methods * (n_methods + 1)) / (6.0 * n_blocks))
    z_crit = stats.norm.ppf(1 - 0.05 / (n_methods * (n_methods - 1)))

    rows: List[Dict] = []
    for (i, m_a), (j, m_b) in combinations(enumerate(methods), 2):
        diff = abs(mean_ranks[i] - mean_ranks[j])
        sig = diff > z_crit * cd
        rows.append({
            "method_a": m_a,
            "method_b": m_b,
            "rank_a": mean_ranks[i],
            "rank_b": mean_ranks[j],
            "rank_diff": diff,
            "cd_threshold": z_crit * cd,
            "significant": sig,
        })

    return {
        "statistic": float(stat),
        "pvalue": float(p),
        "reject": bool(p < 0.05),
        "posthoc": pd.DataFrame(rows),
    }


def pairwise_wilcoxon(
    df: pd.DataFrame,
    score_col: str,
    method_col: str = "method",
    block_col: str = "seed",
    alpha: float = 0.05,
) -> pd.DataFrame:
    methods = sorted(df[method_col].unique())
    blocks = sorted(df[block_col].unique())

    scores: Dict[str, np.ndarray] = {}
    for m in methods:
        vals = []
        for b in blocks:
            v = df[(df[method_col] == m) & (df[block_col] == b)][score_col]
            vals.append(v.mean() if len(v) > 0 else np.nan)
        scores[m] = np.array(vals)

    rows: List[Dict] = []
    for m_a, m_b in combinations(methods, 2):
        a, b = scores[m_a], scores[m_b]
        mask = ~(np.isnan(a) | np.isnan(b))
        a_clean, b_clean = a[mask], b[mask]

        if len(a_clean) < 5:
            rows.append({
                "method_a": m_a,
                "method_b": m_b,
                "statistic": np.nan,
                "pvalue": np.nan,
                "significant": False,
                "effect_size": np.nan,
            })
            continue

        res = stats.wilcoxon(a_clean, b_clean, alternative="two-sided")
        d = cohens_d(a_clean, b_clean)
        n_comp = len(methods) * (len(methods) - 1) / 2
        adj_alpha = alpha / max(n_comp, 1)

        rows.append({
            "method_a": m_a,
            "method_b": m_b,
            "statistic": float(res.statistic),
            "pvalue": float(res.pvalue),
            "significant": bool(res.pvalue < adj_alpha),
            "effect_size": d,
        })

    return pd.DataFrame(rows)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na + nb < 3:
        return 0.0
    pooled_std = np.sqrt(
        ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
        / (na + nb - 2)
    )
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)
