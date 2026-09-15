"""Statistical utilities for the Lab 1 report.

Everything here exists to answer one question honestly: *are the accuracy
differences we measured real, or are they sampling noise?*  With a 100-point
circle test set a single instance is 1 pp, so most of those gaps are noise; with
6010 digit test instances the same gap is meaningful.  The tests below make that
distinction quantitative instead of rhetorical.

Implemented without statsmodels: McNemar's exact test is a binomial test on the
discordant pairs (``scipy.stats.binomtest``) and the Wilson interval is closed
form.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest, norm


# --------------------------------------------------------------------------
# Confidence intervals
# --------------------------------------------------------------------------


def wilson_interval(n_correct: int, n_total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because it stays inside [0, 1] and
    behaves sensibly for the small samples in the circle experiments.
    """
    if n_total == 0:
        return (float("nan"), float("nan"))
    z = norm.ppf(1 - (1 - confidence) / 2)
    p = n_correct / n_total
    denom = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denom
    half = (z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2))) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def accuracy_with_ci(y_true: np.ndarray, y_pred: np.ndarray,
                     confidence: float = 0.95) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    correct = int(np.sum(y_true == y_pred))
    n = int(len(y_true))
    lo, hi = wilson_interval(correct, n, confidence)
    return {
        "n": n,
        "n_correct": correct,
        "accuracy": correct / n if n else float("nan"),
        "ci_low": lo,
        "ci_high": hi,
        "ci_width_pp": 100 * (hi - lo),
    }


# --------------------------------------------------------------------------
# Paired significance testing
# --------------------------------------------------------------------------


@dataclass
class McNemarResult:
    """Outcome of McNemar's test on two classifiers evaluated on the same data."""

    n: int
    #: A correct, B wrong
    b: int
    #: A wrong, B correct
    c: int
    statistic: float
    p_value: float
    method: str
    acc_a: float
    acc_b: float
    delta_pp: float

    @property
    def significant_05(self) -> bool:
        return self.p_value < 0.05

    def summary(self) -> str:
        sig = "SIGNIFICANT" if self.significant_05 else "not significant"
        return (f"A={self.acc_a:.5f} B={self.acc_b:.5f} delta={self.delta_pp:+.2f}pp  "
                f"discordant b={self.b} c={self.c}  p={self.p_value:.3g}  "
                f"({self.method}, {sig} at 0.05)")


def mcnemar_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    exact_threshold: int = 25,
) -> McNemarResult:
    """McNemar's test comparing two classifiers on identical test instances.

    Tests whether the two classifiers have the same error rate, using only the
    instances where they disagree (``b`` and ``c`` in the 2x2 table).  With few
    discordant pairs the chi-square approximation is poor, so an exact binomial
    test is used whenever ``b + c <= exact_threshold``; otherwise the
    continuity-corrected chi-square statistic is reported and its p-value is
    obtained from the chi-square distribution (1 d.o.f.).

    ``b`` = A correct and B wrong, ``c`` = A wrong and B correct.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    if not (len(y_true) == len(pred_a) == len(pred_b)):
        raise ValueError("all inputs must have the same length")

    ok_a = pred_a == y_true
    ok_b = pred_b == y_true
    b = int(np.sum(ok_a & ~ok_b))
    c = int(np.sum(~ok_a & ok_b))
    n_discordant = b + c

    if n_discordant == 0:
        return McNemarResult(
            n=len(y_true), b=0, c=0, statistic=0.0, p_value=1.0,
            method="exact binomial (no discordant pairs)",
            acc_a=float(ok_a.mean()), acc_b=float(ok_b.mean()),
            delta_pp=100 * (ok_a.mean() - ok_b.mean()),
        )

    if n_discordant <= exact_threshold:
        res = binomtest(b, n_discordant, 0.5, alternative="two-sided")
        return McNemarResult(
            n=len(y_true), b=b, c=c, statistic=float(min(b, c)), p_value=float(res.pvalue),
            method="exact binomial",
            acc_a=float(ok_a.mean()), acc_b=float(ok_b.mean()),
            delta_pp=100 * (ok_a.mean() - ok_b.mean()),
        )

    # Continuity-corrected chi-square, 1 d.o.f.
    stat = (abs(b - c) - 1.0) ** 2 / (b + c)
    from scipy.stats import chi2

    p = float(chi2.sf(stat, df=1))
    return McNemarResult(
        n=len(y_true), b=b, c=c, statistic=float(stat), p_value=p,
        method="chi-square with continuity correction",
        acc_a=float(ok_a.mean()), acc_b=float(ok_b.mean()),
        delta_pp=100 * (ok_a.mean() - ok_b.mean()),
    )


def holm_correction(p_values: Sequence[float]) -> list:
    """Holm-Bonferroni step-down adjustment for multiple comparisons.

    Doing many pairwise McNemar tests inflates the family-wise error rate, so the
    p-values are adjusted before any of them is called significant.  Returns the
    adjusted p-values in the original order.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        val = min(1.0, p[idx] * factor)
        # enforce monotonicity of the step-down procedure
        running_max = max(running_max, val)
        adjusted[idx] = running_max
    return [float(v) for v in adjusted]


def mcnemar_matrix(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    adjust: bool = True,
) -> list:
    """All pairwise McNemar comparisons between named prediction vectors.

    Returns a list of dicts ready to be written to CSV, with
    ``p_adjusted`` holding the Holm-corrected value across the whole family of
    comparisons.
    """
    names = list(predictions.keys())
    rows = []
    raw_p = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            res = mcnemar_test(y_true, predictions[a], predictions[b])
            rows.append({
                "model_a": a,
                "model_b": b,
                "acc_a": res.acc_a,
                "acc_b": res.acc_b,
                "delta_pp": res.delta_pp,
                "n_test": res.n,
                "discordant_a_correct": res.b,
                "discordant_b_correct": res.c,
                "statistic": res.statistic,
                "method": res.method,
                "p_value": res.p_value,
                "significant_05": res.significant_05,
            })
            raw_p.append(res.p_value)
    if adjust and rows:
        adjusted = holm_correction(raw_p)
        for row, padj in zip(rows, adjusted):
            row["p_adjusted_holm"] = padj
            row["significant_05_holm"] = padj < 0.05
    return rows
