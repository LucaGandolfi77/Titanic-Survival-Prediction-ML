"""
Diversity Metrics
==================
Pairwise and ensemble-level diversity measures for studying
homogeneous vs heterogeneous ensemble behaviour.

Implements: disagreement, Q-statistic, double-fault, kappa,
and the Krogh-Vedelsby ambiguity decomposition.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.base import ClassifierMixin


def _contingency(y1: np.ndarray, y2: np.ndarray, y_true: np.ndarray):
    """Compute contingency counts N11, N10, N01, N00."""
    c1 = (y1 == y_true).astype(int)
    c2 = (y2 == y_true).astype(int)
    n11 = int(np.sum(c1 * c2))
    n10 = int(np.sum(c1 * (1 - c2)))
    n01 = int(np.sum((1 - c1) * c2))
    n00 = int(np.sum((1 - c1) * (1 - c2)))
    return n11, n10, n01, n00


def disagreement(y1: np.ndarray, y2: np.ndarray, y_true: np.ndarray) -> float:
    """Fraction of samples on which two classifiers disagree."""
    n11, n10, n01, n00 = _contingency(y1, y2, y_true)
    total = n11 + n10 + n01 + n00
    if total == 0:
        return 0.0
    return (n01 + n10) / total


def q_statistic(y1: np.ndarray, y2: np.ndarray, y_true: np.ndarray) -> float:
    """Yule's Q-statistic. Positive → correlated, negative → diverse."""
    n11, n10, n01, n00 = _contingency(y1, y2, y_true)
    num = n11 * n00 - n01 * n10
    den = n11 * n00 + n01 * n10
    if den == 0:
        return 0.0
    return num / den


def double_fault(y1: np.ndarray, y2: np.ndarray, y_true: np.ndarray) -> float:
    """Fraction of samples misclassified by both classifiers."""
    n11, n10, n01, n00 = _contingency(y1, y2, y_true)
    total = n11 + n10 + n01 + n00
    if total == 0:
        return 0.0
    return n00 / total


def kappa_statistic(y1: np.ndarray, y2: np.ndarray, y_true: np.ndarray) -> float:
    """Cohen's Kappa between two classifiers (agreement measure)."""
    n11, n10, n01, n00 = _contingency(y1, y2, y_true)
    n = n11 + n10 + n01 + n00
    if n == 0:
        return 0.0
    p_agree = (n11 + n00) / n
    p1_correct = (n11 + n10) / n
    p2_correct = (n11 + n01) / n
    p_chance = p1_correct * p2_correct + (1 - p1_correct) * (1 - p2_correct)
    if p_chance == 1.0:
        return 1.0
    return (p_agree - p_chance) / (1 - p_chance)


# ── Ensemble-level (average pairwise) ────────────────────────────────


def ensemble_diversity(
    predictions: List[np.ndarray],
    y_true: np.ndarray,
    metric_fn=disagreement,
) -> float:
    """Average pairwise diversity over all pairs of base learners."""
    k = len(predictions)
    if k < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(k):
        for j in range(i + 1, k):
            total += metric_fn(predictions[i], predictions[j], y_true)
            count += 1
    return total / count


def compute_all_diversity(
    predictions: List[np.ndarray],
    y_true: np.ndarray,
) -> Dict[str, float]:
    """Compute all four diversity metrics at ensemble level."""
    return {
        "disagreement": ensemble_diversity(predictions, y_true, disagreement),
        "q_statistic": ensemble_diversity(predictions, y_true, q_statistic),
        "double_fault": ensemble_diversity(predictions, y_true, double_fault),
        "kappa": ensemble_diversity(predictions, y_true, kappa_statistic),
    }


# ── Ambiguity decomposition (Krogh-Vedelsby) ─────────────────────────


def ambiguity_decomposition(
    predictions: List[np.ndarray],
    y_true: np.ndarray,
) -> Dict[str, float]:
    """Krogh-Vedelsby decomposition: Ensemble_error = Avg_error - Ambiguity.

    Uses 0/1 loss formulation.  Returns dict with avg_individual_error,
    ensemble_error, and ambiguity.
    """
    k = len(predictions)
    if k == 0:
        return {"avg_individual_error": 0.0, "ensemble_error": 0.0, "ambiguity": 0.0}

    pred_matrix = np.array(predictions)  # (k, n)
    n = len(y_true)

    # Individual errors
    individual_errors = np.array([
        np.mean(pred_matrix[i] != y_true) for i in range(k)
    ])
    avg_individual_error = float(np.mean(individual_errors))

    # Ensemble prediction (majority vote)
    from scipy.stats import mode
    ensemble_pred = mode(pred_matrix, axis=0, keepdims=False).mode
    ensemble_error = float(np.mean(ensemble_pred != y_true))

    ambiguity = avg_individual_error - ensemble_error

    return {
        "avg_individual_error": avg_individual_error,
        "ensemble_error": ensemble_error,
        "ambiguity": ambiguity,
    }


def extract_base_predictions(
    ensemble: ClassifierMixin,
    X: np.ndarray,
) -> List[np.ndarray]:
    """Extract individual predictions from a fitted ensemble."""
    if hasattr(ensemble, "estimators_"):
        return [est.predict(X) for est in ensemble.estimators_]
    elif hasattr(ensemble, "named_estimators_"):
        return [est.predict(X) for est in ensemble.named_estimators_.values()]
    else:
        return [ensemble.predict(X)]


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=100)
    preds = [
        y_true.copy(),
        rng.integers(0, 2, size=100),
        rng.integers(0, 2, size=100),
    ]
    preds[0][:10] = 1 - preds[0][:10]

    metrics = compute_all_diversity(preds, y_true)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    amb = ambiguity_decomposition(preds, y_true)
    print(f"\nAmbiguity decomposition: {amb}")
