"""Tests for ensembles.diversity_metrics."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ensembles.diversity_metrics import (
    disagreement,
    q_statistic,
    double_fault,
    kappa_statistic,
    ensemble_diversity,
    compute_all_diversity,
    ambiguity_decomposition,
    extract_base_predictions,
)


class TestPairwiseMetrics:
    def setup_method(self):
        self.y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        self.y1 =     np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])  # perfect
        self.y2 =     np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 0])  # all wrong

    def test_disagreement_identical(self):
        d = disagreement(self.y1, self.y1, self.y_true)
        assert d == 0.0

    def test_disagreement_opposite(self):
        d = disagreement(self.y1, self.y2, self.y_true)
        assert d == 1.0

    def test_disagreement_range(self):
        rng = np.random.default_rng(42)
        y3 = rng.choice([0, 1], size=10)
        d = disagreement(self.y1, y3, self.y_true)
        assert 0.0 <= d <= 1.0

    def test_q_statistic_identical(self):
        q = q_statistic(self.y1, self.y1, self.y_true)
        # Perfect agreement → Q=1 or Q=0 (depends on definition edge cases)
        assert q >= 0.0

    def test_double_fault_identical_correct(self):
        df = double_fault(self.y1, self.y1, self.y_true)
        assert df == 0.0  # neither wrong

    def test_double_fault_both_wrong(self):
        df = double_fault(self.y2, self.y2, self.y_true)
        assert df == 1.0  # all wrong

    def test_kappa_identical(self):
        k = kappa_statistic(self.y1, self.y1, self.y_true)
        assert k == pytest.approx(1.0, abs=0.01)


class TestEnsembleDiversity:
    def test_single_prediction(self):
        preds = [np.array([0, 1, 0, 1])]
        y = np.array([0, 1, 0, 1])
        d = ensemble_diversity(preds, y)
        assert d == 0.0

    def test_two_identical_predictions(self):
        p = np.array([0, 1, 0, 1])
        d = ensemble_diversity([p, p], np.array([0, 1, 0, 1]))
        assert d == 0.0

    def test_compute_all_returns_four_keys(self):
        preds = [np.array([0, 1, 0, 1]), np.array([1, 0, 0, 1])]
        y = np.array([0, 1, 0, 1])
        result = compute_all_diversity(preds, y)
        assert set(result.keys()) == {"disagreement", "q_statistic",
                                       "double_fault", "kappa"}


class TestAmbiguityDecomposition:
    def test_perfect_ensemble(self):
        preds = [np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1])]
        y = np.array([0, 1, 0, 1])
        result = ambiguity_decomposition(preds, y)
        assert result["ensemble_error"] == 0.0
        assert result["ambiguity"] == 0.0

    def test_returns_correct_keys(self):
        preds = [np.array([0, 1, 0, 1]), np.array([1, 0, 0, 1])]
        y = np.array([0, 1, 0, 1])
        result = ambiguity_decomposition(preds, y)
        assert "avg_individual_error" in result
        assert "ensemble_error" in result
        assert "ambiguity" in result


class TestExtractBasePredictions:
    def test_with_bagging(self):
        from sklearn.ensemble import BaggingClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        clf = BaggingClassifier(n_estimators=3, random_state=42)
        clf.fit(X, y)
        preds = extract_base_predictions(clf, X)
        assert len(preds) == 3
        assert all(len(p) == len(y) for p in preds)
