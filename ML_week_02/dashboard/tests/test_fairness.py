"""
test_fairness.py – Unit tests for fairness metrics, bias detection, and mitigation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fairness.metrics import (
    demographic_parity_difference,
    equal_opportunity_difference,
    disparate_impact_ratio,
    equalized_odds_difference,
    compute_all_metrics,
)
from src.fairness.bias_detector import detect_bias, bias_summary_text
from src.fairness.mitigation import (
    compute_sample_weights,
    find_equalised_thresholds,
    generate_recommendations,
)


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def fair_data():
    """Roughly fair predictions."""
    np.random.seed(42)
    n = 500
    y_true = np.random.randint(0, 2, n)
    y_pred = y_true.copy()
    # Small amount of noise
    flip = np.random.choice(n, 20, replace=False)
    y_pred[flip] = 1 - y_pred[flip]
    sensitive = np.random.choice(["A", "B"], n)
    return y_true, y_pred, sensitive


@pytest.fixture
def biased_data():
    """Intentionally biased predictions (group B rarely gets positive)."""
    np.random.seed(42)
    n = 500
    sensitive = np.array(["A"] * 250 + ["B"] * 250)
    y_true = np.random.randint(0, 2, n)
    y_pred = y_true.copy()
    # Make group B always predict 0
    y_pred[250:] = 0
    return y_true, y_pred, sensitive


# ── Metrics Tests ─────────────────────────────────────────────

class TestMetrics:
    def test_demographic_parity_fair(self, fair_data):
        y_true, y_pred, sensitive = fair_data
        dp = demographic_parity_difference(y_pred, sensitive)
        assert abs(dp) < 0.15  # should be small for fair data

    def test_demographic_parity_biased(self, biased_data):
        y_true, y_pred, sensitive = biased_data
        dp = demographic_parity_difference(y_pred, sensitive)
        assert abs(dp) > 0.3  # should be large for biased data

    def test_disparate_impact_fair(self, fair_data):
        y_true, y_pred, sensitive = fair_data
        di = disparate_impact_ratio(y_pred, sensitive)
        assert di > 0.7

    def test_disparate_impact_biased(self, biased_data):
        y_true, y_pred, sensitive = biased_data
        di = disparate_impact_ratio(y_pred, sensitive)
        assert di < 0.2  # group B gets no positives

    def test_equal_opportunity(self, fair_data):
        y_true, y_pred, sensitive = fair_data
        eo = equal_opportunity_difference(y_true, y_pred, sensitive)
        assert abs(eo) < 0.2

    def test_equalized_odds(self, fair_data):
        y_true, y_pred, sensitive = fair_data
        eod = equalized_odds_difference(y_true, y_pred, sensitive)
        assert abs(eod) < 0.2

    def test_compute_all_metrics(self, fair_data):
        y_true, y_pred, sensitive = fair_data
        metrics = compute_all_metrics(y_true, y_pred, sensitive)
        assert "demographic_parity_difference" in metrics
        assert "disparate_impact_ratio" in metrics
        assert "equal_opportunity_difference" in metrics


# ── Bias Detector Tests ───────────────────────────────────────

class TestBiasDetector:
    def test_detect_bias_returns_df(self, fair_data):
        y_true, y_pred, sensitive = fair_data
        result = detect_bias(y_true, y_pred, sensitive, "test_attr")
        assert isinstance(result, pd.DataFrame)
        assert "metric" in result.columns
        assert "status" in result.columns

    def test_bias_statuses(self, biased_data):
        y_true, y_pred, sensitive = biased_data
        result = detect_bias(y_true, y_pred, sensitive, "test_attr")
        # Should have at least one FAIL
        statuses = result["status"].tolist()
        assert any("❌" in s for s in statuses)

    def test_summary_text(self, fair_data):
        y_true, y_pred, sensitive = fair_data
        result = detect_bias(y_true, y_pred, sensitive, "attr")
        text = bias_summary_text(result)
        assert isinstance(text, str)
        assert len(text) > 10


# ── Mitigation Tests ──────────────────────────────────────────

class TestMitigation:
    def test_sample_weights(self, fair_data):
        y_true, _, sensitive = fair_data
        weights = compute_sample_weights(y_true, sensitive)
        assert len(weights) == len(y_true)
        assert all(w > 0 for w in weights)

    def test_equalised_thresholds(self, fair_data):
        y_true, y_pred, sensitive = fair_data
        # Use y_pred as "probabilities" (0/1)
        probs = y_pred.astype(float) + np.random.rand(len(y_pred)) * 0.1
        thresholds = find_equalised_thresholds(y_true, probs, sensitive)
        assert isinstance(thresholds, dict)
        assert all(0 < t < 1 for t in thresholds.values())

    def test_generate_recommendations_with_fails(self, biased_data):
        y_true, y_pred, sensitive = biased_data
        bias_df = detect_bias(y_true, y_pred, sensitive, "test")
        recs = generate_recommendations(bias_df)
        assert len(recs) > 0
        assert recs[0]["severity"] in ("CRITICAL", "HIGH", "MEDIUM", "LOW")

    def test_generate_recommendations_all_pass(self, fair_data):
        y_true, y_pred, sensitive = fair_data
        bias_df = detect_bias(y_true, y_pred, sensitive, "test")
        # If all pass, recommendations should still be generated
        recs = generate_recommendations(bias_df)
        assert len(recs) > 0
