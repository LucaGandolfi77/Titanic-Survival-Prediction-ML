"""
test_explainability.py – Unit tests for SHAP, LIME, PDP, counterfactuals, interactions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def clf_data():
    X, y = make_classification(n_samples=200, n_features=6, random_state=42)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(6)])
    return df, pd.Series(y)


@pytest.fixture
def trained_model(clf_data):
    X, y = clf_data
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X, y)
    return model


# ── SHAP Explainer Tests ─────────────────────────────────────

class TestSHAPExplainer:
    def test_compute_shap_values(self, trained_model, clf_data):
        from src.explainability.shap_explainer import SHAPExplainer
        X, _ = clf_data
        exp = SHAPExplainer(trained_model, X.sample(50, random_state=42))
        shap_vals = exp.compute_shap_values(X.head(20))
        assert shap_vals.shape[0] == 20

    def test_global_importance(self, trained_model, clf_data):
        from src.explainability.shap_explainer import SHAPExplainer
        X, _ = clf_data
        exp = SHAPExplainer(trained_model, X.sample(50, random_state=42))
        imp = exp.global_importance(X.head(20))
        assert "feature" in imp.columns
        assert "mean_abs_shap" in imp.columns
        assert len(imp) == 6

    def test_explain_instance(self, trained_model, clf_data):
        from src.explainability.shap_explainer import SHAPExplainer
        X, _ = clf_data
        exp = SHAPExplainer(trained_model, X.sample(50, random_state=42))
        result = exp.explain_instance(X.iloc[0])
        assert "contributions" in result
        assert "base_value" in result


# ── LIME Explainer Tests ─────────────────────────────────────

class TestLIMEExplainer:
    def test_explain_instance(self, trained_model, clf_data):
        from src.explainability.lime_explainer import LIMEExplainer
        X, _ = clf_data
        lime_exp = LIMEExplainer(trained_model, X)
        result = lime_exp.explain_instance(X.iloc[0].values, n_features=5)
        assert result.feature_weights is not None
        assert len(result.feature_weights) == 5

    def test_lime_has_prediction(self, trained_model, clf_data):
        from src.explainability.lime_explainer import LIMEExplainer
        X, _ = clf_data
        lime_exp = LIMEExplainer(trained_model, X)
        result = lime_exp.explain_instance(X.iloc[0].values)
        assert result.prediction is not None


# ── PDP Tests ─────────────────────────────────────────────────

class TestPDP:
    def test_compute_pdp(self, trained_model, clf_data):
        from src.explainability.pdp import compute_pdp
        X, _ = clf_data
        result = compute_pdp(trained_model, X, "feat_0")
        assert len(result["average"]) == len(result["grid_values"])
        assert len(result["grid_values"]) > 0

    def test_compute_pdp_2d(self, trained_model, clf_data):
        from src.explainability.pdp import compute_pdp_2d
        X, _ = clf_data
        result = compute_pdp_2d(trained_model, X, "feat_0", "feat_1", grid_resolution=10)
        assert result["average"].shape == (10, 10)


# ── Counterfactuals Tests ─────────────────────────────────────

class TestCounterfactuals:
    def test_what_if(self, trained_model, clf_data):
        from src.explainability.counterfactuals import what_if_analysis
        X, _ = clf_data
        values = np.linspace(X["feat_0"].min(), X["feat_0"].max(), 10)
        result = what_if_analysis(trained_model, X.iloc[[0]], "feat_0", values)
        assert "feature_value" in result.columns
        assert len(result) == 10

    def test_find_counterfactuals(self, trained_model, clf_data):
        from src.explainability.counterfactuals import find_counterfactuals
        X, y = clf_data
        # Find an instance of class 0 and try to flip to class 1
        idx = (y == 0).values.nonzero()[0][0]
        instance = X.iloc[[idx]]
        cfs = find_counterfactuals(trained_model, instance, 1, X, n_counterfactuals=3)
        assert isinstance(cfs, pd.DataFrame)


# ── Feature Interactions Tests ────────────────────────────────

class TestFeatureInteractions:
    def test_top_interactions(self, trained_model, clf_data):
        from src.explainability.feature_interactions import top_interactions
        X, _ = clf_data
        result = top_interactions(trained_model, X.head(50), top_n=3, features=list(X.columns))
        assert len(result) <= 3
        if not result.empty:
            assert "feature_a" in result.columns
            assert "feature_b" in result.columns
            assert "h_statistic" in result.columns
