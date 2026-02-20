"""Tests for model modules."""

import numpy as np
import pandas as pd
import pytest

from automl.models.registry import get_model_entry, list_available_models
from automl.models.screener import ModelScreener
from automl.models.ensemble import EnsembleBuilder


class TestRegistry:

    def test_list_available(self):
        avail = list_available_models()
        assert "random_forest" in avail
        assert "logistic_regression" in avail

    def test_build_classifier(self):
        entry = get_model_entry("random_forest")
        model = entry.build("classification")
        assert hasattr(model, "fit")

    def test_build_regressor(self):
        entry = get_model_entry("random_forest")
        model = entry.build("regression")
        assert hasattr(model, "fit")

    def test_unknown_model(self):
        with pytest.raises(KeyError):
            get_model_entry("nonexistent_model")


class TestScreener:

    def test_screen_returns_top_k(self, config):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = rng.choice([0, 1], 100)
        screener = ModelScreener(config.screening)
        screener.candidates = ["logistic_regression", "random_forest"]
        screener.top_k = 2
        top = screener.screen(X, y, "classification")
        assert len(top) <= 2
        assert all(isinstance(n, str) for n in top)


class TestEnsemble:

    def test_voting_ensemble(self, config):
        rng = np.random.RandomState(42)
        X = rng.randn(80, 5)
        y = rng.choice([0, 1], 80)

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        models = {
            "rf": RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y),
            "lr": LogisticRegression(max_iter=1000).fit(X, y),
        }

        class _Cfg:
            method = "soft_voting"
            meta_learner = "logistic_regression"

        builder = EnsembleBuilder(_Cfg())
        ens = builder.build(models, X, y, "classification")
        preds = ens.predict(X)
        assert len(preds) == 80
