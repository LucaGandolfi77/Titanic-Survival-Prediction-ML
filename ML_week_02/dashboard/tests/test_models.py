"""
test_models.py – Unit tests for model loading, prediction, and metadata extraction.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.models.loader import save_model, load_model, save_metadata, load_metadata, list_available_models
from src.models.metadata import extract_model_info, compute_performance
from src.models.predictor import Predictor


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]), pd.Series(y)


@pytest.fixture
def trained_rf(sample_data):
    X, y = sample_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def trained_lr(sample_data):
    X, y = sample_data
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def tmp_models_dir(monkeypatch, tmp_path):
    """Redirect MODELS_DIR to a temp directory."""
    import src.models.loader as loader_mod
    monkeypatch.setattr(loader_mod, "MODELS_DIR", tmp_path)
    return tmp_path


# ── Model Loader Tests ────────────────────────────────────────

class TestModelLoader:
    def test_save_and_load(self, trained_rf, tmp_models_dir):
        save_model(trained_rf, "test_rf", models_dir=tmp_models_dir)
        loaded = load_model("test_rf", models_dir=tmp_models_dir)
        assert type(loaded).__name__ == "RandomForestClassifier"

    def test_save_and_load_metadata(self, tmp_models_dir):
        meta = {"accuracy": 0.95, "features": ["a", "b"]}
        save_metadata("test_model", meta, models_dir=tmp_models_dir)
        loaded = load_metadata("test_model", models_dir=tmp_models_dir)
        assert loaded["accuracy"] == 0.95

    def test_list_models(self, trained_rf, tmp_models_dir):
        save_model(trained_rf, "m1", models_dir=tmp_models_dir)
        save_model(trained_rf, "m2", models_dir=tmp_models_dir)
        models = list_available_models(models_dir=tmp_models_dir)
        assert "m1" in models
        assert "m2" in models


# ── Metadata Tests ────────────────────────────────────────────

class TestMetadata:
    def test_extract_model_info_rf(self, trained_rf):
        info = extract_model_info(trained_rf)
        assert info["class"] == "RandomForestClassifier"
        assert info["n_features"] == 5

    def test_extract_model_info_lr(self, trained_lr):
        info = extract_model_info(trained_lr)
        assert "LogisticRegression" in info["class"]

    def test_compute_performance_clf(self, trained_rf, sample_data):
        X, y = sample_data
        perf = compute_performance(trained_rf, X, y)
        assert "accuracy" in perf
        assert "f1" in perf
        assert 0 <= perf["accuracy"] <= 1

    def test_performance_has_roc_auc(self, trained_rf, sample_data):
        X, y = sample_data
        perf = compute_performance(trained_rf, X, y)
        assert "roc_auc" in perf


# ── Predictor Tests ───────────────────────────────────────────

class TestPredictor:
    def test_predict(self, trained_rf, sample_data):
        X, y = sample_data
        predictor = Predictor(trained_rf)
        result = predictor.predict(X)
        assert len(result.labels) == len(X)

    def test_predict_single(self, trained_rf, sample_data):
        X, y = sample_data
        predictor = Predictor(trained_rf)
        result = predictor.predict_single(X.iloc[0])
        assert result.labels is not None

    def test_is_classifier(self, trained_rf, trained_lr):
        assert Predictor(trained_rf).is_classifier
        assert Predictor(trained_lr).is_classifier
