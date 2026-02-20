"""Integration test – run the full pipeline on synthetic data."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def titanic_csv(binary_clf_df, tmp_path):
    """Write binary_clf_df to a temp CSV and return its path."""
    path = tmp_path / "titanic_test.csv"
    binary_clf_df.to_csv(path, index=False)
    return str(path)


class TestFullPipeline:

    @pytest.mark.slow
    def test_binary_classification(self, titanic_csv, tmp_path):
        """End-to-end run on a tiny Titanic-like CSV."""
        from automl.pipeline import AutoMLPipeline

        pipe = AutoMLPipeline(output_dir=str(tmp_path / "out"))
        # Override HPO trials for speed
        pipe.cfg.hpo.n_trials = 3
        pipe.cfg.screening.candidates = ["logistic_regression", "random_forest"]
        pipe.cfg.screening.top_k = 2
        pipe.cfg.ensemble.enabled = False

        pipe.fit(csv_path=titanic_csv, target_column="Survived")

        assert pipe._fitted
        assert "metrics" in pipe.results_
        assert (tmp_path / "out" / "model.pkl").exists()
        assert (tmp_path / "out" / "report.html").exists()
        assert (tmp_path / "out" / "app.py").exists()

    @pytest.mark.slow
    def test_regression(self, regression_df, tmp_path):
        from automl.pipeline import AutoMLPipeline

        csv_path = tmp_path / "regression_test.csv"
        regression_df.to_csv(csv_path, index=False)

        pipe = AutoMLPipeline(output_dir=str(tmp_path / "out"))
        pipe.cfg.hpo.n_trials = 3
        pipe.cfg.screening.candidates = ["logistic_regression", "random_forest"]
        pipe.cfg.screening.top_k = 1
        pipe.cfg.ensemble.enabled = False

        pipe.fit(csv_path=str(csv_path), target_column="target")

        assert pipe._fitted
        assert "r2" in pipe.results_.get("metrics", {}) or "rmse" in pipe.results_.get("metrics", {})
