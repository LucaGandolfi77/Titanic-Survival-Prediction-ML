"""Tests for pipeline builder."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_iris

from evolutionary_automl.search_space.pipeline_builder import build_pipeline, describe_pipeline


class TestBuildPipeline:
    @pytest.fixture
    def iris_data(self):
        return load_iris(return_X_y=True)

    def test_pipeline_fits_and_predicts(self, iris_data):
        X, y = iris_data
        chrom = [0.33, 0.0, 0.5, 0.0, 0.17, 0.31, 0.27, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5]
        pipe = build_pipeline(chrom, n_features=X.shape[1])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == len(y)

    def test_no_scaler_no_feature_sel(self, iris_data):
        X, y = iris_data
        chrom = [0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5]
        pipe = build_pipeline(chrom, n_features=X.shape[1])
        assert pipe.steps[0][0] == "classifier"

    def test_all_steps_present(self, iris_data):
        X, y = iris_data
        # scaler=standard, feat_sel=selectkbest, dim_red=pca, clf=rf
        chrom = [0.33, 0.33, 0.3, 0.5, 0.17, 0.31, 0.27, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5]
        pipe = build_pipeline(chrom, n_features=X.shape[1])
        step_names = [s[0] for s in pipe.steps]
        assert "scaler" in step_names
        assert "feature_selection" in step_names
        assert "classifier" in step_names

    def test_all_classifiers_fit(self, iris_data):
        X, y = iris_data
        for clf_val in np.linspace(0.0, 1.0, 7):
            chrom = [0.33, 0.0, 0.5, 0.0, clf_val] + [0.5] * 8
            pipe = build_pipeline(chrom, n_features=X.shape[1])
            pipe.fit(X, y)
            score = pipe.score(X, y)
            assert score > 0.0


class TestDescribePipeline:
    def test_describe_returns_string(self):
        chrom = [0.33, 0.33, 0.5, 0.0, 0.17, 0.5, 0.27, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5]
        desc = describe_pipeline(chrom, 30)
        assert isinstance(desc, str)
        assert "→" in desc
