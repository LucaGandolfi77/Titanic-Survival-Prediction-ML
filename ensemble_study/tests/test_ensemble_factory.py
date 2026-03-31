"""Tests for ensembles.ensemble_factory."""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ensembles.ensemble_factory import build_method, method_label, METHOD_LABELS
from config import CFG
from sklearn.datasets import make_classification


class TestBuildMethod:
    @pytest.fixture(autouse=True)
    def _data(self):
        self.X, self.y = make_classification(
            n_samples=80, n_features=6, random_state=42
        )

    @pytest.mark.parametrize("name", list(CFG.METHOD_NAMES))
    def test_all_methods_build(self, name):
        clf = build_method(name, n_estimators=5, random_state=42)
        assert hasattr(clf, "fit")
        assert hasattr(clf, "predict")

    @pytest.mark.parametrize("name", list(CFG.METHOD_NAMES))
    def test_all_methods_fit_predict(self, name):
        clf = build_method(name, n_estimators=5, random_state=42)
        clf.fit(self.X, self.y)
        preds = clf.predict(self.X)
        assert len(preds) == len(self.y)
        assert set(preds).issubset(set(self.y))

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            build_method("nonexistent")

    def test_reproducibility(self):
        clf1 = build_method("random_forest", n_estimators=10, random_state=42)
        clf2 = build_method("random_forest", n_estimators=10, random_state=42)
        clf1.fit(self.X, self.y)
        clf2.fit(self.X, self.y)
        np.testing.assert_array_equal(clf1.predict(self.X), clf2.predict(self.X))


class TestMethodLabels:
    def test_all_methods_have_labels(self):
        for name in CFG.METHOD_NAMES:
            assert name in METHOD_LABELS

    def test_label_returns_string(self):
        for name in CFG.METHOD_NAMES:
            lbl = method_label(name)
            assert isinstance(lbl, str)
            assert len(lbl) > 0

    def test_unknown_returns_name(self):
        assert method_label("foo_bar") == "foo_bar"
