"""Tests for algorithm factory — build + fit_predict all 6 methods."""

from __future__ import annotations

import unittest

import numpy as np
from sklearn.datasets import make_blobs

from algorithms.algorithm_factory import (
    build_algorithm,
    fit_predict_algorithm,
    method_label,
    METHOD_LABELS,
)
from config import CFG


class TestBuildAlgorithm(unittest.TestCase):
    """Verify that every registered algorithm can be instantiated."""

    def test_build_all_methods(self):
        for name in CFG.METHOD_NAMES:
            with self.subTest(method=name):
                alg = build_algorithm(name, n_clusters=3, random_state=42)
                self.assertIsNotNone(alg)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            build_algorithm("nonexistent_method")


class TestFitPredictAlgorithm(unittest.TestCase):
    """Verify that every method can fit and predict on simple data."""

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = make_blobs(
            n_samples=200, centers=3, cluster_std=0.5, random_state=42,
        )

    def test_fit_predict_all_methods(self):
        for name in CFG.METHOD_NAMES:
            with self.subTest(method=name):
                labels, alg = fit_predict_algorithm(
                    name, self.X, n_clusters=3, random_state=42,
                )
                self.assertEqual(labels.shape, (len(self.X),))
                self.assertGreaterEqual(len(np.unique(labels)), 1)

    def test_labels_are_integers(self):
        for name in CFG.METHOD_NAMES:
            with self.subTest(method=name):
                labels, _ = fit_predict_algorithm(
                    name, self.X, n_clusters=3, random_state=42,
                )
                self.assertTrue(np.issubdtype(labels.dtype, np.integer))

    def test_deterministic_all_methods(self):
        for name in CFG.METHOD_NAMES:
            with self.subTest(method=name):
                l1, _ = fit_predict_algorithm(
                    name, self.X, n_clusters=3, random_state=99,
                )
                l2, _ = fit_predict_algorithm(
                    name, self.X, n_clusters=3, random_state=99,
                )
                np.testing.assert_array_equal(l1, l2)


class TestMethodLabels(unittest.TestCase):
    """Verify label helpers."""

    def test_all_methods_have_labels(self):
        for name in CFG.METHOD_NAMES:
            label = method_label(name)
            self.assertIsInstance(label, str)
            self.assertGreater(len(label), 0)

    def test_method_labels_dict_size(self):
        self.assertEqual(len(METHOD_LABELS), len(CFG.METHOD_NAMES))

    def test_unknown_method_returns_name(self):
        self.assertEqual(method_label("my_custom"), "my_custom")


if __name__ == "__main__":
    unittest.main()
