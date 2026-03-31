"""Tests for validation indices (internal + external)."""

from __future__ import annotations

import unittest

import numpy as np
from sklearn.datasets import make_blobs

from validation.internal_indices import compute_internal_indices
from validation.external_indices import compute_external_indices


class TestInternalIndices(unittest.TestCase):
    """Verify internal validation metrics."""

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = make_blobs(
            n_samples=200, centers=3, cluster_std=0.5, random_state=42,
        )

    def test_returns_all_keys(self):
        result = compute_internal_indices(self.X, self.y)
        expected = {"silhouette", "calinski_harabasz", "davies_bouldin",
                    "dunn", "wcss", "bcss"}
        self.assertEqual(set(result.keys()), expected)

    def test_silhouette_range(self):
        result = compute_internal_indices(self.X, self.y)
        self.assertGreaterEqual(result["silhouette"], -1.0)
        self.assertLessEqual(result["silhouette"], 1.0)

    def test_calinski_harabasz_positive(self):
        result = compute_internal_indices(self.X, self.y)
        self.assertGreater(result["calinski_harabasz"], 0)

    def test_davies_bouldin_positive(self):
        result = compute_internal_indices(self.X, self.y)
        self.assertGreater(result["davies_bouldin"], 0)

    def test_dunn_positive(self):
        result = compute_internal_indices(self.X, self.y)
        self.assertGreaterEqual(result["dunn"], 0)

    def test_wcss_positive(self):
        result = compute_internal_indices(self.X, self.y)
        self.assertGreater(result["wcss"], 0)

    def test_bcss_positive(self):
        result = compute_internal_indices(self.X, self.y)
        self.assertGreater(result["bcss"], 0)

    def test_well_separated_high_silhouette(self):
        X, y = make_blobs(n_samples=200, centers=3, cluster_std=0.3,
                          random_state=0)
        result = compute_internal_indices(X, y)
        self.assertGreater(result["silhouette"], 0.5)

    def test_single_cluster_edge_case(self):
        labels = np.zeros(len(self.X), dtype=int)
        result = compute_internal_indices(self.X, labels)
        self.assertEqual(result["silhouette"], -1.0)

    def test_all_singletons_edge_case(self):
        labels = np.arange(len(self.X))
        result = compute_internal_indices(self.X, labels)
        self.assertEqual(result["silhouette"], -1.0)


class TestExternalIndices(unittest.TestCase):
    """Verify external validation metrics."""

    @classmethod
    def setUpClass(cls):
        cls.y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    def test_perfect_match(self):
        result = compute_external_indices(self.y_true, self.y_true)
        self.assertAlmostEqual(result["ari"], 1.0, places=5)
        self.assertAlmostEqual(result["nmi"], 1.0, places=5)
        self.assertAlmostEqual(result["v_measure"], 1.0, places=5)

    def test_random_labels_low_ari(self):
        rng = np.random.default_rng(42)
        random_labels = rng.integers(0, 3, size=len(self.y_true))
        result = compute_external_indices(self.y_true, random_labels)
        self.assertLess(result["ari"], 0.5)

    def test_returns_all_keys(self):
        result = compute_external_indices(self.y_true, self.y_true)
        expected = {"ari", "nmi", "fmi", "homogeneity", "completeness", "v_measure"}
        self.assertEqual(set(result.keys()), expected)

    def test_all_values_bounded(self):
        rng = np.random.default_rng(7)
        labels = rng.integers(0, 2, size=len(self.y_true))
        result = compute_external_indices(self.y_true, labels)
        for key in ("nmi", "fmi", "homogeneity", "completeness", "v_measure"):
            self.assertGreaterEqual(result[key], 0.0)
            self.assertLessEqual(result[key], 1.0)
        # ARI can be negative
        self.assertGreaterEqual(result["ari"], -1.0)
        self.assertLessEqual(result["ari"], 1.0)


if __name__ == "__main__":
    unittest.main()
