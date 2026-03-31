"""Tests for bootstrap stability and k-selection methods."""

from __future__ import annotations

import unittest

import numpy as np
from sklearn.datasets import make_blobs

from validation.stability import bootstrap_stability, stability_over_k
from validation.k_selection import (
    select_k_elbow,
    select_k_silhouette,
    select_k_gap,
)
from validation.gap_statistic import gap_statistic


class TestBootstrapStability(unittest.TestCase):
    """Verify bootstrap stability analysis."""

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = make_blobs(
            n_samples=200, centers=3, cluster_std=0.5, random_state=42,
        )

    def test_returns_dict_keys(self):
        result = bootstrap_stability(self.X, n_clusters=3, n_bootstrap=5,
                                     random_state=42)
        for key in ("mean_ari", "std_ari", "all_ari"):
            self.assertIn(key, result)

    def test_mean_ari_range(self):
        result = bootstrap_stability(self.X, n_clusters=3, n_bootstrap=5,
                                     random_state=42)
        self.assertGreaterEqual(result["mean_ari"], -1.0)
        self.assertLessEqual(result["mean_ari"], 1.0)

    def test_all_ari_length(self):
        n_boot = 7
        result = bootstrap_stability(self.X, n_clusters=3, n_bootstrap=n_boot,
                                     random_state=42)
        self.assertEqual(len(result["all_ari"]), n_boot)

    def test_well_separated_high_stability(self):
        X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.3,
                          random_state=0)
        result = bootstrap_stability(X, n_clusters=3, n_bootstrap=10,
                                     random_state=0)
        self.assertGreater(result["mean_ari"], 0.5)


class TestStabilityOverK(unittest.TestCase):
    """Verify stability_over_k output shape and ranges."""

    @classmethod
    def setUpClass(cls):
        cls.X, _ = make_blobs(
            n_samples=150, centers=3, cluster_std=0.5, random_state=42,
        )

    def test_returns_list_of_correct_length(self):
        k_range = (2, 3, 4, 5)
        result = stability_over_k(self.X, k_range=k_range, n_bootstrap=3,
                                  random_state=42)
        self.assertEqual(len(result), len(k_range))

    def test_each_entry_is_dict(self):
        result = stability_over_k(self.X, k_range=(2, 3), n_bootstrap=3,
                                  random_state=42)
        for entry in result:
            self.assertIn("mean_ari", entry)


class TestGapStatistic(unittest.TestCase):
    """Verify gap statistic output."""

    @classmethod
    def setUpClass(cls):
        cls.X, _ = make_blobs(
            n_samples=150, centers=3, cluster_std=0.5, random_state=42,
        )

    def test_returns_required_keys(self):
        result = gap_statistic(self.X, k_range=(2, 3, 4, 5), n_refs=3)
        for key in ("gaps", "sk", "k_range", "best_k", "wcss"):
            self.assertIn(key, result)

    def test_best_k_in_range(self):
        k_range = (2, 3, 4, 5, 6)
        result = gap_statistic(self.X, k_range=k_range, n_refs=3)
        self.assertIn(result["best_k"], k_range)

    def test_gaps_length(self):
        k_range = (2, 3, 4)
        result = gap_statistic(self.X, k_range=k_range, n_refs=3)
        self.assertEqual(len(result["gaps"]), len(k_range))


class TestKSelection(unittest.TestCase):
    """Verify k-selection helpers."""

    @classmethod
    def setUpClass(cls):
        cls.X, _ = make_blobs(
            n_samples=200, centers=3, cluster_std=0.5, random_state=42,
        )

    def test_elbow_returns_dict_with_best_k(self):
        result = select_k_elbow(self.X, k_range=(2, 3, 4, 5, 6))
        self.assertIn("best_k", result)
        self.assertIsInstance(result["best_k"], int)
        self.assertIn(result["best_k"], (2, 3, 4, 5, 6))

    def test_silhouette_returns_dict_with_best_k(self):
        result = select_k_silhouette(self.X, k_range=(2, 3, 4, 5, 6))
        self.assertIn("best_k", result)
        self.assertIsInstance(result["best_k"], int)
        self.assertIn(result["best_k"], (2, 3, 4, 5, 6))

    def test_gap_returns_dict_with_best_k(self):
        result = select_k_gap(self.X, k_range=(2, 3, 4, 5), n_refs=3)
        self.assertIn("best_k", result)
        self.assertIsInstance(result["best_k"], int)
        self.assertIn(result["best_k"], (2, 3, 4, 5))

    def test_true_k_often_selected_for_clean_data(self):
        X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.4,
                          random_state=0)
        k_range = (2, 3, 4, 5, 6, 7)
        k_elb = select_k_elbow(X, k_range=k_range)["best_k"]
        k_sil = select_k_silhouette(X, k_range=k_range)["best_k"]
        # At least one method should find k=3
        self.assertTrue(
            k_elb == 3 or k_sil == 3,
            f"Neither elbow ({k_elb}) nor silhouette ({k_sil}) found true k=3",
        )


if __name__ == "__main__":
    unittest.main()
