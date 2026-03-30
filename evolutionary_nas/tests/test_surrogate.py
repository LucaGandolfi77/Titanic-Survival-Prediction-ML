"""Tests for surrogate predictor and feature extraction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
import numpy as np
from search_space.mlp_space import random_mlp_genome
from search_space.cnn_space import random_cnn_genome
from search_space.genome_encoder import repair_mlp, repair_cnn

rng = np.random.default_rng(42)
from surrogate.feature_extractor import genome_to_features_mlp, genome_to_features_cnn
from surrogate.predictor import SurrogatePredictor


class TestFeatureExtractor:
    def test_mlp_features_shape(self):
        g = random_mlp_genome(rng)
        g = repair_mlp(g)
        f = genome_to_features_mlp(g)
        assert isinstance(f, np.ndarray)
        assert f.ndim == 1
        assert len(f) > 0

    def test_cnn_features_shape(self):
        g = random_cnn_genome(rng)
        g = repair_cnn(g)
        f = genome_to_features_cnn(g)
        assert isinstance(f, np.ndarray)
        assert f.ndim == 1
        assert len(f) > 0

    def test_features_deterministic(self):
        g = random_mlp_genome(rng)
        g = repair_mlp(g)
        f1 = genome_to_features_mlp(g)
        f2 = genome_to_features_mlp(g)
        np.testing.assert_array_equal(f1, f2)


class TestSurrogatePredictor:
    def _make_training_data(self, n=60):
        genomes = []
        scores = []
        for _ in range(n):
            g = random_mlp_genome(rng)
            g = repair_mlp(g)
            genomes.append(g)
            scores.append(np.random.uniform(0.5, 0.99))
        return genomes, scores

    def test_fit_predict(self):
        predictor = SurrogatePredictor()
        genomes, scores = self._make_training_data()
        for g, s in zip(genomes, scores):
            predictor.add_observation(g, "mlp", s)
        predictor.fit()

        test_g = random_mlp_genome(rng)
        test_g = repair_mlp(test_g)
        pred = predictor.predict(test_g, "mlp")
        assert isinstance(pred, float)

    def test_predict_batch(self):
        predictor = SurrogatePredictor()
        genomes, scores = self._make_training_data()
        for g, s in zip(genomes, scores):
            predictor.add_observation(g, "mlp", s)
        predictor.fit()

        batch = [repair_mlp(random_mlp_genome(rng)) for _ in range(5)]
        preds = predictor.predict_batch(batch, "mlp")
        assert len(preds) == 5

    def test_uncertainty(self):
        predictor = SurrogatePredictor()
        genomes, scores = self._make_training_data()
        for g, s in zip(genomes, scores):
            predictor.add_observation(g, "mlp", s)
        predictor.fit()

        test_g = repair_mlp(random_mlp_genome(rng))
        unc = predictor.uncertainty(test_g, "mlp")
        assert isinstance(unc, float)
        assert unc >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
