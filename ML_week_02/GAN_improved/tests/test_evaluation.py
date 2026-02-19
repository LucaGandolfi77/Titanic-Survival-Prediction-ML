"""
Unit tests for evaluation modules.
"""

import pytest
import torch
import numpy as np

from src.evaluation.visualization import GANVisualizer


class TestVisualization:
    @pytest.fixture
    def visualizer(self, tmp_path):
        return GANVisualizer(output_dir=str(tmp_path))

    def test_make_image_grid(self):
        images = torch.randn(16, 1, 28, 28)
        grid = GANVisualizer.make_image_grid(images, nrow=4)
        assert grid.dim() == 3  # [C, H, W]
        assert grid.shape[0] == 3  # make_grid converts 1-ch to 3-ch

    def test_save_sample_grid(self, visualizer, tmp_path):
        images = torch.randn(16, 1, 28, 28)
        path = visualizer.save_sample_grid(images, filename="test_grid.png")
        assert path.exists()
        assert path.suffix == ".png"

    def test_linear_interpolation(self):
        z1 = torch.randn(100)
        z2 = torch.randn(100)
        interp = GANVisualizer.linear_interpolation(z1, z2, n_steps=5)
        assert interp.shape == (5, 100)
        # First and last should match z1 and z2
        assert torch.allclose(interp[0], z1, atol=1e-5)
        assert torch.allclose(interp[-1], z2, atol=1e-5)

    def test_spherical_interpolation(self):
        z1 = torch.randn(100)
        z2 = torch.randn(100)
        interp = GANVisualizer.spherical_interpolation(z1, z2, n_steps=8)
        assert interp.shape == (8, 100)

    def test_plot_training_curves(self, tmp_path):
        g_losses = np.random.rand(100).tolist()
        d_losses = np.random.rand(100).tolist()
        save_path = tmp_path / "curves.png"
        GANVisualizer.plot_training_curves(g_losses, d_losses, save_path)
        assert save_path.exists()

    def test_plot_with_fid(self, tmp_path):
        g_losses = np.random.rand(100).tolist()
        d_losses = np.random.rand(100).tolist()
        fid_scores = [(10, 150.0), (20, 120.0), (30, 90.0)]
        save_path = tmp_path / "curves_fid.png"
        GANVisualizer.plot_training_curves(g_losses, d_losses, save_path, fid_scores)
        assert save_path.exists()
