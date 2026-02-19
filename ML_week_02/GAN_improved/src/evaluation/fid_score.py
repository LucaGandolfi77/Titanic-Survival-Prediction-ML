"""
Fréchet Inception Distance (FID) calculator.

FID measures the distance between the feature distributions of real and
generated images using an InceptionV3 feature extractor.

Lower FID = better quality + diversity.

Reference:
    Heusel et al., "GANs Trained by a Two Time-Scale Update Rule
    Converge to a Local Nash Equilibrium", NeurIPS 2017.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms


class InceptionV3Features(nn.Module):
    """InceptionV3 wrapper that extracts pool3 features (2048-d).

    Used for both FID and Inception Score computation.
    """

    def __init__(self, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.device = torch.device(device)

        # Load pretrained InceptionV3
        inception = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT,
            transform_input=False,
        )
        inception.eval()

        # Extract layers up to pool3 (before aux logits and final fc)
        self.blocks = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        ).to(self.device)

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """Extract 2048-d features from images.

        Args:
            x: Images [B, C, H, W]. Will be resized to 299×299 and
               converted to 3-channel if needed.

        Returns:
            Feature vectors [B, 2048].
        """
        # Ensure 3-channel
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Resize to 299×299 (InceptionV3 input size)
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = torch.nn.functional.interpolate(
                x, size=(299, 299), mode="bilinear", align_corners=False
            )

        # Normalize to ImageNet range
        # Input should be [-1, 1] (from GAN output), scale to [0, 1] first
        x = (x + 1) / 2.0
        x = transforms.functional.normalize(
            x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        features = self.blocks(x)
        return features.view(features.size(0), -1)  # [B, 2048]


class FIDCalculator:
    """Compute Fréchet Inception Distance between real and generated images.

    Usage:
        fid_calc = FIDCalculator(device="mps")
        fid = fid_calc.compute_fid(real_images, fake_images)
    """

    def __init__(
        self,
        device: torch.device | str = "cpu",
        batch_size: int = 64,
    ) -> None:
        self.device = torch.device(device)
        self.batch_size = batch_size
        self._inception: InceptionV3Features | None = None

    @property
    def inception(self) -> InceptionV3Features:
        """Lazy-load InceptionV3 features extractor."""
        if self._inception is None:
            self._inception = InceptionV3Features(device=self.device)
            self._inception.eval()
        return self._inception

    @torch.no_grad()
    def extract_features(self, images: Tensor) -> np.ndarray:
        """Extract InceptionV3 features from a batch of images.

        Args:
            images: Images tensor [N, C, H, W] in [-1, 1].

        Returns:
            Feature array [N, 2048].
        """
        dataset = TensorDataset(images)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_features = []
        for (batch,) in loader:
            batch = batch.to(self.device)
            features = self.inception(batch)
            all_features.append(features.cpu().numpy())

        return np.concatenate(all_features, axis=0)

    def compute_statistics(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute mean and covariance of feature vectors.

        Args:
            features: Feature array [N, 2048].

        Returns:
            Tuple of (mean [2048], covariance [2048, 2048]).
        """
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def compute_fid(
        self,
        real_images: Tensor,
        fake_images: Tensor,
    ) -> float:
        """Compute FID between real and generated images.

        Args:
            real_images: Real images [N, C, H, W] in [-1, 1].
            fake_images: Generated images [M, C, H, W] in [-1, 1].

        Returns:
            FID score (lower is better).
        """
        real_features = self.extract_features(real_images)
        fake_features = self.extract_features(fake_images)

        mu_real, sigma_real = self.compute_statistics(real_features)
        mu_fake, sigma_fake = self.compute_statistics(fake_features)

        return self._calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

    def compute_fid_from_statistics(
        self,
        fake_images: Tensor,
        real_mu: np.ndarray,
        real_sigma: np.ndarray,
    ) -> float:
        """Compute FID using precomputed real data statistics.

        Args:
            fake_images: Generated images [M, C, H, W] in [-1, 1].
            real_mu: Precomputed mean of real features.
            real_sigma: Precomputed covariance of real features.

        Returns:
            FID score.
        """
        fake_features = self.extract_features(fake_images)
        mu_fake, sigma_fake = self.compute_statistics(fake_features)
        return self._calculate_frechet_distance(real_mu, real_sigma, mu_fake, sigma_fake)

    @staticmethod
    def _calculate_frechet_distance(
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
        eps: float = 1e-6,
    ) -> float:
        """Compute the Fréchet distance between two multivariate Gaussians.

        FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1 @ sigma2))

        Args:
            mu1, sigma1: Statistics of distribution 1.
            mu2, sigma2: Statistics of distribution 2.
            eps: Small constant for numerical stability.

        Returns:
            Fréchet distance (FID score).
        """
        diff = mu1 - mu2

        # Product of covariances
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

        # Numerical stability — handle imaginary components
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                raise ValueError("Imaginary component in sqrtm result")
            covmean = covmean.real

        fid = diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        return float(fid)

    def save_statistics(
        self, images: Tensor, save_path: str | Path
    ) -> None:
        """Precompute and save feature statistics for a dataset.

        Args:
            images: Dataset images [N, C, H, W].
            save_path: Path to save the .npz file.
        """
        features = self.extract_features(images)
        mu, sigma = self.compute_statistics(features)
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(save_path), mu=mu, sigma=sigma)

    @staticmethod
    def load_statistics(stats_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """Load precomputed feature statistics.

        Args:
            stats_path: Path to the .npz file.

        Returns:
            Tuple of (mu, sigma).
        """
        data = np.load(str(stats_path))
        return data["mu"], data["sigma"]
