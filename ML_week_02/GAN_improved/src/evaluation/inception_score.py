"""
Inception Score (IS) calculator.

IS measures the quality and diversity of generated images by evaluating
how confidently InceptionV3 classifies them and how diverse those
classifications are.

Higher IS = better quality + diversity.

Reference:
    Salimans et al., "Improved Techniques for Training GANs", NeurIPS 2016.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms


class InceptionScoreCalculator:
    """Compute Inception Score for generated images.

    Usage:
        is_calc = InceptionScoreCalculator(device="mps")
        mean_is, std_is = is_calc.compute_inception_score(fake_images, splits=10)
    """

    def __init__(
        self,
        device: torch.device | str = "cpu",
        batch_size: int = 64,
    ) -> None:
        self.device = torch.device(device)
        self.batch_size = batch_size
        self._model: nn.Module | None = None

    @property
    def model(self) -> nn.Module:
        """Lazy-load InceptionV3 classifier."""
        if self._model is None:
            self._model = models.inception_v3(
                weights=models.Inception_V3_Weights.DEFAULT,
                transform_input=False,
            )
            self._model.eval()
            self._model.to(self.device)
            for param in self._model.parameters():
                param.requires_grad = False
        return self._model

    @torch.no_grad()
    def _get_predictions(self, images: Tensor) -> np.ndarray:
        """Get softmax predictions from InceptionV3.

        Args:
            images: Images [N, C, H, W] in [-1, 1].

        Returns:
            Softmax predictions [N, 1000].
        """
        dataset = TensorDataset(images)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_preds = []
        for (batch,) in loader:
            batch = batch.to(self.device)

            # Ensure 3-channel
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)

            # Resize to 299×299
            if batch.shape[2] != 299 or batch.shape[3] != 299:
                batch = F.interpolate(
                    batch, size=(299, 299), mode="bilinear", align_corners=False
                )

            # Normalize to ImageNet range
            batch = (batch + 1) / 2.0
            batch = transforms.functional.normalize(
                batch, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

            logits = self.model(batch)
            # InceptionV3 returns InceptionOutputs in training mode, plain tensor in eval
            if isinstance(logits, torch.Tensor):
                preds = F.softmax(logits, dim=1)
            else:
                preds = F.softmax(logits.logits, dim=1)

            all_preds.append(preds.cpu().numpy())

        return np.concatenate(all_preds, axis=0)

    def compute_inception_score(
        self,
        images: Tensor,
        splits: int = 10,
    ) -> tuple[float, float]:
        """Compute Inception Score.

        IS = exp(E_x[KL(p(y|x) || p(y))])

        Args:
            images: Generated images [N, C, H, W] in [-1, 1].
            splits: Number of splits for computing mean ± std.

        Returns:
            Tuple of (mean_IS, std_IS).
        """
        preds = self._get_predictions(images)
        n = preds.shape[0]

        # Split into groups and compute IS for each
        scores = []
        split_size = n // splits

        for i in range(splits):
            part = preds[i * split_size : (i + 1) * split_size]
            # p(y) = marginal over the split
            p_y = np.mean(part, axis=0, keepdims=True)
            # KL divergence: sum p(y|x) * log(p(y|x) / p(y))
            kl = part * (np.log(part + 1e-16) - np.log(p_y + 1e-16))
            kl = np.mean(np.sum(kl, axis=1))
            scores.append(np.exp(kl))

        return float(np.mean(scores)), float(np.std(scores))
