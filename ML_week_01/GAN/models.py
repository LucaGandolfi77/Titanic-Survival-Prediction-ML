"""Generator and Discriminator architectures for a vanilla GAN on MNIST.

Both networks are fully-connected MLPs.  A CNN variant is provided at the
bottom of the file as commented-out code for future experimentation.
"""

import torch
import torch.nn as nn

from config import IMAGE_SIZE, LATENT_DIM


class Generator(nn.Module):
    """Maps a latent vector *z* to a flattened 28×28 grayscale image.

    Architecture
    ------------
    z (LATENT_DIM) → 256 → 512 → 1024 → IMAGE_SIZE

    Activations: ReLU (hidden), Tanh (output).
    """

    def __init__(self, latent_dim: int = LATENT_DIM, img_size: int = IMAGE_SIZE) -> None:
        """Initialise the Generator.

        Args:
            latent_dim: Dimensionality of the input noise vector.
            img_size: Number of pixels in the flattened output image.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, img_size),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate fake images from noise.

        Args:
            z: Tensor of shape ``(batch, latent_dim)`` sampled from N(0, 1).

        Returns:
            Tensor of shape ``(batch, img_size)`` with values in [-1, 1].
        """
        return self.net(z)


class Discriminator(nn.Module):
    """Binary classifier that distinguishes real from generated images.

    Architecture
    ------------
    IMAGE_SIZE → 512 → 256 → 1

    Activations: LeakyReLU (hidden), Sigmoid (output).
    """

    def __init__(self, img_size: int = IMAGE_SIZE) -> None:
        """Initialise the Discriminator.

        Args:
            img_size: Number of pixels in the flattened input image.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Classify an image as real (≈1) or fake (≈0).

        Args:
            img: Tensor of shape ``(batch, img_size)``.

        Returns:
            Tensor of shape ``(batch, 1)`` with probabilities.
        """
        return self.net(img)


# ═══════════════════════════════════════════════════════════════════════════
# CNN variants (uncomment to use instead of the MLP versions above)
# ═══════════════════════════════════════════════════════════════════════════
#
# class CNNGenerator(nn.Module):
#     """Deconvolutional generator: z → 1×28×28."""
#
#     def __init__(self, latent_dim: int = LATENT_DIM) -> None:
#         super().__init__()
#         self.fc = nn.Linear(latent_dim, 256 * 7 * 7)
#         self.net = nn.Sequential(
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # → 14×14
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(128, 1, 4, stride=2, padding=1),    # → 28×28
#             nn.Tanh(),
#         )
#
#     def forward(self, z: torch.Tensor) -> torch.Tensor:
#         x = self.fc(z).view(-1, 256, 7, 7)
#         return self.net(x)
#
#
# class CNNDiscriminator(nn.Module):
#     """Convolutional discriminator: 1×28×28 → real/fake."""
#
#     def __init__(self) -> None:
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(1, 64, 4, stride=2, padding=1),   # → 14×14
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 128, 4, stride=2, padding=1),  # → 7×7
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(128 * 7 * 7, 1),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, img: torch.Tensor) -> torch.Tensor:
#         features = self.net(img).view(img.size(0), -1)
#         return self.fc(features)
