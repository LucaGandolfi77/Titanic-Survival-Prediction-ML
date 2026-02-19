"""
Unit tests for GAN model architectures.
"""

import pytest
import torch

from src.models.vanilla_gan import VanillaGenerator, VanillaDiscriminator, build_vanilla_gan
from src.models.dcgan import DCGANGenerator, DCGANDiscriminator, build_dcgan
from src.models.conditional_gan import (
    ConditionalGenerator,
    ConditionalDiscriminator,
    build_conditional_gan,
)
from src.models.layers import (
    MinibatchStdDev,
    PixelNorm,
    SelfAttention,
    ConditionalBatchNorm2d,
    weights_init_normal,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def latent_dim():
    return 100


# ── Vanilla GAN Tests ────────────────────────────────────────────────────

class TestVanillaGAN:
    def test_generator_output_shape(self, device, batch_size, latent_dim):
        gen = VanillaGenerator(latent_dim=latent_dim, image_shape=(1, 28, 28))
        gen = gen.to(device)
        z = torch.randn(batch_size, latent_dim, device=device)
        out = gen(z)
        assert out.shape == (batch_size, 1, 28, 28)

    def test_generator_output_range(self, device, latent_dim):
        gen = VanillaGenerator(latent_dim=latent_dim).to(device)
        z = torch.randn(2, latent_dim, device=device)
        out = gen(z)
        assert out.min() >= -1.0
        assert out.max() <= 1.0

    def test_discriminator_output_shape(self, device, batch_size):
        disc = VanillaDiscriminator(image_shape=(1, 28, 28))
        disc = disc.to(device)
        img = torch.randn(batch_size, 1, 28, 28, device=device)
        out = disc(img)
        assert out.shape == (batch_size, 1)

    def test_build_vanilla_gan(self):
        config = {
            "model": {
                "latent_dim": 64,
                "n_channels": 1,
                "image_size": 28,
                "generator": {"use_batch_norm": True, "leaky_slope": 0.2},
                "discriminator": {"use_spectral_norm": False, "leaky_slope": 0.2},
            }
        }
        gen, disc = build_vanilla_gan(config)
        assert isinstance(gen, VanillaGenerator)
        assert isinstance(disc, VanillaDiscriminator)


# ── DCGAN Tests ──────────────────────────────────────────────────────────

class TestDCGAN:
    def test_generator_output_shape(self, device, batch_size, latent_dim):
        gen = DCGANGenerator(latent_dim=latent_dim, n_channels=1, ngf=32)
        gen = gen.to(device)
        z = torch.randn(batch_size, latent_dim, device=device)
        out = gen(z)
        assert out.shape == (batch_size, 1, 64, 64)

    def test_generator_accepts_4d_input(self, device, latent_dim):
        gen = DCGANGenerator(latent_dim=latent_dim).to(device)
        z = torch.randn(2, latent_dim, 1, 1, device=device)
        out = gen(z)
        assert out.shape == (2, 1, 64, 64)

    def test_discriminator_output_shape(self, device, batch_size):
        disc = DCGANDiscriminator(n_channels=1, ndf=32)
        disc = disc.to(device)
        img = torch.randn(batch_size, 1, 64, 64, device=device)
        out = disc(img)
        assert out.shape == (batch_size, 1)

    def test_spectral_norm_applied(self):
        disc = DCGANDiscriminator(use_spectral_norm=True)
        has_sn = any(
            hasattr(m, "weight_orig")
            for m in disc.modules()
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))
        )
        assert has_sn


# ── Conditional GAN Tests ────────────────────────────────────────────────

class TestConditionalGAN:
    def test_generator_output_shape(self, device, batch_size, latent_dim):
        gen = ConditionalGenerator(
            latent_dim=latent_dim, n_channels=1, n_classes=10, ngf=32
        ).to(device)
        z = torch.randn(batch_size, latent_dim, device=device)
        labels = torch.randint(0, 10, (batch_size,), device=device)
        out = gen(z, labels)
        assert out.shape == (batch_size, 1, 64, 64)

    def test_discriminator_output_shape(self, device, batch_size):
        disc = ConditionalDiscriminator(
            n_channels=1, n_classes=10, ndf=32
        ).to(device)
        img = torch.randn(batch_size, 1, 64, 64, device=device)
        labels = torch.randint(0, 10, (batch_size,), device=device)
        out = disc(img, labels)
        assert out.shape == (batch_size, 1)

    def test_different_labels_different_output(self, device, latent_dim):
        gen = ConditionalGenerator(
            latent_dim=latent_dim, n_classes=10, ngf=32
        ).to(device)
        z = torch.randn(1, latent_dim, device=device).expand(2, -1)
        labels = torch.tensor([0, 5], device=device)
        with torch.no_grad():
            out = gen(z, labels)
        # Different labels should produce different images
        assert not torch.allclose(out[0], out[1], atol=1e-4)


# ── Custom Layers Tests ─────────────────────────────────────────────────

class TestCustomLayers:
    def test_minibatch_stddev(self, device):
        layer = MinibatchStdDev(group_size=2).to(device)
        x = torch.randn(4, 8, 4, 4, device=device)
        out = layer(x)
        assert out.shape == (4, 9, 4, 4)  # +1 channel

    def test_pixel_norm(self, device):
        layer = PixelNorm()
        x = torch.randn(2, 16, 8, 8, device=device)
        out = layer(x)
        assert out.shape == x.shape
        # Check normalization (roughly unit length per pixel)
        norms = torch.mean(out ** 2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=0.1)

    def test_self_attention(self, device):
        layer = SelfAttention(in_channels=32).to(device)
        x = torch.randn(2, 32, 8, 8, device=device)
        out = layer(x)
        assert out.shape == x.shape

    def test_conditional_batchnorm(self, device):
        layer = ConditionalBatchNorm2d(num_features=16, num_classes=10).to(device)
        x = torch.randn(4, 16, 8, 8, device=device)
        y = torch.randint(0, 10, (4,), device=device)
        out = layer(x, y)
        assert out.shape == x.shape
