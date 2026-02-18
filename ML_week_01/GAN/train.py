#!/usr/bin/env python3
"""Main training loop for a vanilla GAN on MNIST.

Run with:
    python train.py
"""

import os

# Prevent OpenMP duplicate-library crash on macOS with mixed conda/pip envs
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import (
    BATCH_SIZE,
    BETAS,
    DATA_DIR,
    DEVICE,
    EPOCHS,
    IMAGE_SIZE,
    LATENT_DIM,
    LEARNING_RATE,
    NUM_SAMPLE_IMAGES,
    SAMPLE_INTERVAL,
)
from models import Discriminator, Generator
from utils import reshape_to_image, save_image_grid, save_models


def get_dataloader() -> DataLoader:
    """Download MNIST and return a DataLoader.

    Images are normalised to [-1, 1] so that the Generator's Tanh output
    matches the real-data distribution.

    Returns:
        A DataLoader yielding ``(images, labels)`` batches.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),  # → [-1, 1]
    ])
    dataset = datasets.MNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )


def train() -> None:
    """Execute the full GAN training procedure."""
    print(f"🚀 Training on device: {DEVICE}")
    print(f"   Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    print(f"   Latent dim: {LATENT_DIM} | Image size: {IMAGE_SIZE}")
    print("-" * 60)

    # ── Data ─────────────────────────────────────────────────────────────
    dataloader = get_dataloader()

    # ── Models ───────────────────────────────────────────────────────────
    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    # ── Optimisers & loss ────────────────────────────────────────────────
    opt_g = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=BETAS)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=BETAS)
    criterion = nn.BCELoss()

    # ── Fixed noise for visual tracking ──────────────────────────────────
    fixed_noise = torch.randn(NUM_SAMPLE_IMAGES, LATENT_DIM, device=DEVICE)

    # ── Training loop ────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        d_loss_accum = 0.0
        g_loss_accum = 0.0
        num_batches = 0

        for real_imgs, _ in dataloader:
            batch = real_imgs.size(0)
            real_flat = real_imgs.view(batch, -1).to(DEVICE)

            # Labels
            real_labels = torch.ones(batch, 1, device=DEVICE)
            fake_labels = torch.zeros(batch, 1, device=DEVICE)

            # ── Train Discriminator ──────────────────────────────────────
            z = torch.randn(batch, LATENT_DIM, device=DEVICE)
            fake_flat = generator(z).detach()

            d_real = discriminator(real_flat)
            d_fake = discriminator(fake_flat)

            loss_d = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # ── Train Generator ──────────────────────────────────────────
            z = torch.randn(batch, LATENT_DIM, device=DEVICE)
            fake_flat = generator(z)
            d_fake = discriminator(fake_flat)

            loss_g = criterion(d_fake, real_labels)  # fool D

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            d_loss_accum += loss_d.item()
            g_loss_accum += loss_g.item()
            num_batches += 1

        # ── Epoch summary ────────────────────────────────────────────────
        avg_d = d_loss_accum / num_batches
        avg_g = g_loss_accum / num_batches
        print(f"[Epoch {epoch:>{len(str(EPOCHS))}}/{EPOCHS}] "
              f"D_loss: {avg_d:.4f} | G_loss: {avg_g:.4f}")

        # ── Save sample grid ────────────────────────────────────────────
        if epoch % SAMPLE_INTERVAL == 0 or epoch == 1:
            with torch.no_grad():
                samples = generator(fixed_noise)
            save_image_grid(reshape_to_image(samples), epoch)

    # ── Persist final weights ────────────────────────────────────────────
    print("-" * 60)
    save_models(generator, discriminator)
    print("✅ Training complete!")


if __name__ == "__main__":
    train()
