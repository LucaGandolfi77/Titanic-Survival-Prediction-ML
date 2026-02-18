"""Hyperparameters and configuration constants for the GAN project.

All tuneable values live here so that every other module can import them
without relying on global mutable state.
"""

from pathlib import Path

import torch

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE: torch.device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# ── Latent space ─────────────────────────────────────────────────────────────
LATENT_DIM: int = 100

# ── Image parameters ─────────────────────────────────────────────────────────
IMAGE_SIZE: int = 28 * 28  # flattened 28×28 grayscale
IMAGE_CHANNELS: int = 1
IMAGE_SHAPE: tuple[int, int, int] = (IMAGE_CHANNELS, 28, 28)

# ── Training ─────────────────────────────────────────────────────────────────
EPOCHS: int = 50
BATCH_SIZE: int = 64
LEARNING_RATE: float = 2e-4
BETAS: tuple[float, float] = (0.5, 0.999)

# ── Logging / saving ────────────────────────────────────────────────────────
SAMPLE_INTERVAL: int = 5          # save a grid of fakes every N epochs
NUM_SAMPLE_IMAGES: int = 64       # images per grid

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
SAMPLES_DIR: Path = OUTPUT_DIR / "samples"
WEIGHTS_DIR: Path = OUTPUT_DIR / "weights"
