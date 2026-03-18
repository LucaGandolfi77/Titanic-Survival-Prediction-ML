"""
utils.py — Shared helpers for Lab 6 (GP & PSO).

Provides:
  - Protected GP primitives (div, sqrt, log, clip)
  - Image I/O and noise generation
  - Metric computation (MSE, PSNR, SSIM)
  - Plotting helpers for surfaces, images, and convergence curves
"""

from __future__ import annotations

import math
import os
import random
from typing import Any, Callable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  (side-effect import)
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
IMAGES_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")


# ---------------------------------------------------------------------------
# 1.  Protected GP primitives
# ---------------------------------------------------------------------------

def protected_div(a: float, b: float) -> float:
    """Protected division: returns 1.0 when |b| < 1e-6."""
    return a / b if abs(b) > 1e-6 else 1.0


def protected_sqrt(a: float) -> float:
    """Protected square-root of |a|."""
    return math.sqrt(abs(a))


def protected_log(a: float) -> float:
    """Protected natural log of |a| (returns 0.0 for very small values)."""
    return math.log(abs(a)) if abs(a) > 1e-6 else 0.0


def clipped_add(a: float, b: float) -> float:
    """Addition clipped to [0, 255] (for image GP)."""
    return float(np.clip(a + b, 0.0, 255.0))


def clipped_sub(a: float, b: float) -> float:
    """Subtraction clipped to [0, 255] (for image GP)."""
    return float(np.clip(a - b, 0.0, 255.0))


def clipped_mul(a: float, b: float) -> float:
    """Multiplication clipped to [0, 255] (for image GP)."""
    return float(np.clip(a * b, 0.0, 255.0))


def clipped_div(a: float, b: float) -> float:
    """Protected division clipped to [0, 255] (for image GP)."""
    val = a / b if abs(b) > 1e-6 else 128.0
    return float(np.clip(val, 0.0, 255.0))


def clipped_abs(a: float) -> float:
    """Abs clipped to [0, 255]."""
    return float(np.clip(abs(a), 0.0, 255.0))


def clipped_avg(a: float, b: float) -> float:
    """Average of two values clipped to [0, 255]."""
    return float(np.clip((a + b) / 2.0, 0.0, 255.0))


def clipped_max(a: float, b: float) -> float:
    """Max of two values clipped to [0, 255]."""
    return float(np.clip(max(a, b), 0.0, 255.0))


def clipped_min(a: float, b: float) -> float:
    """Min of two values clipped to [0, 255]."""
    return float(np.clip(min(a, b), 0.0, 255.0))


def edge_detect(a: float, b: float) -> float:
    """Absolute difference (simple edge detector) clipped to [0, 255]."""
    return float(np.clip(abs(a - b), 0.0, 255.0))


def local_contrast(a: float, b: float) -> float:
    """Local contrast enhancement: 128 + gain*(a - b), clipped to [0, 255].

    Amplifies the deviation of pixel *a* from its neighbour *b*.
    """
    gain = 1.5
    return float(np.clip(128.0 + gain * (a - b), 0.0, 255.0))


# ---------------------------------------------------------------------------
# 2.  Image helpers
# ---------------------------------------------------------------------------

def load_grayscale(path: str) -> np.ndarray:
    """Load an image as a float64 grayscale array in [0, 255].

    Args:
        path: file path to the image.

    Returns:
        2-D numpy array of float64 values in [0, 255].
    """
    from PIL import Image
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float64)


def save_grayscale(arr: np.ndarray, path: str) -> None:
    """Save a float64 array as an 8-bit grayscale PNG.

    Args:
        arr: 2-D array in [0, 255].
        path: destination file path.
    """
    from PIL import Image
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
    img.save(path)
    print(f"  [img] Saved: {path}")


def add_uniform_noise(image: np.ndarray, level: int = 15, seed: int = 42) -> np.ndarray:
    """Add uniform noise U(-level, level) and clip to [0, 255].

    Args:
        image: clean grayscale image (float64).
        level: noise half-range.
        seed: random seed.

    Returns:
        Noisy image array.
    """
    rng = np.random.RandomState(seed)
    noise = rng.uniform(-level, level, size=image.shape)
    return np.clip(image + noise, 0.0, 255.0)


def generate_synthetic_image(h: int = 128, w: int = 128, seed: int = 0) -> np.ndarray:
    """Generate a simple synthetic grayscale image (gradient + circles).

    Args:
        h: height in pixels.
        w: width in pixels.
        seed: random seed for circle positions.

    Returns:
        2-D float64 array in [0, 255].
    """
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    # Smooth gradient background
    img = (xx / w * 180 + yy / h * 60).astype(np.float64)
    # Add some circles
    for _ in range(5):
        cx, cy, r = rng.randint(10, w - 10), rng.randint(10, h - 10), rng.randint(8, 25)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        img[mask] = rng.randint(50, 230)
    return np.clip(img, 0, 255)


def generate_enhanced_image(image: np.ndarray) -> np.ndarray:
    """Create a synthetically 'enhanced' version of an image.

    Applies contrast stretching + mild sharpening to simulate a manually-
    enhanced target.

    Args:
        image: input grayscale image in [0, 255].

    Returns:
        Enhanced image in [0, 255].
    """
    from scipy.ndimage import uniform_filter
    # Contrast stretch to use full [0, 255] range
    lo, hi = np.percentile(image, 2), np.percentile(image, 98)
    if hi - lo < 1:
        hi = lo + 1
    stretched = (image - lo) / (hi - lo) * 255.0
    stretched = np.clip(stretched, 0, 255)

    # Mild unsharp mask
    blurred = uniform_filter(stretched, size=3)
    sharpened = np.clip(stretched + 0.5 * (stretched - blurred), 0, 255)
    return sharpened


# ---------------------------------------------------------------------------
# 3.  Metrics
# ---------------------------------------------------------------------------

def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Squared Error between two arrays."""
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> float:
    """Peak Signal-to-Noise Ratio (dB).  Returns inf when MSE == 0."""
    m = mse(a, b)
    if m < 1e-12:
        return float("inf")
    return 10.0 * math.log10(max_val ** 2 / m)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural Similarity Index (simplified, luminance-only).

    Uses a small 7×7 uniform window.  Constants follow the original paper
    (Wang et al., 2004) with L=255.
    """
    from scipy.ndimage import uniform_filter
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    L = 255.0
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    k = 7
    mu_a = uniform_filter(a, size=k)
    mu_b = uniform_filter(b, size=k)
    sig_a2 = uniform_filter(a * a, size=k) - mu_a * mu_a
    sig_b2 = uniform_filter(b * b, size=k) - mu_b * mu_b
    sig_ab = uniform_filter(a * b, size=k) - mu_a * mu_b
    num = (2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (sig_a2 + sig_b2 + C2)
    return float(np.mean(num / den))


# ---------------------------------------------------------------------------
# 4.  Plotting helpers
# ---------------------------------------------------------------------------

def plot_surface_comparison(
    X: np.ndarray, Y: np.ndarray,
    Z_true: np.ndarray, Z_pred: np.ndarray,
    title: str, filename: str,
) -> None:
    """Plot true vs predicted surface and the absolute error, side-by-side.

    Args:
        X, Y: meshgrid coordinate arrays.
        Z_true: ground-truth surface values.
        Z_pred: GP-predicted surface values.
        title: overall figure title.
        filename: path to save the PNG.
    """
    fig = plt.figure(figsize=(18, 5))

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.plot_surface(X, Y, Z_true, cmap="viridis", alpha=0.8)
    ax1.set_title("True f(x, y)")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.plot_surface(X, Y, Z_pred, cmap="viridis", alpha=0.8)
    ax2.set_title("GP Approximation")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.plot_surface(X, Y, np.abs(Z_true - Z_pred), cmap="hot", alpha=0.8)
    ax3.set_title("Absolute Error")
    ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("|err|")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  [plot] Saved: {filename}")


def plot_images_row(
    images: Sequence[np.ndarray],
    titles: Sequence[str],
    filename: str,
    figsize: Tuple[int, int] | None = None,
) -> None:
    """Show several grayscale images in a single row and save to file.

    Args:
        images: list of 2-D arrays.
        titles: list of titles for each subplot.
        filename: destination path.
        figsize: optional figure size.
    """
    n = len(images)
    if figsize is None:
        figsize = (5 * n, 5)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, img, ttl in zip(axes, images, titles):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(ttl, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  [plot] Saved: {filename}")


def plot_convergence(
    curves: dict[str, List[float]],
    title: str,
    filename: str,
    xlabel: str = "Generation / Iteration",
    ylabel: str = "Best Fitness",
) -> None:
    """Plot one or more convergence curves on the same axes.

    Args:
        curves: mapping from label → list of fitness values per gen/iter.
        title: plot title.
        filename: destination path.
        xlabel, ylabel: axis labels.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, vals in curves.items():
        ax.plot(vals, label=label, linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  [plot] Saved: {filename}")


def plot_swarm_on_surface(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
    positions: np.ndarray, title: str, filename: str,
) -> None:
    """Plot a 2-D colour-map slice with swarm positions overlaid.

    Args:
        X, Y: meshgrid coordinate arrays.
        Z: function values on the grid.
        positions: array of shape (N, 2) with (x, y) of each particle.
        title: plot title.
        filename: destination path.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    c = ax.pcolormesh(X, Y, Z, cmap="viridis", shading="auto")
    fig.colorbar(c, ax=ax, label="f(x, y, z*)")
    ax.scatter(positions[:, 0], positions[:, 1], c="red", s=18, edgecolors="white",
               linewidths=0.5, zorder=5, label="Particles")
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  [plot] Saved: {filename}")
