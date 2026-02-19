#!/usr/bin/env python3
"""
CLI generation entry point — generate images from a trained model.

Usage:
    python generate.py --config config/dcgan.yaml --checkpoint outputs/models/dcgan/checkpoint_final.pt --n 64
    python generate.py --config config/conditional_gan.yaml --checkpoint ... --class-id 7 --n 16
    python generate.py --config config/dcgan.yaml --checkpoint ... --interpolate --n-pairs 5
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.visualization import GANVisualizer
from src.models.conditional_gan import build_conditional_gan
from src.models.dcgan import build_dcgan
from src.models.vanilla_gan import build_vanilla_gan
from src.utils.checkpointing import load_checkpoint
from src.utils.config_loader import load_config

_MODEL_BUILDERS = {
    "vanilla_gan": build_vanilla_gan,
    "dcgan": build_dcgan,
    "conditional_gan": build_conditional_gan,
}


@click.command()
@click.option("--config", "-c", type=click.Path(exists=True), required=True)
@click.option("--checkpoint", "-k", type=click.Path(exists=True), required=True)
@click.option("--n", type=int, default=64, help="Number of images to generate.")
@click.option("--class-id", type=int, default=None, help="Class ID for conditional generation.")
@click.option("--all-classes", is_flag=True, help="Generate samples for all classes (cGAN).")
@click.option("--interpolate", is_flag=True, help="Generate latent interpolation.")
@click.option("--n-pairs", type=int, default=5, help="Number of interpolation pairs.")
@click.option("--n-steps", type=int, default=10, help="Interpolation steps per pair.")
@click.option("--method", type=click.Choice(["linear", "slerp"]), default="slerp")
@click.option("--output", "-o", type=str, default=None, help="Output filename.")
@click.option("--seed", type=int, default=None, help="Random seed.")
def main(
    config: str,
    checkpoint: str,
    n: int,
    class_id: int | None,
    all_classes: bool,
    interpolate: bool,
    n_pairs: int,
    n_steps: int,
    method: str,
    output: str | None,
    seed: int | None,
) -> None:
    """Generate images using a trained GAN model."""
    cfg = load_config(config)
    device = torch.device(cfg["experiment"]["device"])
    stage = cfg["experiment"]["stage"]
    latent_dim = cfg["model"]["latent_dim"]

    if seed is not None:
        torch.manual_seed(seed)

    click.echo(f"Loading {stage} model from: {checkpoint}")

    # Build and load model
    builder = _MODEL_BUILDERS[stage]
    generator, discriminator = builder(cfg)
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    load_checkpoint(checkpoint, generator, discriminator, device=device)
    generator.eval()

    visualizer = GANVisualizer(
        output_dir=cfg.get("paths", {}).get("samples_dir", "outputs").rsplit("/", 1)[0]
        if "/" in cfg.get("paths", {}).get("samples_dir", "outputs")
        else "outputs"
    )

    # ── Interpolation mode ───────────────────────────────────────────────
    if interpolate:
        fname = output or f"{stage}_interpolation.png"
        labels = None
        if stage == "conditional_gan":
            n_classes = cfg["model"].get("n_classes", 10)
            labels = torch.arange(min(n_pairs, n_classes), device=device)

        path = visualizer.save_interpolation(
            generator=generator,
            latent_dim=latent_dim,
            device=device,
            n_pairs=n_pairs,
            n_steps=n_steps,
            method=method,
            filename=fname,
            labels=labels,
        )
        click.echo(f"Interpolation saved to: {path}")
        return

    # ── Class-conditioned grid (all classes) ─────────────────────────────
    if all_classes and stage == "conditional_gan":
        fname = output or f"{stage}_all_classes.png"
        n_classes = cfg["model"].get("n_classes", 10)
        path = visualizer.save_conditional_grid(
            generator=generator,
            latent_dim=latent_dim,
            n_classes=n_classes,
            device=device,
            n_samples_per_class=n // n_classes,
            filename=fname,
        )
        click.echo(f"Conditional grid saved to: {path}")
        return

    # ── Standard generation ──────────────────────────────────────────────
    with torch.no_grad():
        z = torch.randn(n, latent_dim, device=device)
        if stage == "conditional_gan":
            if class_id is not None:
                labels = torch.full((n,), class_id, dtype=torch.long, device=device)
            else:
                n_classes = cfg["model"].get("n_classes", 10)
                labels = torch.randint(0, n_classes, (n,), device=device)
            images = generator(z, labels)
        else:
            images = generator(z)

    fname = output or f"{stage}_generated.png"
    path = visualizer.save_sample_grid(images, filename=fname)
    click.echo(f"Generated {n} images → {path}")


if __name__ == "__main__":
    main()
