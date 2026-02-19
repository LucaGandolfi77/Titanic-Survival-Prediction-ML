#!/usr/bin/env python3
"""
CLI evaluation entry point — compute FID and Inception Score.

Usage:
    python evaluate.py --config config/dcgan.yaml --checkpoint outputs/models/dcgan/checkpoint_final.pt
    python evaluate.py --config config/conditional_gan.yaml --n-samples 10000
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloaders import get_dataloader_from_config
from src.evaluation.fid_score import FIDCalculator
from src.evaluation.inception_score import InceptionScoreCalculator
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
@click.option("--n-samples", "-n", type=int, default=5000, help="Number of samples for evaluation.")
@click.option("--compute-fid / --no-fid", default=True, help="Compute FID score.")
@click.option("--compute-is / --no-is", default=True, help="Compute Inception Score.")
@click.option("--batch-size", "-b", type=int, default=64, help="Batch size for evaluation.")
def main(
    config: str,
    checkpoint: str,
    n_samples: int,
    compute_fid: bool,
    compute_is: bool,
    batch_size: int,
) -> None:
    """Evaluate a trained GAN model with FID and Inception Score."""
    cfg = load_config(config)
    device = torch.device(cfg["experiment"]["device"])
    stage = cfg["experiment"]["stage"]

    click.echo(f"Evaluating {stage} on {device}")
    click.echo(f"Checkpoint: {checkpoint}")

    # Build models
    builder = _MODEL_BUILDERS.get(stage)
    if builder is None:
        raise ValueError(f"Unknown stage: {stage}")

    generator, discriminator = builder(cfg)
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # Load checkpoint
    load_checkpoint(
        checkpoint, generator, discriminator, device=device
    )
    generator.eval()

    click.echo(f"Generating {n_samples} samples...")

    # Generate fake images
    latent_dim = cfg["model"]["latent_dim"]
    all_fakes = []
    remaining = n_samples

    while remaining > 0:
        bs = min(batch_size, remaining)
        z = torch.randn(bs, latent_dim, device=device)

        with torch.no_grad():
            if stage == "conditional_gan":
                labels = torch.randint(0, cfg["model"]["n_classes"], (bs,), device=device)
                fakes = generator(z, labels)
            else:
                fakes = generator(z)

        all_fakes.append(fakes.cpu())
        remaining -= bs

    fake_images = torch.cat(all_fakes, dim=0)[:n_samples]
    click.echo(f"Generated {fake_images.shape[0]} images")

    # Collect real images for FID
    if compute_fid:
        click.echo("Collecting real images for FID computation...")
        dataloader = get_dataloader_from_config(cfg)
        real_list = []
        count = 0
        for imgs, _ in dataloader:
            real_list.append(imgs)
            count += imgs.size(0)
            if count >= n_samples:
                break
        real_images = torch.cat(real_list, dim=0)[:n_samples]

    # Compute metrics
    results = {}

    if compute_fid:
        click.echo("\nComputing FID score (this may take a while)...")
        fid_calc = FIDCalculator(device=str(device), batch_size=batch_size)
        fid = fid_calc.compute_fid(real_images, fake_images)
        results["fid"] = fid
        click.echo(f"  FID Score: {fid:.2f}")

    if compute_is:
        click.echo("\nComputing Inception Score...")
        is_calc = InceptionScoreCalculator(device=str(device), batch_size=batch_size)
        is_mean, is_std = is_calc.compute_inception_score(fake_images, splits=10)
        results["inception_score_mean"] = is_mean
        results["inception_score_std"] = is_std
        click.echo(f"  Inception Score: {is_mean:.2f} ± {is_std:.2f}")

    # Summary
    click.echo(f"\n{'='*50}")
    click.echo("  EVALUATION RESULTS")
    click.echo(f"{'='*50}")
    for key, value in results.items():
        click.echo(f"  {key}: {value:.4f}")
    click.echo(f"{'='*50}")


if __name__ == "__main__":
    main()
