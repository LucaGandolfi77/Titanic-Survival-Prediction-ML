#!/usr/bin/env python3
"""
Precompute InceptionV3 feature statistics (mu, sigma) for real datasets.

This allows significantly faster FID evaluation during training,
since real data statistics only need to be computed once.

Usage:
    python scripts/compute_fid_statistics.py --config config/dcgan.yaml
    python scripts/compute_fid_statistics.py --dataset mnist --n-samples 10000
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataloaders import get_dataloader, get_dataloader_from_config
from src.evaluation.fid_score import FIDCalculator
from src.utils.config_loader import load_config


@click.command()
@click.option("--config", "-c", type=click.Path(exists=True), default=None)
@click.option("--dataset", type=str, default="mnist")
@click.option("--image-size", type=int, default=64)
@click.option("--n-samples", "-n", type=int, default=10000)
@click.option("--batch-size", "-b", type=int, default=64)
@click.option("--output", "-o", type=str, default=None)
def main(
    config: str | None,
    dataset: str,
    image_size: int,
    n_samples: int,
    batch_size: int,
    output: str | None,
) -> None:
    """Precompute FID statistics for a real dataset."""
    device = "cpu"

    if config:
        cfg = load_config(config)
        dataloader = get_dataloader_from_config(cfg)
        dataset_name = cfg["data"]["dataset_name"]
        image_size = cfg["data"].get("image_size", 64)
        device = cfg["experiment"]["device"]
    else:
        dataset_name = dataset
        dataloader = get_dataloader(
            dataset_name=dataset_name,
            image_size=image_size,
            batch_size=batch_size,
        )

    click.echo(f"Computing FID statistics for {dataset_name} ({image_size}×{image_size})")
    click.echo(f"Collecting {n_samples} images...")

    # Collect images
    all_images = []
    count = 0
    for imgs, _ in dataloader:
        all_images.append(imgs)
        count += imgs.size(0)
        if count >= n_samples:
            break

    images = torch.cat(all_images, dim=0)[:n_samples]
    click.echo(f"Collected {images.shape[0]} images")

    # Compute and save statistics
    fid_calc = FIDCalculator(device=device, batch_size=batch_size)

    if output is None:
        output_path = Path("data") / f"fid_stats_{dataset_name}_{image_size}.npz"
    else:
        output_path = Path(output)

    click.echo("Extracting InceptionV3 features (this may take a while)...")
    fid_calc.save_statistics(images, output_path)

    click.echo(f"\nStatistics saved to: {output_path}")
    click.echo("Use FIDCalculator.load_statistics() to load for fast FID computation.")


if __name__ == "__main__":
    main()
