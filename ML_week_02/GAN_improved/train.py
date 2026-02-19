#!/usr/bin/env python3
"""
CLI training entry point for GAN stages.

Usage:
    python train.py --config config/vanilla_gan.yaml
    python train.py --config config/dcgan.yaml --resume
    python train.py --config config/conditional_gan.yaml --epochs 200
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloaders import get_dataloader_from_config
from src.training import ConditionalGANTrainer, DCGANTrainer, VanillaGANTrainer
from src.utils.config_loader import load_config

# Registry of trainers by stage name
_TRAINER_REGISTRY = {
    "vanilla_gan": VanillaGANTrainer,
    "dcgan": DCGANTrainer,
    "conditional_gan": ConditionalGANTrainer,
}


@click.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to YAML configuration file.",
)
@click.option(
    "--resume / --no-resume",
    default=False,
    help="Resume training from the latest checkpoint.",
)
@click.option(
    "--epochs", "-e",
    type=int,
    default=None,
    help="Override number of training epochs.",
)
@click.option(
    "--batch-size", "-b",
    type=int,
    default=None,
    help="Override batch size.",
)
@click.option(
    "--device", "-d",
    type=click.Choice(["auto", "cpu", "cuda", "mps"]),
    default=None,
    help="Override compute device.",
)
@click.option(
    "--seed", "-s",
    type=int,
    default=None,
    help="Override random seed.",
)
def main(
    config: str,
    resume: bool,
    epochs: int | None,
    batch_size: int | None,
    device: str | None,
    seed: int | None,
) -> None:
    """Train a GAN model from a YAML configuration file."""
    # Load config
    cfg = load_config(config)

    # Apply CLI overrides
    if epochs is not None:
        cfg["training"]["n_epochs"] = epochs
    if batch_size is not None:
        cfg["data"]["batch_size"] = batch_size
    if device is not None:
        cfg["experiment"]["device"] = device
        if device == "auto":
            from src.utils.config_loader import _detect_device
            cfg["experiment"]["device"] = _detect_device()
    if seed is not None:
        cfg["experiment"]["seed"] = seed

    # Set random seed
    seed_val = cfg["experiment"].get("seed", 42)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)

    # Get trainer class
    stage = cfg["experiment"]["stage"]
    if stage not in _TRAINER_REGISTRY:
        supported = ", ".join(_TRAINER_REGISTRY.keys())
        raise ValueError(f"Unknown stage '{stage}'. Supported: {supported}")

    trainer_cls = _TRAINER_REGISTRY[stage]

    # Build dataloader
    click.echo(f"Loading dataset: {cfg['data']['dataset_name']}")
    dataloader = get_dataloader_from_config(cfg)
    click.echo(f"Dataset size: {len(dataloader.dataset):,} images")
    click.echo(f"Batch size: {cfg['data']['batch_size']}, batches: {len(dataloader)}")

    # Create trainer and run
    click.echo(f"\n{'='*60}")
    click.echo(f"  Stage: {stage}")
    click.echo(f"  Device: {cfg['experiment']['device']}")
    click.echo(f"  Epochs: {cfg['training']['n_epochs']}")
    click.echo(f"  Resume: {resume}")
    click.echo(f"{'='*60}\n")

    trainer = trainer_cls(cfg)
    results = trainer.train(dataloader, resume=resume)

    click.echo(f"\nTraining complete!")
    click.echo(f"  Total steps: {results['total_steps']:,}")
    click.echo(f"  Time: {results['training_time_minutes']:.1f} minutes")
    if results.get("fid_scores"):
        last_fid = results["fid_scores"][-1][1]
        click.echo(f"  Final FID: {last_fid:.2f}")


if __name__ == "__main__":
    main()
