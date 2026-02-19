#!/usr/bin/env python3
"""
Pre-download InceptionV3 weights for FID/IS computation.

Run this once before evaluation to avoid download delays during experiments.

Usage:
    python scripts/download_inception.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torchvision import models


def main() -> None:
    print("Downloading InceptionV3 pretrained weights...")
    print("This is a one-time download (~100 MB).\n")

    model = models.inception_v3(
        weights=models.Inception_V3_Weights.DEFAULT,
        transform_input=False,
    )
    model.eval()

    # Verify it works
    dummy_input = torch.randn(1, 3, 299, 299)
    with torch.no_grad():
        output = model(dummy_input)

    if isinstance(output, torch.Tensor):
        print(f"Output shape: {output.shape}")
    else:
        print(f"Output shape: {output.logits.shape}")

    print("\nInceptionV3 weights downloaded and verified successfully!")
    print("You can now run FID and Inception Score evaluations.")


if __name__ == "__main__":
    main()
