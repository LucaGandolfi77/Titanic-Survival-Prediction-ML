#!/usr/bin/env python3
"""
CLI Entry Point — Train MNIST Classifier
=========================================

Usage examples::

    python train_mnist.py
    python train_mnist.py --epochs 20 --lr 0.001 --batch-size 64 --hidden 512
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from examples.mnist_example import main as mnist_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a neural network on MNIST from scratch (NumPy only).",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--hidden", type=int, default=256, help="Hidden layer size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mnist_main(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
    )


if __name__ == "__main__":
    main()
