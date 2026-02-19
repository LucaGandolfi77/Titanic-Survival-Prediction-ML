#!/usr/bin/env python3
"""
benchmark.py – Measure inference speed across model sizes and formats.

Reports FPS, latency (ms), and memory usage for each configuration.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import sys
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.utils.config import project_root
from src.utils.logging import get_logger

logger = get_logger(__name__)


def benchmark_model(
    weights: str,
    device: str = "cpu",
    imgsz: int = 640,
    n_warmup: int = 5,
    n_iterations: int = 50,
) -> dict:
    """Benchmark a single model configuration."""
    from src.inference.predictor import DefectPredictor

    predictor = DefectPredictor(weights=weights, device=device, imgsz=imgsz)

    # Generate random image
    image = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    for _ in range(n_warmup):
        predictor.predict_image(image)

    # Benchmark
    times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        predictor.predict_image(image)
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    return {
        "weights": weights,
        "device": device,
        "imgsz": imgsz,
        "mean_ms": round(float(times.mean()), 2),
        "std_ms": round(float(times.std()), 2),
        "min_ms": round(float(times.min()), 2),
        "max_ms": round(float(times.max()), 2),
        "fps": round(1000.0 / times.mean(), 1),
        "p50_ms": round(float(np.percentile(times, 50)), 2),
        "p95_ms": round(float(np.percentile(times, 95)), 2),
        "p99_ms": round(float(np.percentile(times, 99)), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark defect detection models")
    parser.add_argument("--device", default="cpu", help="Device (cpu, mps, cuda:0)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--iterations", type=int, default=50, help="Number of iterations")
    parser.add_argument("--models", nargs="+", default=["yolov8n.pt", "yolov8s.pt"],
                        help="Model weights to benchmark")
    args = parser.parse_args()

    print("=" * 70)
    print("  Defect Detection — Inference Benchmark")
    print("=" * 70)

    results = []
    for model in args.models:
        print(f"\nBenchmarking: {model} on {args.device} @ {args.imgsz}px")
        try:
            r = benchmark_model(model, args.device, args.imgsz, n_iterations=args.iterations)
            results.append(r)
            print(f"  Mean: {r['mean_ms']:.1f} ms  |  FPS: {r['fps']:.1f}  |  "
                  f"P95: {r['p95_ms']:.1f} ms  |  P99: {r['p99_ms']:.1f} ms")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"{'Model':<20} {'Mean (ms)':<12} {'FPS':<10} {'P95 (ms)':<12}")
    print("-" * 60)
    for r in results:
        print(f"{r['weights']:<20} {r['mean_ms']:<12.1f} {r['fps']:<10.1f} {r['p95_ms']:<12.1f}")


if __name__ == "__main__":
    main()
