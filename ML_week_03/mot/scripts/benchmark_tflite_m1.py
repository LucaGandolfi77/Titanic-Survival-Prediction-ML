"""
benchmark_tflite_m1.py – Measure TFLite inference speed on Apple Silicon M1.

Runs 500 frames of random noise through the TFLite model and prints
latency statistics + throughput (FPS).

Usage
-----
    python scripts/benchmark_tflite_m1.py
    python scripts/benchmark_tflite_m1.py --model models/siamese_tracker.tflite --frames 1000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def benchmark(model_path: Path, n_frames: int = 500) -> None:
    import tensorflow as tf

    if not model_path.exists():
        print(f"ERROR: model not found at {model_path}")
        print("Run `make export` first.")
        return

    interp = tf.lite.Interpreter(model_path=str(model_path), num_threads=4)
    interp.allocate_tensors()
    inp_details = interp.get_input_details()
    out_details = interp.get_output_details()

    print(f"Model : {model_path}")
    print(f"Size  : {model_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"Inputs: {[(d['name'], d['shape'].tolist()) for d in inp_details]}")
    print(f"Frames: {n_frames}\n")

    # Warm up
    for d in inp_details:
        interp.set_tensor(d["index"], np.random.rand(*d["shape"]).astype(np.float32))
    for _ in range(10):
        interp.invoke()

    # Benchmark
    latencies: list[float] = []
    for i in range(n_frames):
        for d in inp_details:
            interp.set_tensor(d["index"], np.random.rand(*d["shape"]).astype(np.float32))
        t0 = time.perf_counter()
        interp.invoke()
        latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    print("─── Results ───────────────────────────────────")
    print(f"  Mean   : {arr.mean():.2f} ms")
    print(f"  Std    : {arr.std():.2f} ms")
    print(f"  Median : {np.median(arr):.2f} ms")
    print(f"  P95    : {np.percentile(arr, 95):.2f} ms")
    print(f"  P99    : {np.percentile(arr, 99):.2f} ms")
    print(f"  FPS    : {1000.0 / arr.mean():.1f}")
    print("────────────────────────────────────────────────")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TFLite model on M1")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "siamese_tracker.tflite")
    parser.add_argument("--frames", type=int, default=500)
    args = parser.parse_args()
    benchmark(args.model, args.frames)


if __name__ == "__main__":
    main()
