"""
convert_to_coreml.py – (Optional) Convert TFLite model to CoreML for iOS.

Requires ``coremltools`` (pip install coremltools).

Usage
-----
    python scripts/convert_to_coreml.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def convert(
    tflite_path: Path | None = None,
    output_path: Path | None = None,
) -> None:
    try:
        import coremltools as ct
    except ImportError:
        print("coremltools not installed. Run: pip install coremltools")
        return

    if tflite_path is None:
        tflite_path = ROOT / "models" / "siamese_tracker.tflite"
    if output_path is None:
        output_path = ROOT / "models" / "siamese_tracker.mlmodel"

    if not tflite_path.exists():
        print(f"TFLite model not found: {tflite_path}")
        return

    print(f"Converting {tflite_path} → CoreML …")
    model = ct.convert(
        str(tflite_path),
        source="tensorflow",
        convert_to="mlprogram",
    )
    model.save(str(output_path))
    print(f"CoreML model saved to {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="TFLite → CoreML conversion")
    parser.add_argument("--tflite", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    convert(args.tflite, args.output)


if __name__ == "__main__":
    main()
