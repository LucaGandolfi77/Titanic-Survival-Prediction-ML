"""
export_tflite.py – Convert trained Keras model → optimised TFLite.

Supports FP32, FP16, and INT8 (dynamic-range) quantization.

Usage
-----
    python -m src.training.export_tflite
    python -m src.training.export_tflite --quant fp16
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.training.siamese_model import build_inference_model

ROOT = Path(__file__).resolve().parents[3]  # ML_week_03/mot


def export(
    weights_path: Path | None = None,
    output_path: Path | None = None,
    quantization: str = "fp16",
    template_size: int = 127,
    search_size: int = 255,
) -> Path:
    """Export to TFLite and verify output consistency.

    Parameters
    ----------
    quantization : one of ``"fp32"``, ``"fp16"``, ``"int8"``.
    """
    if weights_path is None:
        weights_path = ROOT / "models" / "siamese_tracker_tf" / "best_weights.h5"
    if output_path is None:
        output_path = ROOT / "models" / "siamese_tracker.tflite"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── build & load ──────────────────────────────────────────────
    model = build_inference_model(
        template_shape=(template_size, template_size, 3),
        search_shape=(search_size, search_size, 3),
        weights_path=weights_path if weights_path.exists() else None,
    )

    # ── concrete function for TFLite converter ────────────────────
    @tf.function(input_signature=[
        tf.TensorSpec([1, template_size, template_size, 3], tf.float32, name="template"),
        tf.TensorSpec([1, search_size, search_size, 3], tf.float32, name="search"),
    ])
    def _serving(template: tf.Tensor, search: tf.Tensor) -> tf.Tensor:
        return model([template, search], training=False)

    concrete = _serving.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete])

    # ── quantization options ──────────────────────────────────────
    if quantization == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Dynamic-range quantization (no representative dataset needed)
    # else: fp32 — no extra flags

    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"TFLite model saved to {output_path}  ({size_mb:.2f} MB, quant={quantization})")

    # ── verification ──────────────────────────────────────────────
    _verify(model, output_path, template_size, search_size)
    return output_path


def _verify(
    keras_model: tf.keras.Model,
    tflite_path: Path,
    template_size: int,
    search_size: int,
    atol: float = 0.05,
) -> None:
    """Quick sanity check: TFLite output ≈ Keras output."""
    t = np.random.rand(1, template_size, template_size, 3).astype(np.float32)
    s = np.random.rand(1, search_size, search_size, 3).astype(np.float32)

    keras_out = keras_model.predict([t, s], verbose=0)

    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp_details = interp.get_input_details()
    out_details = interp.get_output_details()

    for d in inp_details:
        if "template" in d["name"].lower() or d["index"] == inp_details[0]["index"]:
            interp.set_tensor(d["index"], t)
        else:
            interp.set_tensor(d["index"], s)
    interp.invoke()
    tflite_out = interp.get_tensor(out_details[0]["index"])

    diff = np.max(np.abs(keras_out - tflite_out))
    status = "PASS" if diff < atol else "WARN"
    print(f"Verification {status}: max absolute diff = {diff:.6f} (tol={atol})")


# ─────────────────────── CLI ──────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Export Siamese tracker to TFLite")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quant", choices=["fp32", "fp16", "int8"], default="fp16")
    args = parser.parse_args()
    export(args.weights, args.output, args.quant)


if __name__ == "__main__":
    main()
