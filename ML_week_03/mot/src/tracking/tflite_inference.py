"""
tflite_inference.py – TFLite interpreter wrapper for the Siamese tracker.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import tensorflow as tf  # full TF (dev machine)
    _Interpreter = tf.lite.Interpreter
except Exception:
    from tflite_runtime.interpreter import Interpreter as _Interpreter  # type: ignore[assignment]


class TFLiteTracker:
    """Thin wrapper around the TFLite interpreter for a single forward pass.

    Parameters
    ----------
    model_path : path to ``.tflite`` file.
    use_xnnpack : enable XNNPACK delegate for best M1 / mobile perf.
    num_threads : number of CPU threads to use.
    """

    def __init__(
        self,
        model_path: Path | str,
        use_xnnpack: bool = True,
        num_threads: int = 4,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"TFLite model not found: {model_path}")

        # Build experimental delegates list
        delegates = None
        if use_xnnpack:
            try:
                xnn = tf.lite.experimental.load_delegate("libXNNPACK.dylib")  # type: ignore[attr-defined]
                delegates = [xnn]
            except Exception:
                delegates = None  # XNNPACK auto-enabled by default in recent TF

        self._interpreter = _Interpreter(
            model_path=str(model_path),
            num_threads=num_threads,
            experimental_delegates=delegates,
        )
        self._interpreter.allocate_tensors()

        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        # Identify which input is template vs search by shape
        self._template_idx: int = -1
        self._search_idx: int = -1
        for d in self._input_details:
            h = d["shape"][1]
            if h <= 127:
                self._template_idx = d["index"]
            else:
                self._search_idx = d["index"]
        # Fallback: first = template, second = search
        if self._template_idx < 0:
            self._template_idx = self._input_details[0]["index"]
            self._search_idx = self._input_details[1]["index"]

    def infer(
        self,
        template: np.ndarray,
        search: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Run inference.

        Parameters
        ----------
        template : (1, Ht, Wt, 3) float32 [0, 1]
        search   : (1, Hs, Ws, 3) float32 [0, 1]

        Returns
        -------
        response_map : (H, W) float32
        elapsed_ms   : inference time in milliseconds
        """
        self._interpreter.set_tensor(self._template_idx, template)
        self._interpreter.set_tensor(self._search_idx, search)

        t0 = time.perf_counter()
        self._interpreter.invoke()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        raw = self._interpreter.get_tensor(self._output_details[0]["index"])
        response_map = raw[0, :, :, 0]  # (H, W)
        return response_map, elapsed_ms
