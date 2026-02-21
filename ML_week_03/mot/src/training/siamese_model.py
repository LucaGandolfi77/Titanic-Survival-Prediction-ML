"""
siamese_model.py – Lightweight Siamese tracker (SiamFC / LightFC style).

Architecture
~~~~~~~~~~~~
* Shared backbone  : 4-block depthwise-separable CNN (MobileNet-flavour)
* Correlation head  : depth-wise cross-correlation → response map
* Output            : single-channel heatmap (score map)
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ─────────────────────────── building blocks ───────────────────────────
def _depthwise_separable_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int = 3,
    stride: int = 1,
    name: str = "",
) -> tf.Tensor:
    """Depthwise-separable conv → BN → ReLU6 (MobileNet-style)."""
    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=stride,
        padding="valid",
        use_bias=False,
        name=f"{name}_dw",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.ReLU(max_value=6.0, name=f"{name}_relu1")(x)

    x = layers.Conv2D(
        filters, 1, strides=1, padding="valid", use_bias=False, name=f"{name}_pw"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.ReLU(max_value=6.0, name=f"{name}_relu2")(x)
    return x


# ────────────────────────── backbone ──────────────────────────────────
def build_siamese_backbone(
    input_shape: Tuple[int | None, int | None, int] = (None, None, 3),
    channels: Tuple[int, ...] = (32, 64, 128, 256),
    name: str = "backbone",
) -> keras.Model:
    """Build 4-block lightweight depthwise-separable backbone.

    The spatial dimensions default to ``None`` so the same backbone
    accepts both the 127×127 template and the 255×255 search patch.

    Returns a ``keras.Model`` that maps an image tensor to a feature map.
    """
    inp = layers.Input(shape=input_shape, name=f"{name}_input")
    x = inp

    # Initial standard conv to get into feature space
    x = layers.Conv2D(
        channels[0], 3, strides=2, padding="valid", use_bias=False, name=f"{name}_stem"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_stem_bn")(x)
    x = layers.ReLU(max_value=6.0, name=f"{name}_stem_relu")(x)

    strides = [1, 2, 1, 1]  # keep spatial dims reasonable
    for i, (ch, s) in enumerate(zip(channels, strides)):
        x = _depthwise_separable_block(x, ch, kernel_size=3, stride=s, name=f"{name}_b{i}")

    return keras.Model(inputs=inp, outputs=x, name=name)


# ──────────────── cross-correlation head ──────────────────────────────
class CrossCorrelation(layers.Layer):
    """Depth-wise cross-correlation between template and search feature maps.

    Uses ``tf.vectorized_map`` so it works inside Keras symbolic graph builds.
    """

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:  # type: ignore[override]
        template_feat, search_feat = inputs
        # template_feat: (B, Ht, Wt, C)  –  kernel
        # search_feat  : (B, Hs, Ws, C)  –  input

        def _single_xcorr(pair: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
            t, s = pair  # (Ht, Wt, C), (Hs, Ws, C)
            kernel = tf.expand_dims(t, axis=-1)       # (Ht, Wt, C, 1)
            feat = tf.expand_dims(s, axis=0)           # (1, Hs, Ws, C)
            resp = tf.nn.depthwise_conv2d(
                feat, kernel, strides=[1, 1, 1, 1], padding="VALID",
            )
            # resp: (1, H_out, W_out, C) → mean over channels
            return tf.reduce_mean(resp, axis=-1, keepdims=True)[0]  # (H, W, 1)

        return tf.vectorized_map(_single_xcorr, (template_feat, search_feat))

    def compute_output_shape(self, input_shape):  # type: ignore[override]
        t_shape, s_shape = input_shape   # [(B, Ht, Wt, C), (B, Hs, Ws, C)]
        if t_shape[1] is not None and s_shape[1] is not None:
            out_h = s_shape[1] - t_shape[1] + 1
            out_w = s_shape[2] - t_shape[2] + 1
        else:
            out_h = out_w = None
        return (t_shape[0], out_h, out_w, 1)


# ──────────────── full tracker model ──────────────────────────────────
def build_siamese_tracker_model(
    template_shape: Tuple[int, int, int] = (127, 127, 3),
    search_shape: Tuple[int, int, int] = (255, 255, 3),
    channels: Tuple[int, ...] = (32, 64, 128, 256),
) -> keras.Model:
    """Siamese tracker: shared backbone + cross-correlation head.

    Returns
    -------
    keras.Model with two inputs (*template*, *search*) and output *response_map*.
    """
    # Backbone accepts any spatial size (None, None, 3) — shared weights
    backbone = build_siamese_backbone((None, None, 3), channels, name="backbone")

    template_input = layers.Input(shape=template_shape, name="template")
    search_input = layers.Input(shape=search_shape, name="search")

    template_feat = backbone(template_input)
    search_feat = backbone(search_input)

    response_map = CrossCorrelation(name="xcorr")([template_feat, search_feat])

    # Learnable bias + scale (like SiamFC)
    response_map = layers.Conv2D(
        1, 1, padding="same", use_bias=True, name="head_adjust"
    )(response_map)

    model = keras.Model(
        inputs=[template_input, search_input],
        outputs=response_map,
        name="siamese_tracker",
    )
    return model


# ─────────────── convenience for export ──────────────────────────────
def build_inference_model(
    template_shape: Tuple[int, int, int] = (127, 127, 3),
    search_shape: Tuple[int, int, int] = (255, 255, 3),
    weights_path: Path | str | None = None,
    channels: Tuple[int, ...] = (32, 64, 128, 256),
) -> keras.Model:
    """Load (or build) the tracker model ready for TFLite conversion."""
    model = build_siamese_tracker_model(template_shape, search_shape, channels)
    if weights_path is not None:
        model.load_weights(str(weights_path))
    return model


# ──────────────── quick sanity test ──────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    m = build_siamese_tracker_model()
    m.summary()
    t = np.random.rand(2, 127, 127, 3).astype(np.float32)
    s = np.random.rand(2, 255, 255, 3).astype(np.float32)
    out = m.predict([t, s])
    print(f"Response map shape: {out.shape}")  # expect (2, H, W, 1)
