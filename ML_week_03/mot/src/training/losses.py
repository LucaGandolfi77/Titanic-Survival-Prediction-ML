"""
losses.py – Loss functions for Siamese tracker training.

* Logistic loss  (binary cross-entropy on response map)
* Balanced logistic loss (weight positive/negative pixels)
"""
from __future__ import annotations

import tensorflow as tf


def logistic_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Standard binary cross-entropy (logistic) loss on response map.

    Parameters
    ----------
    y_true : (B, H, W, 1) Gaussian ground-truth in [0, 1].
    y_pred : (B, H, W, 1) raw logits from the model.
    """
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    )


def balanced_logistic_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Class-balanced logistic loss — up-weights the (rare) positive region.

    Positive pixels are those with y_true > 0.5.
    """
    pos_mask = tf.cast(y_true > 0.5, tf.float32)
    neg_mask = 1.0 - pos_mask

    n_pos = tf.maximum(tf.reduce_sum(pos_mask), 1.0)
    n_neg = tf.maximum(tf.reduce_sum(neg_mask), 1.0)

    weight = pos_mask * (n_neg / (n_pos + n_neg)) + neg_mask * (n_pos / (n_pos + n_neg))

    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return tf.reduce_mean(bce * weight)
