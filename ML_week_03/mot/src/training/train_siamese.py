"""
train_siamese.py – Main training script for the Siamese tracker.

Usage
-----
    python -m src.training.train_siamese                 # defaults
    python -m src.training.train_siamese --config configs/training_config.yaml
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

from src.training.dataset import build_tf_dataset
from src.training.losses import balanced_logistic_loss
from src.training.siamese_model import build_siamese_tracker_model

ROOT = Path(__file__).resolve().parents[3]  # ML_week_03/mot


def _cosine_lr(epoch: int, total: int, base_lr: float, warmup: int) -> float:
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(cfg_path: Path | None = None) -> None:
    # ── load config ───────────────────────────────────────────────
    if cfg_path is None:
        cfg_path = ROOT / "configs" / "training_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    tcfg = cfg["training"]
    dcfg = cfg["data"]

    epochs: int = tcfg["epochs"]
    batch_size: int = tcfg["batch_size"]
    base_lr: float = tcfg["learning_rate"]
    warmup: int = tcfg["warmup_epochs"]
    patience: int = tcfg.get("early_stopping_patience", 7)
    ckpt_dir = ROOT / tcfg.get("checkpoint_dir", "models/siamese_tracker_tf")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    template_size: int = dcfg["template_size"]
    search_size: int = dcfg["search_size"]

    # ── dataset ───────────────────────────────────────────────────
    data_dir = ROOT / "data"
    ds = build_tf_dataset(
        data_dir,
        template_size=template_size,
        search_size=search_size,
        max_pairs_per_video=dcfg["max_pairs_per_video"],
        augmentations=dcfg.get("augmentations"),
        batch_size=batch_size,
    )
    val_ds = ds.take(2)  # quick eval split (for real projects split properly)
    train_ds = ds.skip(2)

    # ── model ─────────────────────────────────────────────────────
    model = build_siamese_tracker_model(
        template_shape=(template_size, template_size, 3),
        search_shape=(search_size, search_size, 3),
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr, decay=tcfg["weight_decay"])

    # ── TensorBoard ───────────────────────────────────────────────
    log_dir = ROOT / "logs" / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(str(log_dir))

    # ── training loop ─────────────────────────────────────────────
    best_val_loss = float("inf")
    wait = 0

    for epoch in range(epochs):
        lr = _cosine_lr(epoch, epochs, base_lr, warmup)
        optimizer.learning_rate.assign(lr)

        epoch_loss: list[float] = []
        for (template_batch, search_batch), label_batch in train_ds:
            with tf.GradientTape() as tape:
                pred = model([template_batch, search_batch], training=True)
                # Resize label to match pred spatial dims
                pred_h, pred_w = pred.shape[1], pred.shape[2]
                label_resized = tf.image.resize(label_batch, (pred_h, pred_w))
                loss = balanced_logistic_loss(label_resized, pred)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss.append(float(loss))

        avg_train = float(np.mean(epoch_loss)) if epoch_loss else 0.0

        # ── validation ────────────────────────────────────────────
        val_losses: list[float] = []
        for (vt, vs), vl in val_ds:
            vp = model([vt, vs], training=False)
            pred_h, pred_w = vp.shape[1], vp.shape[2]
            vl_resized = tf.image.resize(vl, (pred_h, pred_w))
            val_losses.append(float(balanced_logistic_loss(vl_resized, vp)))
        avg_val = float(np.mean(val_losses)) if val_losses else avg_train

        # ── logging ───────────────────────────────────────────────
        with summary_writer.as_default():
            tf.summary.scalar("loss/train", avg_train, step=epoch)
            tf.summary.scalar("loss/val", avg_val, step=epoch)
            tf.summary.scalar("lr", lr, step=epoch)

        print(f"Epoch {epoch + 1:03d}/{epochs}  lr={lr:.6f}  "
              f"train_loss={avg_train:.5f}  val_loss={avg_val:.5f}")

        # ── early stopping + checkpoint ───────────────────────────
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            model.save_weights(str(ckpt_dir / "best_weights.h5"))
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    model.save_weights(str(ckpt_dir / "final_weights.h5"))
    print(f"Training complete. Best val loss: {best_val_loss:.5f}")
    print(f"Weights saved to {ckpt_dir}")


# ─────────────────────── CLI entry point ──────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Train Siamese tracker")
    parser.add_argument("--config", type=Path, default=None, help="Path to training YAML")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
