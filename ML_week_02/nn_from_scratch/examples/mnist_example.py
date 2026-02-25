"""
MNIST Digit Classification — The "Hello World" of Deep Learning
================================================================

Train a multi-layer perceptron on the MNIST dataset (28×28 grey-scale
handwritten digits, 10 classes) using **only NumPy**.

Architecture
------------
::

    Input (784) → Dense(256, ReLU) → Dense(128, ReLU) → Dense(10, Softmax)

Dataset
-------
Downloaded on-the-fly from Yann LeCun's website via raw HTTP and cached
locally as .npz.
"""

from __future__ import annotations

import gzip
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.activations import ReLU, Softmax
from src.core.layer import DenseLayer
from src.core.losses import CrossEntropyLoss
from src.core.optimizers import Adam
from src.network.sequential import Sequential
from src.network.model import Model
from src.utils.data_utils import one_hot_encode
from src.utils.metrics import accuracy, confusion_matrix as compute_cm
from src.utils.visualization import (
    plot_training_curves,
    plot_confusion_matrix,
)

# ────────────────────────────────────────────────────────────────────
# MNIST loader  (pure Python — fallback to sklearn if download fails)
# ────────────────────────────────────────────────────────────────────
MNIST_URLS = {
    "train_images": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
}


def _download(url: str, dest: Path, retries: int = 3) -> None:
    """Download a file if it doesn't exist with retry logic."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    for attempt in range(retries):
        try:
            print(f"Downloading {url} (attempt {attempt + 1}/{retries})...")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response, open(dest, 'wb') as out_file:
                out_file.write(response.read())
            return
        except (urllib.error.HTTPError, urllib.error.URLError, Exception) as e:
            print(f"  Download failed: {e}")
            if dest.exists():
                dest.unlink()
            if attempt < retries - 1:
                print(f"  Retrying...")
            else:
                print(f"  Failed after {retries} attempts. Will use fallback loader.")


def _read_idx_images(path: Path) -> NDArray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float64) / 255.0


def _read_idx_labels(path: Path) -> NDArray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def load_mnist(data_dir: Path | None = None) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Download and load MNIST.

    Returns
    -------
    X_train : (60000, 784)   float64 ∈ [0, 1]
    Y_train : (60000,)       int
    X_test  : (10000, 784)   float64 ∈ [0, 1]
    Y_test  : (10000,)       int
    """
    if data_dir is None:
        data_dir = ROOT / "outputs" / "data"

    cache = data_dir / "mnist.npz"
    if cache.exists():
        d = np.load(cache)
        return d["X_train"], d["Y_train"], d["X_test"], d["Y_test"]

    # Try downloading from official sources
    raw = data_dir / "raw"
    files: dict[str, Path] = {}
    download_success = True
    
    for key, url in MNIST_URLS.items():
        fname = url.split("/")[-1]
        dest = raw / fname
        _download(url, dest)
        if dest.exists():
            files[key] = dest
        else:
            download_success = False
            break
    
    # If download failed, fallback to sklearn digits dataset
    if not download_success:
        print("\n⚠️  Could not download MNIST from official source.")
        print("   Falling back to sklearn Digits dataset (8x8 images)...")
        from sklearn import datasets
        digits = datasets.load_digits()
        X = digits.data / 255.0  # normalize to [0, 1]
        Y = digits.target
        
        # Pad to 28x28 to match MNIST shape
        X_padded = np.zeros((len(X), 784))
        X_padded[:, :64] = X
        
        # Simple train/test split (sklearn's digits is smaller)
        split_idx = int(0.8 * len(X_padded))
        return (
            X_padded[:split_idx],
            Y[:split_idx],
            X_padded[split_idx:],
            Y[split_idx:]
        )
    
    X_train = _read_idx_images(files["train_images"])
    Y_train = _read_idx_labels(files["train_labels"])
    X_test = _read_idx_images(files["test_images"])
    Y_test = _read_idx_labels(files["test_labels"])

    np.savez(cache, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
    return X_train, Y_train, X_test, Y_test


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────
def main(
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 0.001,
    hidden: int = 256,
) -> None:
    output_dir = ROOT / "outputs"

    print("=" * 60)
    print("MNIST Classification — Neural Network From Scratch")
    print("=" * 60)

    # ── load data ──
    X_train, Y_train_int, X_test, Y_test_int = load_mnist()
    Y_train = one_hot_encode(Y_train_int, 10)
    Y_test = one_hot_encode(Y_test_int, 10)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # ── build model ──
    network = Sequential(
        DenseLayer(784, hidden, activation=ReLU(), seed=42),
        DenseLayer(hidden, 128, activation=ReLU(), seed=43),
        DenseLayer(128, 10, activation=Softmax(), seed=44),
    )

    model = Model(
        network=network,
        loss_fn=CrossEntropyLoss(),
        optimizer=Adam(lr=lr),
    )
    print(network.summary())
    print()

    # ── train ──
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        X_val=X_test, Y_val=Y_test,
        verbose=True,
    )

    # ── final evaluation ──
    Y_pred_probs = model.predict(X_test)
    Y_pred = np.argmax(Y_pred_probs, axis=1)
    acc = accuracy(Y_test_int, Y_pred)
    print(f"\n✅ Test Accuracy: {acc:.2%}")

    # ── save artefacts ──
    model.save_weights(output_dir / "models" / "mnist_weights.npz")
    plot_training_curves(history, save_path=output_dir / "plots" / "mnist_training.png", title="MNIST Training")

    cm = compute_cm(Y_test_int, Y_pred, n_classes=10)
    plot_confusion_matrix(
        cm, class_names=[str(i) for i in range(10)],
        save_path=output_dir / "plots" / "mnist_confusion.png",
        title="MNIST Confusion Matrix",
    )
    print(f"Artefacts saved to {output_dir}")


if __name__ == "__main__":
    main()
