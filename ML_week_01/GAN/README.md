# 🧠 Vanilla GAN — MNIST (Apple Silicon)

A from-scratch **Generative Adversarial Network** trained on MNIST using
PyTorch with the **MPS** (Metal Performance Shaders) backend for native
Apple Silicon acceleration.

---

## Project structure

```
GAN/
├── train.py          # main training loop
├── models.py         # Generator & Discriminator (MLP + CNN variant)
├── utils.py          # image grid saving, model persistence
├── config.py         # all hyperparameters & paths
├── requirements.txt  # pinned deps for macOS ARM
└── README.md         # ← you are here
```

## Quick start

### 1 · Create environment

**Option A — conda**

```bash
conda create -n gan python=3.11 -y
conda activate gan
pip install -r requirements.txt
```

**Option B — venv**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2 · Train

```bash
python train.py
```

Training auto-detects **MPS** on Apple Silicon and falls back to CPU.

### 3 · Outputs

After training completes you will find:

| Path | Description |
|------|-------------|
| `outputs/samples/epoch_XXXX.png` | Grid of generated digits saved every 5 epochs |
| `outputs/weights/generator.pth` | Generator state dict |
| `outputs/weights/discriminator.pth` | Discriminator state dict |

## Hyperparameters

Edit [`config.py`](config.py) to change:

| Constant | Default | Notes |
|----------|---------|-------|
| `EPOCHS` | 50 | Total training epochs |
| `BATCH_SIZE` | 64 | Mini-batch size |
| `LEARNING_RATE` | 0.0002 | Adam LR for both G and D |
| `BETAS` | (0.5, 0.999) | Adam momentum |
| `LATENT_DIM` | 100 | Noise vector size |
| `SAMPLE_INTERVAL` | 5 | Save grid every N epochs |

## Architecture

### Generator (MLP)

```
z (100) → 256 (BN+ReLU) → 512 (BN+ReLU) → 1024 (BN+ReLU) → 784 (Tanh)
```

### Discriminator (MLP)

```
784 → 512 (LeakyReLU+Dropout) → 256 (LeakyReLU+Dropout) → 1 (Sigmoid)
```

A **CNN** variant is included in `models.py` as commented code.

## License

MIT — for educational purposes.
