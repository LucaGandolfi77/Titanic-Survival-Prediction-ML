# GAN Improved — From Vanilla to Conditional GAN

Production-grade GAN implementations demonstrating the full evolution from basic MLP-based generation to class-conditioned convolutional architectures.

## Project Overview

| Stage | Architecture | Key Innovation | Expected FID (MNIST) |
|-------|-------------|----------------|---------------------|
| **Stage 1** | Vanilla GAN (MLP) | Baseline adversarial training | ~150–200 |
| **Stage 2** | DCGAN | Convolutional backbone, spatial hierarchy | ~50–100 |
| **Stage 3** | Conditional GAN | Class-conditioned generation, Projection D | ~30–80 |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train Stage 1 (Vanilla GAN)
python train.py --config config/vanilla_gan.yaml --epochs 50

# Train Stage 2 (DCGAN)
python train.py --config config/dcgan.yaml

# Train Stage 3 (Conditional GAN)
python train.py --config config/conditional_gan.yaml

# Evaluate a model
python evaluate.py --config config/dcgan.yaml \
    --checkpoint outputs/models/dcgan/checkpoint_final.pt

# Generate images
python generate.py --config config/dcgan.yaml \
    --checkpoint outputs/models/dcgan/checkpoint_final.pt --n 64

# Generate class-conditioned grid (Stage 3)
python generate.py --config config/conditional_gan.yaml \
    --checkpoint outputs/models/conditional_gan/checkpoint_final.pt \
    --all-classes

# Latent space interpolation
python generate.py --config config/dcgan.yaml \
    --checkpoint outputs/models/dcgan/checkpoint_final.pt \
    --interpolate --method slerp --n-pairs 5
```

## Troubleshooting

### OpenMP runtime error

On macOS you may encounter the following message when launching training (or other scripts):

```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

This happens when multiple libraries bring their own copy of the OpenMP runtime. Two safe ways to work around it:

1. **Set an environment variable** before running the command:
   ```bash
   export KMP_DUPLICATE_LIB_OK=TRUE
   python train.py --config config/dcgan.yaml
   ```
   This tells the runtime to ignore the duplicate, but use at your own risk.

2. **Apply it programmatically** at the top of `train.py` (or other entry point) so it is always set:
   ```python
   import os
   os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
   ```

The root cause is usually static linking of `libomp` in some third‑party wheel; the variable simply allows execution to continue. For production or high‑performance runs consider rebuilding problematic libraries without the static runtime.

If you continue to see aborts or crashes, uninstall one of the conflicting packages (e.g. `pip uninstall -y mkl-service`) or install a single OpenMP runtime via `brew install libomp` and ensure only one copy is on your `DYLD_LIBRARY_PATH`.


## Project Structure

```
GAN_improved/
├── config/                          # YAML experiment configs
│   ├── vanilla_gan.yaml
│   ├── dcgan.yaml
│   └── conditional_gan.yaml
├── src/
│   ├── models/                      # Model architectures
│   │   ├── vanilla_gan.py           # MLP Generator + Discriminator
│   │   ├── dcgan.py                 # Conv Generator + Discriminator
│   │   ├── conditional_gan.py       # cGAN + Projection Discriminator
│   │   └── layers.py                # SpectralNorm, SelfAttention, etc.
│   ├── training/                    # Training loops
│   │   ├── trainer.py               # Abstract base trainer
│   │   ├── vanilla_trainer.py       # Stage 1 trainer
│   │   ├── dcgan_trainer.py         # Stage 2 trainer
│   │   └── conditional_trainer.py   # Stage 3 trainer
│   ├── evaluation/                  # Metrics & visualization
│   │   ├── fid_score.py             # FID with InceptionV3
│   │   ├── inception_score.py       # Inception Score
│   │   └── visualization.py         # Grids, interpolation, plots
│   ├── data/
│   │   └── dataloaders.py           # Dataset wrappers (MNIST, F-MNIST, CIFAR)
│   └── utils/
│       ├── config_loader.py         # YAML config + device detection
│       ├── logger.py                # TensorBoard + file logging
│       └── checkpointing.py         # Save/load/resume training
├── notebooks/                       # Analysis notebooks
│   ├── 01_vanilla_gan_analysis.ipynb
│   ├── 02_dcgan_analysis.ipynb
│   └── 03_conditional_generation_demo.ipynb
├── tests/                           # Pytest unit tests
├── scripts/                         # Utility scripts
├── train.py                         # CLI: training
├── evaluate.py                      # CLI: FID & IS evaluation
├── generate.py                      # CLI: image generation
├── Makefile                         # Convenience commands
├── Dockerfile
└── requirements.txt
```

## Architecture Details

### Stage 1: Vanilla GAN (MLP)
- **Generator**: `z(100) → 256 → 512 → 1024 → 784` with BatchNorm + LeakyReLU
- **Discriminator**: `784 → 512 → 256 → 1` with Spectral Norm + Dropout
- **Loss**: BCE with logits (no Sigmoid for numerical stability)

### Stage 2: DCGAN
Follows [Radford et al. (2016)](https://arxiv.org/abs/1511.06434) guidelines:
- Replace pooling → strided convolutions
- BatchNorm in G (not D when using SpectralNorm)
- ReLU in G, LeakyReLU(0.2) in D
- **Generator**: `z → 4×4 → 8×8 → 16×16 → 32×32 → 64×64`
- **Discriminator**: mirror of G with stride-2 convolutions

### Stage 3: Conditional GAN
- **Generator**: Class embedding concatenated with z before deconvolution
- **Discriminator**: Projection discriminator ([Miyato & Koyama, 2018](https://arxiv.org/abs/1802.05637))
  - `output = linear(features) + ⟨embed(class), features⟩`
- Enables generating any specific digit class on demand

## Training Stability Techniques

| Technique | Where | Effect |
|-----------|-------|--------|
| Spectral Normalization | Discriminator | Lipschitz constraint → stable gradients |
| Label Smoothing | Labels | Real: 1.0→0.9, prevents overconfident D |
| Label Noise | Labels | Small random noise on labels |
| BCEWithLogits | Loss | Numerically stable (no Sigmoid) |
| Weight Init N(0, 0.02) | Both | DCGAN standard initialization |
| Gradient Penalty (opt.) | Discriminator | WGAN-GP regularization |

## Evaluation Metrics

### FID (Fréchet Inception Distance)
$$FID = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$$

Lower FID = better quality + diversity. Computed using InceptionV3 pool3 features (2048-d).

### Inception Score
$$IS = \exp(\mathbb{E}_x[KL(p(y|x) \| p(y))])$$

Higher IS = sharper images + diverse classes.

## Monitoring

```bash
# Launch TensorBoard
make tensorboard
# or
tensorboard --logdir outputs/tensorboard --port 6006
```

Logged metrics:
- Generator / Discriminator loss (per step + per epoch)
- Sample image grids
- Gradient histograms
- FID scores over training

## Testing

```bash
make test          # Run all tests
make test-models   # Model architecture tests only
make test-cov      # With coverage report
```

## Device Support

| Device | Status | Notes |
|--------|--------|-------|
| Apple MPS | ✅ Primary target | `num_workers=0` required |
| NVIDIA CUDA | ✅ Supported | Full feature support |
| CPU | ✅ Fallback | Slow but functional |

Auto-detection priority: MPS → CUDA → CPU (set `device: "auto"` in config).

## License

This project is for educational and portfolio purposes.
