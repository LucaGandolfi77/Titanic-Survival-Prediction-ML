# Neural Architecture Search (NAS) from Scratch

Evolutionary algorithm-based NAS that automatically discovers optimal CNN
architectures for CIFAR-10 classification.

## Overview

| Component | Description |
|-----------|-------------|
| **Search space** | Conv2D, MaxPool, AvgPool, BatchNorm, Dropout, Dense; skip connections |
| **Algorithm** | Evolutionary: tournament selection → crossover → mutation |
| **Fitness** | Validation accuracy after *N* training epochs |
| **Advanced** | Weight inheritance, PNAS-style predictor network |

## Quick Start

```bash
# install
pip install -e ".[dev]"

# run NAS (full — ~24 h on M1)
python scripts/run_search.py

# quick smoke test (< 5 min)
python scripts/run_search.py --config configs/fast.yaml

# train best architecture found
python scripts/train_best.py outputs/best_genome.json --epochs 100

# compare vs ResNet-18
python scripts/compare_resnet.py outputs/best_genome.json

# visualise results
python scripts/visualize.py outputs/
```

## Project Structure

```
nas/
├── configs/            # YAML configuration files
├── src/                # Core library
│   ├── genome.py       # Genome encoding (layer genes)
│   ├── search_space.py # Valid layer types & parameter ranges
│   ├── builder.py      # Genome → PyTorch nn.Module
│   ├── trainer.py      # Train one architecture (early-stop)
│   ├── evolution.py    # Selection, crossover, mutation
│   ├── fitness.py      # Parallel fitness evaluation
│   ├── weight_inherit.py # Copy shared weights parent→child
│   ├── predictor.py    # Meta-model for architecture perf.
│   ├── visualization.py # Fitness curves, evolution tree
│   └── utils.py        # Config, logging, serialisation
├── scripts/            # Entry-point scripts
├── tests/              # Pytest suite
├── notebooks/          # Analysis notebook
├── outputs/            # Generated artifacts
└── data/               # CIFAR-10 (auto-downloaded)
```

## Algorithm Details

1. **Initialisation** – create *P* random genomes within the search space.
2. **Evaluation** – train each genome for *E* epochs; record validation
   accuracy as fitness.  Architectures below an accuracy threshold at epoch 3
   are killed early.
3. **Selection** – tournament selection picks the top 25 % of the population.
4. **Crossover** – single-point swap of layer segments between two parents.
5. **Mutation** – add/remove layer, change hyperparameter, toggle skip
   connection.
6. Repeat for *G* generations.

### Advanced Features

* **Weight inheritance**: child networks inherit trained weights from parent
  layers that are unchanged, dramatically speeding up convergence.
* **Predictor network (PNAS)**: an LSTM meta-model learns to predict
  architecture accuracy from the genome encoding, enabling pre-screening of
  candidates before expensive GPU training.

## Configuration

See `configs/default.yaml` for all tuneable knobs.  Override any value
via CLI: `python scripts/run_search.py --population 30 --generations 100`.

## Requirements

* Python ≥ 3.10
* PyTorch ≥ 2.0 (MPS backend for Apple Silicon)
* torchvision, matplotlib, plotly, networkx, pyyaml, loguru
