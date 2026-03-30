# Evolutionary Neural Architecture Search for Compact Networks

**Master's Thesis Project — Machine Learning**

Automated discovery of lightweight MLP and CNN architectures via evolutionary optimization (GA & NSGA-II), enhanced with weight sharing, XGBoost surrogate models, and predictive early stopping.

---

## Features

| Feature | Description |
|---|---|
| **MLP Search Space** | 14-gene genome: layers, widths, activation, dropout, batch norm, optimizer, LR, WD, batch size |
| **CNN Search Space** | 19-gene genome: conv blocks, filters, kernel size, depthwise separable, skip connections, pooling, dense head |
| **Single-Objective GA** | DEAP tournament selection, typed SBX crossover, Gaussian mutation, elitism via HallOfFame |
| **Multi-Objective (NSGA-II)** | Pareto optimization: accuracy vs. parameter count |
| **Weight Sharing** | One-shot supernets (MLP & CNN) with sliced forward passes for sub-architecture evaluation |
| **Surrogate Model** | XGBoost predictor with uncertainty-aware active learning; reduces evaluations after warm-up |
| **Predictive Early Stopping** | Small MLP trained on partial learning curves to predict final accuracy |
| **Datasets** | MNIST, FashionMNIST, CIFAR-10, CIFAR-100 |
| **Statistical Evaluation** | Wilcoxon signed-rank, Friedman test, Cohen's d, multi-seed cross-validation |
| **Visualizations** | Fitness curves, Pareto fronts (Matplotlib + Plotly), architecture diagrams, genome heatmaps, surrogate quality, comparison boxplots, weight distributions |
| **Baselines** | Fixed small architectures, random search, grid search (lite) |

---

## Project Structure

```
evolutionary_nas/
├── config.py                   # NASConfig dataclass, device detection, seeds
├── requirements.txt
├── __init__.py
│
├── search_space/
│   ├── mlp_space.py            # MLP gene definitions, random genome, decode
│   ├── cnn_space.py            # CNN gene definitions, random genome, decode
│   ├── genome_encoder.py       # encode/decode/repair/hash/describe
│   └── constraints.py          # param budget & validity checks
│
├── models/
│   ├── mlp_builder.py          # DynamicMLP (nn.Module), build_mlp()
│   ├── cnn_builder.py          # DynamicCNN with depthwise-sep, skip, GAP
│   ├── model_utils.py          # count_parameters, model_size, inference_time, FLOPs
│   └── weight_sharing/
│       ├── supernetwork_mlp.py # One-shot MLP supernet
│       ├── supernetwork_cnn.py # One-shot CNN supernet
│       └── path_sampler.py     # Subnet evaluation & supernet training
│
├── training/
│   ├── datasets.py             # MNIST/FashionMNIST/CIFAR10/CIFAR100 loaders
│   ├── trainer.py              # Full training with LR scheduling & early stop
│   └── fast_trainer.py         # Fast eval + LearningCurvePredictor
│
├── fitness/
│   ├── evaluator.py            # FitnessEvaluator: single & multi-objective
│   ├── cache.py                # Thread-safe FitnessCache
│   └── metrics.py              # Accuracy, F1, all metrics
│
├── surrogate/
│   ├── feature_extractor.py    # Genome → feature vector (one-hot, log-scale)
│   ├── predictor.py            # XGBoost mean + uncertainty model
│   ├── active_learning.py      # Acquisition function for candidate selection
│   └── surrogate_trainer.py    # Orchestrates lifecycle: warmup → retrain → select
│
├── evolution/
│   ├── operators.py            # Typed crossover (SBX) & mixed-type mutation
│   ├── initializer.py          # Random & biased-small population init
│   ├── callbacks.py            # GenerationStats, EvolutionLogger, EarlyStopping
│   ├── single_objective.py     # run_single_objective_ga() with DEAP
│   └── multi_objective.py      # run_nsga2() with DEAP
│
├── evaluation/
│   ├── statistical_tests.py    # Wilcoxon, Friedman, Cohen's d
│   ├── crossval.py             # Multi-seed evaluation
│   └── report_generator.py     # Markdown & LaTeX table generation
│
├── visualization/
│   ├── fitness_curves.py       # Best/mean/std fitness evolution
│   ├── pareto_front.py         # Matplotlib & Plotly Pareto plots
│   ├── architecture_diagram.py # MLP & CNN architecture block diagrams
│   ├── genome_heatmap.py       # Gene-value heatmap across generations
│   ├── surrogate_quality.py    # Predicted vs actual scatter, ρ curve
│   ├── comparison_boxplot.py   # Method comparison, diversity, early-stop savings
│   └── weight_distribution.py  # Weight histograms, learning curves
│
├── baselines/
│   ├── fixed_small.py          # Predefined Tiny/Small/Medium configs
│   ├── random_search.py        # Random genome sampling
│   └── grid_search_lite.py     # Reduced subspace grid search
│
├── experiments/
│   ├── run_mlp_search.py       # Full MLP NAS experiment
│   ├── run_cnn_search.py       # Full CNN NAS experiment
│   ├── run_multi_objective.py  # NSGA-II (accuracy vs. params)
│   ├── run_comparison.py       # NAS vs all baselines + statistical tests
│   └── results/                # plots/, logs/, checkpoints/, tables/
│
└── tests/
    ├── test_genome_encoder.py
    ├── test_mlp_builder.py
    ├── test_cnn_builder.py
    ├── test_fitness_evaluator.py
    ├── test_surrogate.py
    └── test_operators.py
```

---

## Quick Start

### 1. Install

```bash
cd evolutionary_nas
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Dry Run (< 2 minutes, verifies full pipeline)

```bash
# MLP search on MNIST, 3 generations, pop=10
python -m experiments.run_mlp_search --dry-run

# CNN search
python -m experiments.run_cnn_search --dry-run

# NSGA-II multi-objective
python -m experiments.run_multi_objective --dry-run

# Comparison with baselines
python -m experiments.run_comparison --dry-run
```

### 3. Full Experiments

```bash
# Single-objective MLP search (FashionMNIST, CIFAR-10, CIFAR-100)
python -m experiments.run_mlp_search

# Single-objective CNN search
python -m experiments.run_cnn_search

# Multi-objective NSGA-II
python -m experiments.run_multi_objective

# Full comparison with statistical testing
python -m experiments.run_comparison
```

### 4. Run Tests

```bash
cd evolutionary_nas
python -m pytest tests/ -v
```

---

## Design Choices

### Genome Encoding
- **Fixed-length float lists** for DEAP compatibility
- Gene types: `int`, `float`, `log_float`, `cat`, `cat_zero`
- Type-aware crossover: SBX for continuous, swap for categorical, blend/swap for integer
- Type-aware mutation: Gaussian perturbation for continuous, resample for categorical
- Automatic repair after every genetic operation to maintain validity

### Fitness Evaluation
- **Fast evaluation** (FAST_EPOCHS=5) during search for rapid filtering
- **Full evaluation** (FULL_EPOCHS=30) for top architectures
- MD5-based genome hashing with dataset-aware cache keys
- Parameter budget constraint: MAX_PARAMS=500,000

### Surrogate Model
- XGBoost regressor trained after SURROGATE_WARMUP=50 real evaluations
- Upper-bound model for uncertainty estimation
- Acquisition function: `predicted_acc + w × uncertainty`
- Retrained every 5 generations with accumulated data

### Multi-Objective
- NSGA-II with crowding distance tournament selection
- Objectives: maximize accuracy, minimize parameter count
- Pareto front extraction via `sortNondominated`

---

## Reproducibility

- 10 random seeds: `[42, 7, 13, 99, 100, 21, 55, 77, 11, 33]`
- `set_seed()` sets Python `random`, NumPy, PyTorch (CPU + MPS)
- All results logged as JSON with full genome and fitness history
- Wilcoxon signed-rank tests for pairwise method comparison
- Friedman test for multi-method comparison

---

## Device Support

- **Apple Silicon (M1/M2/M3)**: Automatic MPS backend detection
- **CPU**: Fallback when MPS unavailable
- No CUDA dependency — designed for laptop-scale experiments

---

## References

1. Real, E. et al. (2019). *Regularized Evolution for Image Classifier Architecture Search*. AAAI.
2. Deb, K. et al. (2002). *A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II*. IEEE TEC.
3. Liu, H. et al. (2019). *DARTS: Differentiable Architecture Search*. ICLR.
4. Pham, H. et al. (2018). *Efficient Neural Architecture Search via Parameter Sharing*. ICML.
5. Baker, B. et al. (2018). *Accelerating Neural Architecture Search using Performance Prediction*. ICLR Workshop.
