# Lab 6 — Genetic Programming & Particle Swarm Optimization

University lab covering **Genetic Programming (GP)** for symbolic regression and image processing, plus **Particle Swarm Optimization (PSO)** for 3-D function maximization. Built with [DEAP](https://github.com/DEAP/deap).

## Project Structure

```
lab6/
├── ex0a.py          # Symbolic regression via GP
├── ex0b.py          # GP image denoising & enhancement
├── ex1.py           # (BONUS) GP enhancement generalization study
├── ex2.py           # PSO function maximization
├── utils.py         # Shared helpers (primitives, metrics, plotting)
├── images/          # Generated input images
├── results/         # All saved PNG plots and output images
└── README.md
```

## Setup

```bash
# Install dependencies
pip install deap numpy matplotlib Pillow scipy

# Or install DEAP from source
pip install git+https://github.com/DEAP/deap@master numpy matplotlib Pillow scipy
```

## Running Each Exercise

All scripts are standalone and save plots to `results/`.

```bash
cd lab6/

# Exercise 0a — Symbolic Regression
python ex0a.py

# Exercise 0b — GP Image Denoising + Enhancement
python ex0b.py

# Exercise 1 (BONUS) — GP Enhancement Generalization Study
python ex1.py

# Exercise 2 — PSO Function Maximization
python ex2.py
```

## Exercise Details

### Exercise 0a — Symbolic Regression
- **Target:** `f(x,y) = sin(x)·cos(y) + log(1+x²)·exp(-y²/10)` (cannot be replicated exactly by the GP function set)
- **Function set:** `{+, -, *, pdiv, sin, cos}`
- **Outputs:** 3-D surface comparison (true / GP / error), convergence plot, hyperparameter sweep

### Exercise 0b — GP Image Processing
- **Part A (Denoising):** Adds uniform ±15 noise, trains a 3×3 GP filter, tests on unseen image
- **Part B (Enhancement):** Maps raw → enhanced target, tests generalization
- **Custom operators:** `edge_detect(a,b)` = |a−b|, `local_contrast(a,b)` = 128 + 1.5·(a−b)
- **Outputs:** Before/after image rows, convergence plots, PSNR & SSIM metrics

### Exercise 1 (BONUS) — Generalization Study
- Trains on 3 different image pairs, evaluates every tree on all pairs
- **Outputs:** Grouped bar chart of PSNR & SSIM across all (train, eval) combinations

### Exercise 2 — PSO
- **Target:** `f(x,y,z) = [1 + cos(2π·(1+exp(-(x-1)²/25 - (y-1.27)²)))] / (1+z²)`
- **Part 1:** Grid-search (c1, c2, w) with 50 particles
- **Part 2:** Best params + 100 particles
- **Part 3:** Further search with 100 particles
- **Comparisons:** Global vs ring topology, clip vs reflect speed limiting
- **Outputs:** Convergence curves, 2-D function slice with swarm overlay

## Reproducibility

All scripts use `random.seed(42)` and `np.random.seed(42)`.

## Dependencies

| Package    | Min Version |
|------------|-------------|
| Python     | 3.10+       |
| deap       | 1.4.1       |
| numpy      | any recent  |
| matplotlib | any recent  |
| Pillow     | any recent  |
| scipy      | any recent  |
