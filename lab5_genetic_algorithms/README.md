# Lab 5 — Genetic Algorithms with DEAP

University lab implementing four exercises on Genetic Algorithms using the [DEAP](https://github.com/DEAP/deap) library.

## Project Structure

```
lab5_genetic_algorithms/
├── requirements.txt
├── README.md
├── main.py                        # Entry point (argparse)
├── utils/
│   ├── __init__.py
│   └── plotting.py                # Shared matplotlib utilities
├── exercise1/                     # Function Optimization
│   ├── __init__.py
│   ├── float_repr.py              # GA with float representation
│   └── binary_repr.py             # GA with binary (90-bit) representation
├── exercise2/                     # Pattern Guessing
│   ├── __init__.py
│   ├── smiley.txt                 # Target 16×14 smiley pattern
│   ├── pattern_guesser.py         # Main GA to match target
│   └── scaling_experiment.py      # Nf vs size + hyperparam search
├── exercise3/                     # N-Queens Problem
│   ├── __init__.py
│   ├── nqueens_smart.py           # Permutation representation
│   └── nqueens_dumb.py            # 2*N integer-pair representation
└── exercise4/                     # Ant Game
    ├── __init__.py
    ├── ant_game.py                # Board logic + environment
    └── ant_ga.py                  # GA evolving move sequences
```

## Setup

```bash
pip install -r requirements.txt
# or install DEAP from source:
pip install git+https://github.com/DEAP/deap@master
```

## Usage

```bash
# Run a specific exercise
python main.py --exercise 1
python main.py --exercise 2
python main.py --exercise 3
python main.py --exercise 4

# Run all exercises
python main.py --all
```

## Exercises Overview

### Exercise 1 — Function Optimization
Minimizes `f(x,y,z) = (1.5 + sin(z)) * (sqrt((20-x)² + (30-y)²) + 1)` using two representations:
- **Float**: 3 real-valued genes in [-250, 250]
- **Binary**: 90 bits (30 per variable), decoded to floats

### Exercise 2 — Pattern Guessing
Evolves a binary string to match a 16×14 smiley face pattern (224 bits). Includes:
- Pattern matching GA with exact Nf counting
- Scaling experiment: Nf vs pattern size
- Hyperparameter search: heatmap of (pop_size, ngen) combinations

### Exercise 3 — N-Queens
Solves the N-Queens problem with two representations:
- **Smart**: permutation (no row/col conflicts by construction)
- **Dumb**: 2*N integer pairs (all conflicts possible)
Compares success rates for N = 8, 16, 32, 64, 128.

### Exercise 4 — Ant Game
Evolves a 20-move sequence (40 bits) for an ant navigating a 10×10 toroidal grid to collect food. Visualizes the optimal path found.

## Output
Each exercise generates PNG plots in its respective folder. All seeds are fixed (`random.seed(42)`) for reproducibility.
