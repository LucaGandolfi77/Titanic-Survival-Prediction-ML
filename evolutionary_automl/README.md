# Evolutionary AutoML — Automatic Optimization of Full ML Pipelines

**Master's Thesis Project — Machine Learning, Computer Engineering**

An evolutionary computation framework for automatically optimizing complete
scikit-learn machine learning pipelines — including preprocessing, feature
selection, dimensionality reduction, classifier selection, and hyperparameter
tuning — using Genetic Algorithms and Multi-Objective Evolutionary Strategies
(NSGA-II).

---

## Quick Start

```bash
git clone https://github.com/your-username/evolutionary_automl.git
cd evolutionary_automl
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Quick dry-run verification (< 60 seconds)
python experiments/run_single_objective.py --dry-run
python experiments/run_multi_objective.py --dry-run
python experiments/run_comparison.py --dry-run

# Full experiments (may take several hours)
python experiments/run_single_objective.py
python experiments/run_multi_objective.py
python experiments/run_comparison.py
```

---

## Project Structure

```
evolutionary_automl/
├── config.py                       # Centralized configuration (seeds, paths, GA params)
├── requirements.txt                # Python dependencies
│
├── search_space/
│   ├── space_definition.py         # Full search space: scalers, selectors, classifiers, HPs
│   ├── pipeline_builder.py         # Build sklearn Pipeline from a chromosome
│   └── validators.py               # Chromosome validation and repair
│
├── genome/
│   ├── chromosome.py               # Encoding, decoding, hashing
│   ├── operators.py                # Gene-type-aware crossover and mutation
│   └── initializer.py              # Population initialization (random + seeded)
│
├── fitness/
│   ├── evaluator.py                # Fitness evaluation with CV, timeout, caching
│   ├── cache.py                    # Thread-safe fitness cache
│   └── metrics.py                  # F1, training time, feature count, memory
│
├── evolution/
│   ├── single_objective.py         # GA with DEAP: tournament + elitism
│   ├── multi_objective.py          # NSGA-II with DEAP: Pareto front search
│   └── callbacks.py                # Logging, early stopping, generation stats
│
├── baselines/
│   ├── random_search.py            # RandomizedSearchCV-equivalent baseline
│   ├── grid_search.py              # Grid search on reduced subspace
│   └── manual_tuning.py            # Hand-designed sensible pipelines
│
├── evaluation/
│   ├── statistical_tests.py        # Wilcoxon, Mann-Whitney, Friedman tests
│   ├── crossval.py                 # Multi-seed stratified k-fold
│   └── report_generator.py         # Markdown + LaTeX table generation
│
├── visualization/
│   ├── fitness_curves.py           # Best/mean/std F1 per generation
│   ├── pareto_front.py             # 2D Pareto front (Plotly + matplotlib)
│   ├── confusion_matrix.py         # Confusion matrix for best pipeline
│   ├── genome_heatmap.py           # Gene frequency heatmap across generations
│   ├── pipeline_graph.py           # Block diagram of best pipeline
│   └── comparison_plots.py         # Boxplots, diversity, convergence
│
├── experiments/
│   ├── run_single_objective.py     # Single-objective GA experiment
│   ├── run_multi_objective.py      # NSGA-II experiment
│   ├── run_comparison.py           # Full method comparison
│   └── results/                    # Auto-generated outputs
│       ├── plots/
│       ├── logs/
│       └── tables/
│
└── tests/
    ├── test_chromosome.py
    ├── test_pipeline_builder.py
    ├── test_fitness_evaluator.py
    └── test_operators.py
```

---

## Search Space Design

The search space encodes a complete ML pipeline as a fixed-length chromosome
of 13 genes, each normalized to [0, 1]:

| Gene | Component | Type | Options |
|------|-----------|------|---------|
| 0 | Scaler | Categorical (4) | None, StandardScaler, MinMaxScaler, RobustScaler |
| 1 | Feature Selection | Categorical (4) | None, SelectKBest, SelectFromModel, VarianceThreshold |
| 2 | K features ratio | Float | Ratio of features to select |
| 3 | Dimensionality Reduction | Categorical (3) | None, PCA, TruncatedSVD |
| 4 | Classifier | Categorical (7) | DT, RF, GB, SVC, KNN, LR, MLP |
| 5–12 | Hyperparameters | Mixed | Per-classifier decoded ranges |

**Design rationale**: A uniform [0,1] encoding allows standard genetic operators
to work on all gene types. Categorical genes are mapped via rounding to the
nearest valid option. Continuous hyperparameters use linear or log-scale
decoding. This avoids the need for heterogeneous representations while
maintaining full expressiveness.

---

## Evolutionary Operators

### Crossover: Two-Point Typed (cx_two_point_typed)

Standard two-point crossover with type-aware behavior at swap boundaries:
- **Categorical genes**: direct value swap
- **Continuous genes**: BLX-α blending (α ∈ [-0.1, 1.1]) to explore beyond
  the parents' range

### Mutation: Mixed-Type (mut_mixed_type)

Per-gene independent mutation with probability `indpb = 0.15`:
- **Categorical genes**: uniform random resample from [0, 1]
- **Continuous genes**: Gaussian perturbation (σ = 0.15), clipped to [0, 1]

### Chromosome Repair

After crossover and mutation, chromosomes are repaired to ensure:
- All genes are within [0, 1]
- Length is exactly 13
- Incompatible hyperparameter combinations are fixed at pipeline build time

---

## Fitness Function

- **Primary objective**: Macro-averaged F1 via StratifiedKFold (k=5)
- **Secondary objective** (NSGA-II): Training time in seconds
- **Timeout**: Configurable per-evaluation time budget (default 60s)
- **Cache**: Thread-safe dictionary keyed by chromosome hash + dataset name
- **Error handling**: Failed evaluations return (0.0, penalty_time)

---

## Algorithms

### Strategy A — Single-Objective GA
- Selection: Tournament (k=3)
- Crossover probability: 0.7
- Mutation probability: 0.2
- Elitism: Top-5 Hall of Fame preserved each generation
- Population: 50, Generations: 30

### Strategy B — NSGA-II (Multi-Objective)
- Objectives: Maximize F1, Minimize training time
- Selection: NSGA-II (crowding distance)
- Population: 100, Generations: 40
- Output: Full Pareto front of non-dominated solutions

---

## Datasets

| Dataset | Samples | Features | Classes | Purpose |
|---------|---------|----------|---------|---------|
| iris | 150 | 4 | 3 | Sanity check |
| breast_cancer | 569 | 30 | 2 | Binary medical classification |
| wine | 178 | 13 | 3 | Correlated features |
| digits | 1,797 | 64 | 10 | Medium-scale multi-class |

---

## Statistical Evaluation

All stochastic methods run with N=10 independent seeds. Statistical tests:

1. **Wilcoxon signed-rank**: Paired comparison (GA vs each baseline)
2. **Friedman test**: Multi-method comparison across datasets
3. **Effect sizes**: Reported alongside p-values
4. **Summary statistics**: mean ± std, median, IQR per method/dataset

---

## Visualizations

All saved to `experiments/results/plots/`:

| File | Description |
|------|-------------|
| `ga_fitness_*.png` | Best/mean/std F1 per generation with shaded band |
| `pareto_front_*.html/png` | Interactive + static Pareto front |
| `confusion_matrix_*.png` | Confusion matrix for best GA pipeline |
| `comparison_boxplot_*.png` | F1 distribution boxplot across methods |
| `convergence_*.png` | GA vs NSGA-II convergence comparison |
| `ga_diversity_*.png` | Population diversity over generations |
| `ga_pipeline_*.png` | Block diagram of best discovered pipeline |

---

## Configuration

All parameters are centralized in `config.py`:

```python
RANDOM_SEEDS = (42, 7, 13, 99, 100, 21, 55, 77, 11, 33)
POP_SIZE_GA = 50          N_GEN_GA = 30
POP_SIZE_NSGA = 100       N_GEN_NSGA = 40
CV_FOLDS = 5              MAX_EVAL_SECONDS = 60
N_RUNS = 10
```

---

## Running Tests

```bash
cd evolutionary_automl
python -m pytest tests/ -v
```

---

## M1/Apple Silicon Notes

- Uses `joblib` with `loky` backend (no direct `multiprocessing.Pool`)
- All file paths use `pathlib.Path`
- Random states set explicitly before each experiment
- No OpenMP parallelism inside evolutionary loops

---

## References

1. Fortin, F.-A., De Rainville, F.-M., Gardner, M.-A., Parizeau, M., & Gagné, C. (2012).
   **DEAP: Evolutionary Algorithms Made Easy.** *JMLR*, 13, 2171–2175.

2. Olson, R. S., Bartley, N., Urbanowicz, R. J., & Moore, J. H. (2016).
   **Evaluation of a Tree-based Pipeline Optimization Tool for Automating Data Science.**
   *GECCO 2016*.

3. Feurer, M., Klein, A., Eggensperger, K., Springenberg, J. T., Blum, M., & Hutter, F. (2015).
   **Efficient and Robust Automated Machine Learning.** *NeurIPS 2015*.

4. Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
   **A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II.**
   *IEEE Trans. Evolutionary Computation*, 6(2), 182–197.

5. Demšar, J. (2006).
   **Statistical Comparisons of Classifiers over Multiple Data Sets.**
   *JMLR*, 7, 1–30.

---

## License

This project is developed for academic research purposes as part of a
Master's thesis in Computer Engineering.
