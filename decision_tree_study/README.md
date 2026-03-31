# Systematic Study of Overfitting, Pruning, and Generalization in Decision Trees

> Under Controlled Noise and Dataset Size Conditions

## Research Questions

| # | Question |
|---|---------|
| Q1 | How does max_depth affect the overfitting gap for unpruned decision trees? |
| Q2 | Which pruning strategy (pre-depth, pre-samples, CCP, combined) best preserves generalisation under label noise? |
| Q3 | At what dataset sizes does pruning become most beneficial? |
| Q4 | How does feature noise interact differently with pre-pruning vs post-pruning? |
| Q5 | Is there an optimal noise×depth region where combined pruning dominates? |

## Quick Start

```bash
cd decision_tree_study
pip install -r requirements.txt

# Dry-run: iris only, ≈30 s
python -m experiments.run_all --dry-run

# Full run: all 8 datasets × all configs
python -m experiments.run_all

# Run tests
python -m pytest tests/ -v
```

## Project Structure

```
decision_tree_study/
├── config.py                   # StudyConfig dataclass with all constants
├── requirements.txt
├── data/
│   ├── loaders.py              # 3 real + 5 synthetic datasets
│   ├── noise_injector.py       # Label & feature noise injection
│   └── dataset_sampler.py      # Stratified subsampling
├── trees/
│   ├── tree_factory.py         # Build trees for 5 pruning strategies
│   ├── pruning_strategies.py   # CV-based hyper-parameter selection
│   ├── tree_metrics.py         # Accuracy, F1, depth, leaves, gap
│   └── tree_inspector.py       # Rule extraction, impurity stats
├── experiments/
│   ├── exp_depth.py            # Exp 1: accuracy vs max_depth
│   ├── exp_pruning.py          # Exp 2: strategy comparison
│   ├── exp_dataset_size.py     # Exp 3: learning curves
│   ├── exp_noise_label.py      # Exp 4: label noise sweep
│   ├── exp_noise_feature.py    # Exp 5: feature noise sweep
│   ├── exp_interaction.py      # Exp 6: 2D interaction heatmaps
│   ├── exp_ccp_alpha.py        # Exp 7: CCP alpha path
│   └── run_all.py              # Orchestrator (--dry-run supported)
├── evaluation/
│   ├── statistical_tests.py    # Wilcoxon, Friedman, Nemenyi, correlations
│   ├── crossval.py             # Multi-seed stratified K-fold
│   └── report_generator.py     # Markdown + LaTeX tables
├── visualization/
│   ├── style.py                # Shared palette, save_fig, apply_style
│   ├── tree_plot.py            # Decision tree rendering
│   ├── depth_curves.py         # Accuracy vs depth line plots
│   ├── pruning_comparison.py   # Bar + box plots of strategies
│   ├── learning_curves.py      # Accuracy vs dataset size
│   ├── noise_curves.py         # Label / feature noise degradation
│   ├── interaction_heatmaps.py # 2D heatmaps
│   ├── ccp_path_plot.py        # CCP alpha path with knee-point
│   ├── confusion_matrices.py   # Normalised confusion matrices
│   ├── feature_importance.py   # Horizontal bar chart
│   └── overfitting_gap.py      # Gap box-plots and line charts
└── tests/
    ├── test_noise_injector.py
    ├── test_tree_metrics.py
    ├── test_crossval.py
    └── test_statistical_tests.py
```

## Methodology

### Datasets
- **Real**: Iris, Breast Cancer, Wine (from sklearn)
- **Synthetic**: 5 variants via `make_classification` with increasing complexity (features, classes, redundancy)

### Pruning Strategies
| Strategy | Description |
|----------|-------------|
| `none` | No pruning — fully grown tree |
| `pre_depth` | Pre-pruning via `max_depth` (CV-selected) |
| `pre_samples` | Pre-pruning via `min_samples_leaf` + `min_samples_split` |
| `ccp` | Post-pruning via cost-complexity pruning (CV-selected alpha) |
| `combined` | Pre-pruning (depth) + post-pruning (CCP alpha) |

### Noise Injection
- **Label noise**: symmetric (uniform random) and asymmetric (class i → i+1 mod C), rates 0–30%
- **Feature noise**: Gaussian N(0, σ·std) + optional masking, σ ∈ {0, 0.1, 0.3, 0.5, 1.0}

### Experimental Grid
- 10 random seeds × 10-fold CV
- Depths: [1, 2, 3, 4, 5, 7, 10, 15, 20, None]
- Dataset sizes: [50, 100, 200, 500, 1000, 2000, 5000]

### Statistical Analysis
- Paired Wilcoxon signed-rank test for pairwise comparisons
- Friedman test + Nemenyi post-hoc for multi-group comparisons
- Cohen's d for effect size
- Pearson and Spearman correlations

## Figures Generated (15+)

1. Decision tree rendering
2. Accuracy vs depth curves (train/val/test)
3. Pruning strategy bar chart
4. Pruning strategy box plots
5. Learning curves per strategy
6. Label noise degradation (symmetric)
7. Label noise degradation (asymmetric)
8. Feature noise degradation
9. Interaction heatmap: noise × depth
10. Interaction heatmap: noise × size
11. CCP alpha path with knee-point
12. Confusion matrix grid
13. Feature importance bar chart
14. Overfitting gap by strategy
15. Overfitting gap vs depth

## Output

All results are saved under `experiments/results/`:
- `*.csv` — raw experiment data
- `plots/` — PNG figures at 300 DPI
- `report.md` — auto-generated Markdown summary
- `tables.tex` — LaTeX table snippets

## Requirements

- Python ≥ 3.11
- scikit-learn ≥ 1.4
- numpy ≥ 1.26
- pandas ≥ 2.1
- scipy ≥ 1.12
- matplotlib ≥ 3.8
- seaborn ≥ 0.13
