# Ensemble Learning for Small and Noisy Datasets

**A Systematic Comparison of Bagging, Random Forest, Hard Voting, AdaBoost, and Gradient Boosting Under Controlled Data Conditions**

## Research Questions

| ID | Question |
|----|----------|
| Q1 | How does dataset size affect ensemble accuracy, and at what sample size do ensembles stop outperforming a single Decision Tree? |
| Q2 | Which ensemble family (homogeneous vs heterogeneous) is more robust to class imbalance? |
| Q3 | How does symmetric label noise degrade each ensemble's performance, and which method retains the highest accuracy at 30 % noise? |
| Q4 | Are boosting methods (AdaBoost, Gradient Boosting) more sensitive to outliers than bagging-based methods? |
| Q5 | Is there a measurable trade-off between base-learner diversity and ensemble accuracy, and does the Krogh–Vedelsby ambiguity decomposition explain the gap between homogeneous and heterogeneous ensembles? |

## Methods

| # | Method | Type | Key Hyperparameters |
|---|--------|------|---------------------|
| 1 | Bagging | Homogeneous | n_estimators ∈ {10,25,50,100,200} |
| 2 | Random Forest | Homogeneous | n_estimators, max_features ∈ {sqrt,log2,None}, max_depth ∈ {None,5,10} |
| 3 | AdaBoost (SAMME) | Homogeneous | n_estimators, learning_rate ∈ {0.1,0.5,1.0}, base: stump |
| 4 | Gradient Boosting | Homogeneous | n_estimators, learning_rate ∈ {0.05,0.1,0.2}, max_depth ∈ {3,5} |
| 5 | Hard Voting | Heterogeneous | DT + SVC + KNN + LogReg + NaiveBayes |
| 6 | Soft Voting | Heterogeneous | Same 5 base learners, probability averaging |
| 7 | Decision Tree | Baseline | Default sklearn |
| 8 | Logistic Regression | Baseline | max_iter=1000 |

## Experiments

| Exp | Focus | Sweep Variable | Key Metric |
|-----|-------|----------------|------------|
| 1 | Dataset size | n ∈ {30,50,100,200,500,1000,2000} | F1 (macro) |
| 2 | Class imbalance | ratio ∈ {1:1,1:2,1:5,1:10,1:20} | F1, AUC |
| 3 | Label noise | rate ∈ {0,5,10,20,30}% | Accuracy |
| 4 | Outliers | fraction ∈ {0,2,5,10,20}% | Accuracy |
| 5 | Diversity–accuracy | Disagreement, Q-stat, κ, double-fault | F1 vs diversity |
| 6 | n_estimators | n ∈ {5,10,25,50,100,200} | Accuracy |
| 7 | Interactions | noise×size, imbalance×outliers | Accuracy |

## Datasets

- **Real**: Breast Cancer (569×30), Wine (178×13), Iris (150×4)
- **Synthetic**: Clean, Noisy (flip_y=0.15), Redundant (10 redundant features), Moons (noise=0.3)

## Diversity Metrics

- **Disagreement** — fraction of samples on which two classifiers disagree
- **Q-statistic** — Yule's Q (positive → correlated, negative → diverse)
- **Double-fault** — fraction misclassified by both
- **Cohen's κ** — agreement beyond chance
- **Ambiguity decomposition** (Krogh & Vedelsby 1995): avg_individual_error = ensemble_error + ambiguity

## Statistical Evaluation

- 15 independent seeds × RepeatedStratifiedKFold(5, 3)
- Paired Wilcoxon signed-rank with effect size r = Z/√N
- Friedman test + Nemenyi post-hoc (Critical Difference diagram)
- Cohen's d for effect size
- Spearman ρ for diversity–accuracy correlation

## Quick Start

```bash
cd ensemble_study
pip install -r requirements.txt

# Dry run (Exp 1 + 5, breast_cancer only, 3 seeds, ~60 s)
python -m experiments.run_all --dry-run

# Full study (all 7 experiments × 7 datasets × 15 seeds)
python -m experiments.run_all

# Run tests
pytest tests/ -v
```

## Project Structure

```
ensemble_study/
├── config.py                      # Central frozen dataclass configuration
├── requirements.txt
├── README.md
├── data/
│   ├── __init__.py
│   ├── loaders.py                 # 7 datasets (3 real + 4 synthetic)
│   ├── noise_injector.py          # Symmetric label noise, feature noise
│   ├── imbalance_generator.py     # Undersample-based imbalance
│   ├── outlier_injector.py        # k-sigma outlier injection
│   └── dataset_sampler.py         # Stratified subsampling
├── ensembles/
│   ├── __init__.py
│   ├── homogeneous.py             # Bagging, RF, AdaBoost, GradBoost
│   ├── heterogeneous.py           # Hard Voting (5 diverse learners)
│   ├── soft_voting.py             # Soft Voting
│   ├── ensemble_factory.py        # build_method(name) dispatcher
│   └── diversity_metrics.py       # Pairwise + ensemble-level + ambiguity
├── experiments/
│   ├── __init__.py
│   ├── utils.py                   # evaluate_method(), run_method_cv()
│   ├── exp_dataset_size.py        # Exp 1
│   ├── exp_class_imbalance.py     # Exp 2
│   ├── exp_label_noise.py         # Exp 3
│   ├── exp_outliers.py            # Exp 4
│   ├── exp_diversity_accuracy.py  # Exp 5 (thesis core)
│   ├── exp_n_estimators.py        # Exp 6
│   ├── exp_interaction.py         # Exp 7
│   ├── run_all.py                 # Orchestrator (--dry-run)
│   └── results/
├── evaluation/
│   ├── __init__.py
│   ├── statistical_tests.py       # Wilcoxon, Friedman, Nemenyi, Cohen's d
│   ├── crossval.py                # RepeatedStratifiedKFold wrapper
│   ├── imbalance_metrics.py       # F1, balanced acc, AUC, G-mean, MCC
│   └── report_generator.py        # Markdown + LaTeX tables
├── visualization/
│   ├── __init__.py
│   ├── _common.py                 # Shared palette, save helper
│   ├── learning_curves.py         # Exp 1
│   ├── imbalance_curves.py        # Exp 2
│   ├── noise_curves.py            # Exp 3
│   ├── outlier_curves.py          # Exp 4
│   ├── diversity_scatter.py       # Exp 5
│   ├── n_estimators_curves.py     # Exp 6
│   ├── interaction_heatmaps.py    # Exp 7
│   ├── comparison_boxplot.py      # Overall boxplot
│   ├── confusion_matrices.py      # Per-method confusion grid
│   ├── roc_curves.py              # ROC per method
│   ├── ambiguity_decomposition.py # Stacked bar
│   ├── critical_difference.py     # Nemenyi CD diagram
│   └── correlation_heatmap.py     # Spearman ρ matrix
└── tests/
    ├── test_noise_injector.py
    ├── test_imbalance_generator.py
    ├── test_diversity_metrics.py
    ├── test_statistical_tests.py
    └── test_ensemble_factory.py
```

## References

- Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
- Freund, Y. & Schapire, R. E. (1997). *A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting*. JCSS, 55(1), 119–139.
- Krogh, A. & Vedelsby, J. (1995). *Neural Network Ensembles, Cross Validation, and Active Learning*. NIPS 7, 231–238.
- Kuncheva, L. I. (2004). *Combining Pattern Classifiers: Methods and Algorithms*. Wiley.
- Dietterich, T. G. (2000). *Ensemble Methods in Machine Learning*. MCS 2000, 1–15.
