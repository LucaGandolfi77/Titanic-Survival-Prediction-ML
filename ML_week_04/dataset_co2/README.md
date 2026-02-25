# 🌍 Global CO₂ Emissions – ML Analysis

Comprehensive machine-learning analysis of global CO₂ emission data from
[Our World in Data](https://github.com/owid/co2-data) (1750–2021, 278 countries).

## Dataset

- **50,598** rows × **79** columns
- Country-year panel data covering emissions by fuel source (coal, oil, gas),
  GDP, population, temperature-change attribution, land-use change, and more

## ML Tasks

| # | Task | Type |
|---|---|---|
| 1 | **CO₂ emissions prediction** | Regression (Random Forest, GB, Ridge, Lasso) |
| 2 | **Emission-level classification** | Multi-class (Low / Medium / High / Very High) |
| 3 | **CO₂ growth-rate forecasting** | Regression with temporal features |
| 4 | **Country clustering** | Unsupervised (K-Means, DBSCAN, PCA) |

## Techniques

- 10+ supervised models (LR, DT, RF, GB, SVM, KNN, NB, MLP, Voting, Stacking)
- GridSearchCV & RandomizedSearchCV hyperparameter tuning
- Stratified K-Fold cross-validation
- Feature importance (tree-based + permutation)
- Confusion matrices, ROC / AUC, learning curves
- Self-contained HTML report with embedded visualisations

## Quick Start

```bash
pip install -r requirements.txt
jupyter notebook co2_ml_analysis.ipynb
```

## Outputs

- `outputs/plots/` – 16+ PNG visualisations
- `outputs/co2_cleaned.csv` – cleaned dataset
- `outputs/co2_ml_report.html` – self-contained HTML report
