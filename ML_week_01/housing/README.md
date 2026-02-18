# 🏠 Iowa Housing Price Prediction

End-to-end regression pipeline on the
[Kaggle Ames Housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
(80+ features → predict `SalePrice`).

---

## Project structure

```
housing/
├── housing_regression.ipynb   # Full pipeline (12 sections)
├── data/
│   ├── train.csv              # 1460 rows, 81 columns
│   └── test.csv               # 1459 rows, 80 columns
├── outputs/
│   ├── submission.csv         # Generated at runtime
│   └── figures/               # All plots saved here
├── requirements.txt
└── README.md
```

## Quick start

```bash
# 1. Virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Dependencies
pip install -r requirements.txt

# 3. Run
# Open housing_regression.ipynb in VSCode → "Run All Cells"
```

## Pipeline overview

| # | Section | Description |
|---|---------|-------------|
| 0 | Config & Imports | Constants, seeds, library imports |
| 1 | Load & Inspect | Shape, dtypes, numeric vs categorical split |
| 2 | Target Analysis | SalePrice distribution, log transform, QQ plot |
| 3 | Missing Data | Domain-aware imputation (NA = "None" for absent features) |
| 4 | EDA (10 charts) | Correlations, scatter plots, violin/box plots, pairplot |
| 5 | Feature Engineering | 12 new features, ordinal + one-hot encoding, skewness fix |
| 6 | Model Comparison | 5 models × 5-fold CV (RMSE, R², timing) |
| 7 | sklearn Pipeline | ColumnTransformer + Pipeline (leak-proof) |
| 8 | Hyperparameter Tuning | GridSearchCV on best model |
| 9 | Model Interpretation | Feature importance, residuals, coefficients |
| 10 | Submission | Predict test set → `submission.csv` |
| 11 | Key Takeaways | Findings, best model, next steps |

## Models compared

| Model | CV RMSE (log) |
|-------|---------------|
| Linear Regression | ~0.18 |
| Ridge | ~0.14 |
| Lasso | ~0.14 |
| Random Forest | ~0.14 |
| **XGBoost** | **~0.13** |

## License

MIT — for educational purposes.
