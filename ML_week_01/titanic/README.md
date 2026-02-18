# 🚢 Titanic Survival Prediction

End-to-end binary classification on the classic
[Kaggle Titanic dataset](https://www.kaggle.com/c/titanic) using pure
**scikit-learn**.

---

## Project structure

```
titanic/
├── titanic_eda.ipynb   # Full pipeline notebook (8 sections)
├── train.csv           # Kaggle training set (891 rows)
├── test.csv            # Kaggle test set (418 rows)
├── requirements.txt    # Pinned dependencies
└── README.md           # ← you are here
```

## Quick start

```bash
# 1. Create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open in VSCode and "Run All Cells"
code titanic_eda.ipynb
```

## Pipeline overview

| Section | What happens |
|---------|-------------|
| 0 | Setup & imports |
| 1 | Load data & first look (nulls heatmap) |
| 2 | EDA — 8 charts with bullet insights |
| 3 | Feature engineering (impute, encode, create, scale) |
| 4 | Model comparison (LogReg, RF, GBC, SVC) — 5-fold CV |
| 5 | Hyperparameter tuning (GridSearchCV) |
| 6 | Hold-out evaluation (confusion matrix, ROC, feature importance) |
| 7 | Predict on test set → `submission.csv` |
| 8 | Key takeaways |

## Key results

| Model | CV Accuracy (mean ± std) |
|-------|--------------------------|
| Logistic Regression | ~80 % |
| Random Forest | ~82 % |
| **Gradient Boosting** | **~83 %** |
| SVC | ~82 % |

> Results may vary slightly due to random splits.

## License

MIT — for educational purposes.
