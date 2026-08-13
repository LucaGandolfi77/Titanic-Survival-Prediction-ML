# Lab 1 — kNN & Decision Trees

University lab exercise originally based on WEKA, re-implemented in Python with scikit-learn.

## Exercises

| # | Topic | File |
|---|-------|------|
| 1 | kNN on circle/square problem | `exercise1_knn.py` |
| 1b | Feature engineering variants | `exercise1b_repr.py` |
| 2 | Decision Tree on digit images | `exercise2_digits.py` |
| 2b | Overfitting control (min_samples_leaf) | `exercise2b_overfit.py` |

## Setup

```bash
pip install -r requirements.txt
```

## Data

Place the following ARFF files in `data/`:

- `circletrain.arff`, `circletest.arff`, `circleall.arff` — circle/square 2-D problem
- `Bigtest1_104.arff`, `Bigtest2_104.arff` — 13×8 binary digit images (0–9)

If the circle ARFF files are missing, the code will generate them automatically.

## Run

```bash
python main.py
```

All results are printed to stdout. Plots and CSV files are saved to `outputs/`.

## Project Structure

```
lab1_knn_dectrees/
├── main.py              # Orchestrator
├── data_loader.py       # ARFF parser + synthetic data generator
├── exercise1_knn.py     # Ex 1: kNN (scratch + sklearn)
├── exercise1b_repr.py   # Ex 1b: feature engineering
├── exercise2_digits.py  # Ex 2: digit classification
├── exercise2b_overfit.py# Ex 2b: overfitting analysis
├── utils.py             # Shared helpers
├── data/                # .arff datasets
├── outputs/             # Generated plots & CSVs
├── requirements.txt
└── README.md
```
