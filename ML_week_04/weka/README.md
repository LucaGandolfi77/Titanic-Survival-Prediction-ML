# PyWeka – Machine Learning Explorer

A Python desktop application that replicates the core features of **Weka** (Waikato Environment for Knowledge Analysis), built with **Tkinter**, **scikit-learn**, **XGBoost**, **matplotlib**, and **seaborn**.

---

## Features

| Tab | Description |
|-----|-------------|
| **Preprocess** | Load CSV/TSV/Excel/ARFF, view stats, handle missing values (mean/median/mode/zero/drop), encode categoricals (Label/One-Hot/Ordinal), scale (Standardize/Normalize), remove columns, remove outliers, discretize, cast types, sample, undo |
| **Classify** | 13 classifiers (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, Bagging, Extra Trees, KNN, SVM RBF, SVM Linear, Naive Bayes, MLP, XGBoost). Evaluation: percentage split, k-fold CV, training set. Confusion matrix popup |
| **Regression** | 13 regressors (Linear, Ridge, Lasso, ElasticNet, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, Bagging, Extra Trees, KNN, SVR, XGBoost). Actual vs Predicted scatter popup |
| **Cluster** | K-Means, Mini-Batch K-Means, DBSCAN, Agglomerative, Mean Shift. Elbow/Silhouette plots, PCA scatter visualization |
| **Associate** | Apriori association rules (via mlxtend) with support/confidence/lift. Fallback co-occurrence analysis |
| **Select Attributes** | SelectKBest, Random Forest Importance, RFE, Correlation Analysis |
| **Visualize** | Histogram, Box Plot, Scatter Plot, Correlation Heatmap, Pairplot, Class Distribution, Violin Plot, Bar Chart, Distribution Grid. Configurable hue/bins/alpha |

## Built-in Sample Datasets
- Iris, Wine, Breast Cancer, Diabetes (from scikit-learn)

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the app
python app.py

# Or load a dataset directly
python app.py path/to/dataset.csv
```

## Requirements

- Python 3.10+
- Tkinter (included with Python on most systems)
- See `requirements.txt` for Python packages

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Cmd/Ctrl + O | Open dataset |
| Cmd/Ctrl + S | Save dataset |
| Cmd/Ctrl + Z | Undo last operation |
| Cmd/Ctrl + Q | Quit |

## Project Structure

```
weka/
├── app.py                  # Entry point
├── requirements.txt
├── README.md
├── core/
│   ├── __init__.py
│   ├── data_manager.py     # Dataset I/O, undo, introspection
│   ├── preprocessor.py     # Preprocessing operations
│   └── evaluator.py        # ML model training & evaluation
└── ui/
    ├── __init__.py
    ├── main_window.py       # Main window + menu + tabs
    ├── preprocess_tab.py    # Preprocess panel
    ├── classify_tab.py      # Classification panel
    ├── regression_tab.py    # Regression panel
    ├── cluster_tab.py       # Clustering panel
    ├── associate_tab.py     # Association rules panel
    ├── select_tab.py        # Feature selection panel
    ├── visualize_tab.py     # Visualization panel
    └── widgets.py           # Reusable UI components
```
