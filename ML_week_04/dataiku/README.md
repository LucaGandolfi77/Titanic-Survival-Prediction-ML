# DataikuLite DSS

A lightweight, **production-ready** Python desktop application that replicates the
core functionality of **Dataiku Data Science Studio (DSS)**.

Built entirely with the Python standard library GUI toolkit **tkinter (ttk)** and
the scientific Python ecosystem (**pandas, numpy, scikit-learn, matplotlib, seaborn**).

---

## Features

| Module | Capabilities |
|---|---|
| **Project Management** | Create / open / save projects as JSON bundles. Project explorer tree sidebar. |
| **Dataset Manager** | Import CSV, Excel, JSON, Parquet. Paginated preview (100 rows/page). Column stats, dtype detection, schema editor. |
| **Visual Flow (Pipeline)** | Drag-and-drop canvas with Dataset → Recipe → Output nodes. Connect, configure, and run pipelines. |
| **Prepare Recipe** | Step-based transformations: filter, drop, rename, fill NA, encode, normalise, standardise, extract datetime, custom formula. Real-time 5-row preview. |
| **EDA** | Auto-EDA report (correlation heatmap, distributions, missing values). Manual chart builder (scatter, line, bar, box, histogram, heatmap) with PNG export. |
| **ML Lab** | Classification (Logistic Regression, Random Forest, SVM, KNN, XGBoost), Regression (Linear, Ridge, Lasso, RF), Clustering (KMeans, DBSCAN). Hyperparameter panel, background training, metrics, confusion matrix, feature importance. Export `.pkl`. |
| **Notebook** | Jupyter-like code cells with shared namespace. Run individual / all cells, inject DataFrames, export as `.py`. |
| **Dark Theme** | Full dark UI (#1e1e2e palette) with accent colour #7c3aed. |

---

## Project Structure

```
ML_week_04/dataiku/
├── main.py               # Entry point – main window
├── requirements.txt
├── README.md
├── core/
│   ├── __init__.py
│   ├── project.py        # ProjectManager
│   ├── dataset.py        # DatasetManager
│   ├── recipe.py         # RecipeEngine (all transformations)
│   ├── ml.py             # MLLab (train / evaluate)
│   └── notebook.py       # NotebookManager
├── gui/
│   ├── __init__.py
│   ├── flow_canvas.py    # Drag-and-drop pipeline canvas
│   ├── eda_panel.py      # EDA charts and stats
│   ├── ml_panel.py       # ML Lab GUI
│   └── themes.py         # Dark theme constants
└── utils/
    ├── __init__.py
    └── helpers.py         # File I/O, type detection, threading, JSON
```

---

## Installation

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# Optional – XGBoost support
pip install xgboost

# 2. tkinter is included with most Python distributions.
#    On Ubuntu/Debian, install if missing:
sudo apt-get install python3-tk
```

---

## How to Run

```bash
cd ML_week_04/dataiku
python main.py
```

The application opens a dark-themed window with:
- **Left sidebar**: project explorer (datasets, models, notebook)
- **Centre**: tabbed workspace (Flow, Datasets, EDA, ML Lab, Notebook, Prepare)
- **Right panel**: context-sensitive inspector (column stats, schema)
- **Bottom**: status bar

### Quick-start workflow

1. **File → Import Dataset** – load a CSV/Excel/JSON/Parquet file
2. **Datasets tab** – browse data, paginate, view column stats in the inspector
3. **EDA tab** – select the dataset, click **Auto EDA** for a report, or build custom charts
4. **ML Lab tab** – pick target/features, choose algorithm, tune hyper-parameters, click **Train**
5. **Prepare tab** – build a step-by-step transformation pipeline, preview changes, run
6. **Flow tab** – design a visual pipeline with drag-and-drop nodes
7. **Notebook tab** – write and execute Python code cells (shared namespace)
8. **Project → Save** – persist everything as JSON + Parquet

---

## Extending the ML Module

To add a new algorithm (e.g. **Gradient Boosting**):

1. Open [`core/ml.py`](core/ml.py)
2. Add the algorithm to the appropriate registry dict:

```python
from sklearn.ensemble import GradientBoostingClassifier

CLASSIFICATION_ALGORITHMS["Gradient Boosting"] = {
    "class": GradientBoostingClassifier,
    "defaults": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
    "params": {
        "n_estimators": {"type": "int", "min": 10, "max": 1000, "default": 100},
        "max_depth": {"type": "int", "min": 1, "max": 20, "default": 3},
        "learning_rate": {"type": "float", "min": 0.001, "max": 1.0, "default": 0.1},
    },
}
```

3. That's it! The algorithm will automatically appear in the ML Lab GUI with
   its hyperparameter controls.

---

## Tech Stack

| Layer | Technology |
|---|---|
| GUI | tkinter + ttk |
| Data | pandas, numpy |
| Visualisation | matplotlib, seaborn (embedded) |
| ML | scikit-learn (+ optional XGBoost) |
| File I/O | CSV, Excel (openpyxl), JSON, Parquet (pyarrow) |
| Persistence | JSON project files + Parquet data files |

---

## License

Educational / portfolio project.
