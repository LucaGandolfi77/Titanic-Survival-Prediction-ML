# PyWeka – Machine Learning Explorer

A complete **Machine Learning workbench** inspired by [Weka](https://www.cs.waikato.ac.nz/ml/weka/), available in two versions:

| Version | Stack | Run |
|---------|-------|-----|
| 🖥️ **Desktop** | Python · Tkinter · scikit-learn · XGBoost | `python app.py` |
| 🌐 **Web** | HTML · CSS · Vanilla JS (zero back-end) | Open `web/index.html` in any browser |

---

## ✨ Features at a Glance

| Tab | Desktop | Web |
|-----|---------|-----|
| **Preprocess** | Load CSV/TSV/Excel/ARFF, handle missing values, encode categoricals, scale, remove outliers, undo | Drag-and-drop CSV, same preprocessing operations |
| **Classify** | 13 algorithms, % split / k-fold CV / training set eval, confusion matrix | 5 algorithms (KNN, NB, DT, Logistic Reg, Random Forest), same eval modes, interactive confusion matrix |
| **Regression** | 13 algorithms, R²/RMSE/MAE, actual-vs-predicted plot | 4 algorithms (Linear, Ridge, KNN, DT), same metrics, Plotly scatter |
| **Cluster** | K-Means, DBSCAN, Agglomerative, Mean Shift, Mini-Batch | K-Means with K-Means++ init, elbow plot, silhouette |
| **Visualize** | 9 plot types (histogram, scatter, box, heatmap, pairplot, violin, bar, distribution grid) | 6 interactive Plotly chart types with hue support |
| **Select Attributes** | KBest, RF importance, RFE, correlation | Correlation, variance, RF importance |
| **Associate** | Apriori rules (mlxtend) | — |

### Built-in Sample Datasets
Both versions include **Iris**, **Wine**, and **Diabetes** samples ready to load with one click.

---

## 🖥️ Desktop App – Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch
python app.py

# Or open a dataset directly
python app.py path/to/data.csv
```

### Requirements
- Python 3.10+
- Tkinter (ships with Python on most systems; macOS: `brew install python-tk@3.xx`)
- See `requirements.txt` for packages

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Cmd/Ctrl + O | Open dataset |
| Cmd/Ctrl + S | Save dataset |
| Cmd/Ctrl + Z | Undo |
| Cmd/Ctrl + Q | Quit |

---

## 🌐 Web Version – Quick Start

The `web/` folder contains a **fully self-contained** browser app — no server, no build step, no installation.

```
# Option 1 – just open the file
open web/index.html          # macOS
xdg-open web/index.html      # Linux
start web/index.html          # Windows

# Option 2 – serve locally (avoids CORS for file loading)
cd web && python -m http.server 8000
# then visit http://localhost:8000
```

### Embedding in Your Website

Copy the three files (`index.html`, `style.css`, `app.js`) into your site.
The only external dependencies are loaded from CDN:

| Library | Purpose |
|---------|---------|
| [Papa Parse](https://www.papaparse.com/) | CSV parsing |
| [Plotly.js](https://plotly.com/javascript/) | Interactive charts |
| [Google Fonts](https://fonts.google.com/) | Inter + JetBrains Mono |

### ML Algorithms (100 % vanilla JS, from scratch)

**Classification:** KNN · Gaussian Naive Bayes · Decision Tree (CART) · Logistic Regression · Random Forest

**Regression:** Linear Regression · Ridge Regression · KNN Regressor · Decision Tree Regressor

**Clustering:** K-Means (K-Means++ initialization)

**Feature Selection:** Correlation · Variance · RF Importance

---

## 📂 Project Structure

```
weka/
├── app.py                  # Desktop entry point
├── requirements.txt
├── README.md
├── core/
│   ├── __init__.py
│   ├── data_manager.py     # Dataset I/O, undo, introspection
│   ├── preprocessor.py     # Preprocessing operations
│   └── evaluator.py        # ML model training & evaluation
├── ui/
│   ├── __init__.py
│   ├── main_window.py       # Main window + menu + tabs
│   ├── preprocess_tab.py    # Preprocess panel
│   ├── classify_tab.py      # Classification panel
│   ├── regression_tab.py    # Regression panel
│   ├── cluster_tab.py       # Clustering panel
│   ├── associate_tab.py     # Association rules panel
│   ├── select_tab.py        # Feature selection panel
│   ├── visualize_tab.py     # Visualization panel
│   └── widgets.py           # Reusable UI components
└── web/
    ├── index.html           # Web app – HTML structure
    ├── style.css            # Web app – dark theme styles
    └── app.js               # Web app – ML engine + UI (vanilla JS)
```
