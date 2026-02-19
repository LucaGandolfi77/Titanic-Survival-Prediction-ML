# 🧪 ML Playground — Interactive Machine Learning Dashboard

A **no-code, production-ready** web application built with **Streamlit** that lets
anyone upload a dataset, explore it visually, train multiple ML models with one
click, compare performance, and download trained models + predictions.

---

## ✨ Features

| Page | What it does |
|------|-------------|
| **🏠 Home** | Upload CSV/Excel or pick a sample dataset; select the target column |
| **📊 Data Explorer** | Descriptive stats, distributions, correlations, outlier detection |
| **🤖 Model Training** | Configure preprocessing, pick algorithms, tune hyperparameters, train |
| **📈 Results** | Metrics table, confusion matrix, ROC/PR curves, radar chart, SHAP |
| **🔮 Predictions** | Upload new data for batch inference or fill a single row manually |

### Supported Algorithms

**Classification:** Logistic Regression · Random Forest · Gradient Boosting ·
SVM · KNN · XGBoost · LightGBM · CatBoost

**Regression:** Linear Regression · Ridge · Random Forest · Gradient Boosting ·
XGBoost · LightGBM · CatBoost

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
make install          # or: pip install -r requirements.txt

# 2. Launch the dashboard
make run              # or: streamlit run app.py

# 3. Open in browser
#    → http://localhost:8501
```

---

## 📁 Project Structure

```
streamlit/
├── app.py                          # Main entry-point (Home page)
├── pages/                          # Streamlit multi-page app
│   ├── 1_📊_Data_Explorer.py
│   ├── 2_🤖_Model_Training.py
│   ├── 3_📈_Results.py
│   └── 4_🔮_Predictions.py
├── src/                            # Core library
│   ├── data/                       #   loader, preprocessing, splitter
│   ├── eda/                        #   statistics, distributions, correlations, outliers
│   ├── models/                     #   registry, trainer, evaluator, explainer
│   ├── visualization/              #   metrics_plots, feature_plots, decision_boundary, comparison
│   ├── ui/                         #   sidebar, data_upload, model_config, results_display
│   └── utils/                      #   session_state, caching, export
├── config/
│   └── model_config.yaml           # Default hyperparameter ranges
├── assets/
│   ├── styles.css                  # Custom CSS
│   ├── logo.png
│   └── sample_datasets/            # iris, wine, breast_cancer
├── outputs/                        # Saved models, predictions, reports
├── tests/                          # Pytest suite
├── .streamlit/config.toml          # Streamlit theme & server config
├── requirements.txt
├── setup.py
├── Makefile
└── README.md
```

---

## 🧰 Tech Stack

- **Frontend:** Streamlit 1.30+
- **ML:** scikit-learn · XGBoost · LightGBM · CatBoost · SHAP
- **Visualization:** Plotly · Matplotlib · Seaborn
- **Data:** Pandas · NumPy

---

## 🧪 Testing

```bash
make test
# or
python -m pytest tests/ -v
```

---

## 📝 How It Works

1. **Data Loading** – Upload CSV/Excel or choose a bundled sample.
   Auto-validates shape, drops empty rows/columns.
2. **EDA** – Descriptive stats, per-feature histograms/box-plots,
   correlation heatmap, outlier detection (IQR / Z-score).
3. **Preprocessing** – Imputation (mean/median/mode), encoding
   (one-hot/label/ordinal), scaling (standard/min-max).
4. **Training** – Select multiple models from the registry; configure
   hyperparameters via sliders; train in one click.
5. **Evaluation** – Accuracy, precision, recall, F1, ROC-AUC, log-loss
   (classification) or R², MAE, RMSE (regression).
6. **Visualisation** – Confusion matrices, ROC & PR curves, radar charts,
   feature importance, SHAP summary, decision boundaries.
7. **Export** – Download trained `.pkl` models and prediction CSVs.

---

## 📄 License

MIT — feel free to use, modify, and share.
