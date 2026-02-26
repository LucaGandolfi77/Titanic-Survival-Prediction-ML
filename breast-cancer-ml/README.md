# 🩺 Breast Cancer Classification — ML Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

A production-grade machine-learning pipeline that classifies breast tumours as
**malignant** or **benign** using the Wisconsin Breast Cancer dataset
(`sklearn.datasets.load_breast_cancer`).  
Four models are trained, cross-validated and compared end-to-end.

---

## 📊 Dataset Overview

| Property | Value |
|---|---|
| Source | `sklearn.datasets.load_breast_cancer()` |
| Samples | 569 |
| Features | 30 (float64) |
| Classes | malignant (0) — 212 &nbsp;/&nbsp; benign (1) — 357 |
| Feature groups | mean, SE and worst of: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension |

---

## 🗂 Project Structure

```
breast-cancer-ml/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── config/
│   └── config.yaml
├── data/
│   └── loader.py
├── eda/
│   └── explore.py
├── models/
│   ├── base_model.py
│   ├── logistic_regression.py
│   ├── random_forest.py
│   ├── svm.py
│   └── xgboost_model.py
├── pipeline/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── notebooks/
│   └── breast_cancer_analysis.ipynb
├── tests/
│   ├── test_loader.py
│   ├── test_models.py
│   └── test_pipeline.py
└── outputs/
    ├── models/          # serialised .pkl files
    ├── reports/         # classification reports, cv_results.csv
    └── plots/           # PNG figures
```

---

## ⚙️ Installation

```bash
# clone the repo
git clone https://github.com/<user>/breast-cancer-ml.git
cd breast-cancer-ml

# create a virtual-env (optional but recommended)
python -m venv .venv && source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Exploratory Data Analysis

```bash
python -m eda.explore
```

Generates class-distribution bar chart, correlation heatmap, and box-plots
in `outputs/plots/`.

### 2. Train all models

```bash
python -m pipeline.train
```

Runs 5-fold stratified cross-validation on four models and saves:

- trained models → `outputs/models/*.pkl`
- CV results → `outputs/reports/cv_results.csv`

### 3. Evaluate

```bash
python -m pipeline.evaluate
```

Prints classification reports and saves confusion-matrix heatmaps and a
combined ROC-curve comparison plot to `outputs/plots/`.

### 4. Predict

```bash
python -m pipeline.predict --model random_forest \
    --input "[17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]"
```

Returns the predicted class and probability.

---

## 📈 Results (example)

| Model | Accuracy | F1 (macro) | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.974 | 0.972 | 0.996 |
| Random Forest | 0.965 | 0.962 | 0.995 |
| SVM (RBF) | 0.974 | 0.972 | 0.997 |
| XGBoost | 0.965 | 0.962 | 0.995 |

> Actual numbers depend on hyperparameters in `config/config.yaml`.

---

## 🖼 EDA Samples

| Class Distribution | Correlation Heatmap |
|---|---|
| ![dist](outputs/plots/class_distribution.png) | ![heatmap](outputs/plots/correlation_heatmap.png) |

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/awesome`)
3. Commit your changes (`git commit -m 'Add awesome feature'`)
4. Push to the branch (`git push origin feature/awesome`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.
