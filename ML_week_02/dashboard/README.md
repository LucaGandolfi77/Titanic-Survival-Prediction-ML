# Explainable AI Dashboard

Production-ready **Streamlit** dashboard for model interpretability, fairness analysis, and automated reporting.

## Features

| Stage | Capabilities |
|-------|-------------|
| **🏠 Overview** | Load datasets (built-in or upload), train/load models, performance metrics |
| **🌐 Global Explanations** | SHAP importance, beeswarm plots, dependence plots, feature interactions (H-statistic), PDP/ICE |
| **🔬 Local Explanations** | SHAP waterfall & force plots, LIME explanations, what-if analysis, counterfactual search, SHAP vs LIME comparison |
| **⚖️ Fairness Analysis** | Demographic parity, equal opportunity, disparate impact (80% rule), equalized odds, bias heatmaps, mitigation recommendations, threshold optimisation, re-weighting |
| **📊 Reports** | Executive summary (HTML), technical report (HTML/TXT), one-click download |

## Tech Stack

- **Streamlit 1.30+** — dashboard framework
- **SHAP 0.44+** — global & local model explanations
- **LIME 0.2+** — alternative local explanations
- **Fairlearn 0.9+** — fairness metrics & bias detection
- **scikit-learn / XGBoost / LightGBM / CatBoost** — model support
- **Plotly** — interactive visualisations
- **Jinja2** — HTML report templating


## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate sample datasets & pre-trained models
python _generate.py

# 3. Run the dashboard
streamlit run app.py --server.port 8502

# 4. Run tests
pytest tests/ -v
```

Or with Makefile:
```bash
make install
make generate
make run
make test
```

Or with Docker:
```bash
docker build -t xai-dashboard .
docker run -p 8502:8502 xai-dashboard
```


## Project Structure

```
dashboard/
├── app.py                          # Main entry point
├── _generate.py                    # Generate sample data & models
├── requirements.txt
├── Makefile
├── Dockerfile
├── .streamlit/config.toml
├── config/dashboard_config.yaml
├── assets/styles.css
│
├── pages/
│   ├── 1_overview.py               # Data & model setup
│   ├── 2_global_explanations.py    # SHAP global analysis
│   ├── 3_local_explanations.py     # SHAP + LIME local
│   ├── 4_fairness_analysis.py      # Bias detection & mitigation
│   └── 5_reports.py                # Report generation
│
├── src/
│   ├── models/
│   │   ├── loader.py               # Load/save models
│   │   ├── predictor.py            # Unified prediction wrapper
│   │   └── metadata.py             # Model info & metrics
│   │
│   ├── explainability/
│   │   ├── shap_explainer.py       # SHAP (auto algorithm selection)
│   │   ├── lime_explainer.py       # LIME wrapper
│   │   ├── pdp.py                  # Partial dependence & ICE
│   │   ├── counterfactuals.py      # What-if & counterfactual search
│   │   └── feature_interactions.py # H-statistic interactions
│   │
│   ├── fairness/
│   │   ├── metrics.py              # DP, EO, DI, EOd metrics
│   │   ├── bias_detector.py        # Auto bias scan (PASS/WARN/FAIL)
│   │   └── mitigation.py           # Re-weighting, threshold optimisation
│   │
│   ├── visualization/
│   │   ├── shap_plots.py           # SHAP-specific Plotly charts
│   │   ├── lime_plots.py           # LIME charts
│   │   ├── fairness_plots.py       # Fairness charts & gauges
│   │   └── custom_plots.py         # General plots (CM, ROC, radar, etc.)
│   │
│   ├── reporting/
│   │   ├── summary_generator.py    # Build report data dicts
│   │   ├── pdf_exporter.py         # HTML rendering & PDF export
│   │   └── templates/
│   │       ├── executive_summary.html
│   │       └── technical_report.html
│   │
│   └── utils/
│       ├── data_loader.py          # Dataset loading & preparation
│       ├── session_state.py        # Streamlit state management
│       └── helpers.py              # Config, detection, colour helpers
│
├── data/
│   ├── credit_risk_sample.csv      # Generated sample dataset
│   ├── medical_sample.csv          # Generated sample dataset
│   └── protected_attributes.json   # Protected attribute definitions
│
├── models/                         # Pre-trained model .pkl files
│
├── tests/
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_explainability.py
│   ├── test_fairness.py
│   └── test_reporting.py
│
├── notebooks/
│   └── exploration.ipynb           # Interactive exploration notebook
│
└── outputs/reports/                # Generated reports
```


## Supported Models

| Framework | Types |
|-----------|-------|
| scikit-learn | RandomForest, GradientBoosting, LogisticRegression, SVM, DecisionTree, k-NN, MLP, AdaBoost |
| XGBoost | XGBClassifier, XGBRegressor |
| LightGBM | LGBMClassifier, LGBMRegressor |
| CatBoost | CatBoostClassifier, CatBoostRegressor |

SHAP automatically selects the appropriate explainer (Tree/Linear/Kernel) based on the model type.


## Fairness Metrics

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Demographic Parity Difference | ≤ 0.10 | Difference in positive prediction rates |
| Equal Opportunity Difference | ≤ 0.10 | Difference in true positive rates |
| Disparate Impact Ratio | ≥ 0.80 | 80% rule (EEOC) |
| Equalised Odds Difference | ≤ 0.10 | Max of TPR and FPR differences |


## Configuration

Edit `config/dashboard_config.yaml` to customise:
- SHAP background samples and max display features
- LIME number of features and kernel width
- PDP grid resolution
- Fairness thresholds
- Report output directory


## License

MIT
