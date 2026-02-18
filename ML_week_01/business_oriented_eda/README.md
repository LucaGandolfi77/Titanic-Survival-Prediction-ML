# 📞 Telco Customer Churn — Analysis & Prediction

## Overview
End-to-end **binary classification** notebook on the IBM Telco Customer Churn
dataset. The project combines technical ML rigour with **business-oriented
storytelling** — every chart and metric is translated into actionable insights
for non-technical stakeholders.

## Key Business Results
| KPI | Value |
|-----|-------|
| Overall churn rate | ~26 % |
| Best model | XGBoost (tuned) |
| ROC-AUC (test) | ~0.85 |
| Revenue at risk (monthly) | ~$139 k |
| Estimated annual ROI of retention campaign | Computed in Section 8 |

## Project Structure
```
business_oriented_eda/
├── churn_analysis.ipynb        # Main notebook (12 sections)
├── data/
│   └── telco_churn.csv         # Auto-downloaded at runtime
├── outputs/
│   ├── figures/                # All saved plots (.png)
│   ├── churn_report.html       # Business summary report
│   └── churn_model.pkl         # Saved best model pipeline
├── requirements.txt
└── README.md
```

## Notebook Sections
| # | Section | Focus |
|---|---------|-------|
| 0 | Configuration & Imports | Constants, libraries, directories |
| 1 | Load & First Inspection | Data quality, churn rate KPI |
| 2 | Business KPI Dashboard | 8 C-suite ready charts |
| 3 | Customer Segmentation EDA | 5 deep-dive analyses |
| 4 | Feature Engineering | 8 new business features |
| 5 | Class Imbalance Strategy | Why `class_weight="balanced"` |
| 6 | Pipeline + Model Training | 5-model CV comparison |
| 7 | Hyperparameter Tuning | RandomizedSearchCV |
| 8 | Final Evaluation & ROI | ML metrics + business ROI |
| 9 | Feature Importance | Explainability + risk tiers |
| 10 | Save Model & Report | .pkl export + HTML report |
| 11 | Key Takeaways | Business + technical summary |

## Quick Start
```bash
cd ML_week_01/business_oriented_eda
pip install -r requirements.txt
# Open churn_analysis.ipynb in VSCode and Run All Cells
```

## Environment
- Python 3.11 (Anaconda)
- Apple Silicon M1 (MPS)
- VSCode + Jupyter extension
