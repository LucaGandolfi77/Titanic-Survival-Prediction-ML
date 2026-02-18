# ✈️ Anomaly Detection & RUL Prediction — NASA CMAPSS Turbofan

## Overview
End-to-end **anomaly detection** notebook on the NASA CMAPSS FD001 Turbofan
Engine Degradation dataset. Two complementary approaches:

1. **Isolation Forest** — unsupervised statistical anomaly detection
2. **Autoencoder (PyTorch)** — deep-learning reconstruction-based detection

Both are evaluated on detection lead time: *how many cycles before failure
can we detect degradation?*

## Key Results
| Metric | Isolation Forest | Autoencoder |
|--------|-----------------|-------------|
| ROC-AUC | ~0.88 | ~0.92 |
| Avg lead time | Computed in notebook | Computed in notebook |
| Training | Seconds (sklearn) | ~1 min (PyTorch/MPS) |

## Dataset
**NASA CMAPSS FD001** — 100 turbofan engines, run-to-failure.
- 26 columns: engine ID, cycle, 3 operational settings, 21 sensors
- Auto-downloaded at runtime from GitHub

## Project Structure
```
anomaly_detection/
├── anomaly_detection.ipynb     # Main notebook
├── data/                       # Auto-downloaded .txt files
├── outputs/
│   ├── figures/                # All plots (.png)
│   ├── models/                 # isolation_forest.pkl, autoencoder.pth
│   └── results/                # anomaly_scores.csv, detection_report.html
├── requirements.txt
└── README.md
```

## Quick Start
```bash
cd ML_week_01/anomaly_detection
pip install -r requirements.txt
# Open anomaly_detection.ipynb in VSCode → Run All Cells
```

## Environment
- Python 3.11 (Anaconda) · Apple Silicon M1 (MPS) · VSCode + Jupyter
