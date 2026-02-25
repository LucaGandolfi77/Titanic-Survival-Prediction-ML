# 🍷 Wine Reviews (130k) – ML Analysis

## Dataset
- **Source**: winemag-data-130k-v2.csv
- **Size**: 129,971 reviews × 14 columns
- **Countries**: 43 unique (US 54k, France 22k, Italy 19k)
- **Varieties**: 707 unique (Pinot Noir, Chardonnay, Cabernet Sauvignon top 3)
- **Points**: 80–100 (mean 88.4), **Price**: $4–$3,300 (mean $35.4)

## ML Tasks
| # | Task | Type | Target |
|---|------|------|--------|
| 1 | Wine Quality Classification | Multi-class Classification | Points binned into Low/Medium/High/Premium |
| 2 | Wine Price Regression | Regression | Price (USD) |
| 3 | Variety Classification | Multi-class Classification | Top 10 grape varieties |
| 4 | Wine Clustering | Unsupervised | K-Means on wine features |

## Outputs
- `wine_ml_analysis.ipynb` – Full analysis notebook
- `outputs/wine_ml_report.html` – HTML report with embedded plots
- `outputs/plots/` – Individual PNG plots
