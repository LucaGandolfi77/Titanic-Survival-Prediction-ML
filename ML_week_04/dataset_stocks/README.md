# 📈 S&P 500 Stock Prices (2014–2017) – ML Analysis

## Dataset
- **Source**: S&P 500 Stock Prices 2014-2017.csv
- **Size**: 497,472 rows × 7 columns
- **Stocks**: 505 unique symbols, ~1,007 trading days each
- **Period**: 2014-01-02 to 2017-12-29
- **Features**: symbol, date, open, high, low, close, volume

## ML Tasks
| # | Task | Type | Target |
|---|------|------|--------|
| 1 | Price Direction Classification | Binary Classification | Next-day up / down |
| 2 | Daily Return Regression | Regression | Next-day return % |
| 3 | Volatility Classification | Multi-class Classification | Low / Medium / High volatility |
| 4 | Stock Clustering | Unsupervised | K-Means on stock-level features |

## Outputs
- `stocks_ml_analysis.ipynb` – Full analysis notebook
- `outputs/stocks_ml_report.html` – HTML report with embedded plots
- `outputs/plots/` – Individual PNG plots
