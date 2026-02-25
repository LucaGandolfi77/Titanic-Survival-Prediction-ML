# Space Missions – ML Analysis

## Dataset
- **Source**: Space Missions (`Space+Missions/space_missions.csv`)
- **Rows**: 4,630 launches (1957–2022)
- **Columns**: 9 (Company, Location, Date, Time, Rocket, Mission, RocketStatus, Price, MissionStatus)

## ML Tasks
| # | Task | Type | Target |
|---|------|------|--------|
| 1 | Mission Success Classification | Binary Classification | Success vs Failure |
| 2 | Mission Price Regression | Regression | Price (millions USD) |
| 3 | Rocket Status Prediction | Binary Classification | Active vs Retired |
| 4 | Launch Clustering | Unsupervised | K-Means on launch features |

## Additional Analysis
- Hyperparameter Tuning (GridSearchCV, RandomizedSearchCV)
- Cross-Validation & Learning Curves
- Feature Importance Analysis
- Voting & Stacking Ensembles

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook space_ml_analysis.ipynb
```
