# FIFA World Cup & International Football — ML Analysis

## Dataset
- **international_matches.csv** — 17,769 international football matches (1872–2022)
- **world_cup_matches.csv** — 900 FIFA World Cup matches (1930–2018)
- **world_cups.csv** — 22 tournament summaries
- **2022_world_cup_squads.csv** — 831 player records
- **2022_world_cup_groups.csv** — 32 teams in 2022 groups
- **2022_world_cup_matches.csv** — 64 scheduled 2022 matches

## ML Tasks
| # | Task | Type | Target |
|---|------|------|--------|
| 1 | Match Outcome Prediction | Multi-class Classification | Home Win / Draw / Away Win |
| 2 | Total Goals Regression | Regression | Total goals per match |
| 3 | Player Position Classification | Multi-class Classification | GK / Defender / Midfielder / Forward |
| 4 | Country Performance Clustering | Unsupervised | K-Means / DBSCAN clusters |

## Models Used
- Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- AdaBoost, SVM (linear & rbf), KNN, Naive Bayes, MLP
- Voting & Stacking Ensembles
- Linear / Ridge / Lasso / DT / RF / GB Regressors

## Outputs
- `outputs/football_ml_report.html` — comprehensive HTML report
- `outputs/plots/` — all generated visualizations
