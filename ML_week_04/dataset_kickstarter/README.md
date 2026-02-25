# Kickstarter Projects — ML Analysis

## Dataset
- **kickstarter_projects.csv** — 374,853 crowdfunding projects (2009–2018)
- 15 categories, 159 subcategories, 22 countries
- Features: Goal, Pledged, Backers, Category, Country, Launch/Deadline dates, State

## ML Tasks
| # | Task | Type | Target |
|---|------|------|--------|
| 1 | Project Success Prediction | Binary Classification | Successful vs Failed |
| 2 | Pledged Amount Regression | Regression | USD pledged |
| 3 | Backer Count Regression | Regression | Number of backers |
| 4 | Project Category Clustering | Unsupervised | K-Means / DBSCAN |

## Models Used
- Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- AdaBoost, SVM (linear & rbf), KNN, Naive Bayes, MLP
- Voting & Stacking Ensembles
- Linear / Ridge / Lasso / DT / RF / GB Regressors

## Outputs
- `outputs/kickstarter_ml_report.html` — comprehensive HTML report
- `outputs/plots/` — all generated visualizations
