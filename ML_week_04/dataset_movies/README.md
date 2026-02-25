# Rotten Tomatoes Movies – ML Analysis

## Dataset
- **Source**: Rotten Tomatoes Movies (`Rotten Tomatoes Movies.csv`)
- **Rows**: 16,638 movies
- **Columns**: 17 (movie_title, movie_info, critics_consensus, rating, genre, directors, writers, cast, in_theaters_date, on_streaming_date, runtime_in_minutes, studio_name, tomatometer_status, tomatometer_rating, tomatometer_count, audience_rating, audience_count)

## ML Tasks
| # | Task | Type | Target |
|---|------|------|--------|
| 1 | Tomatometer Status Prediction | Multi-class Classification | tomatometer_status (Rotten / Fresh / Certified Fresh) |
| 2 | Audience Rating Prediction | Regression | audience_rating |
| 3 | Tomatometer Rating Prediction | Regression | tomatometer_rating |
| 4 | Movie Clustering | Unsupervised | K-Means on movie features |

## Additional Analysis
- Hyperparameter Tuning (GridSearchCV, RandomizedSearchCV)
- Cross-Validation & Learning Curves
- Feature Importance Analysis
- Voting & Stacking Ensembles

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook movies_ml_analysis.ipynb
```
