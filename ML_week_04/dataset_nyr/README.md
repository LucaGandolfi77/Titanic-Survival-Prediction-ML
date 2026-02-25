# New Year's Resolutions – ML & NLP Analysis

## Dataset
- **Source**: New Year's Resolutions Tweets (`New_years_resolutions.csv`)
- **Rows**: 4,723 tweets
- **Columns**: 10 (tweet_created, tweet_text, tweet_category, tweet_topics, tweet_location, tweet_state, tweet_region, user_timezone, user_gender, retweet_count)

## ML & NLP Tasks
| # | Task | Type | Target / Method |
|---|------|------|-----------------|
| 1 | Tweet Category Classification | Multi-class Classification | 10 categories |
| 2 | Gender Prediction from Tweet | Binary Classification | male / female |
| 3 | Sentiment Analysis (TextBlob) | NLP | Polarity & Subjectivity |
| 4 | Sentiment Analysis (BERT) | Deep Learning NLP | HuggingFace pipeline |
| 5 | Tweet Clustering | Unsupervised | K-Means on TF-IDF |

## Additional Analysis
- TF-IDF text vectorization
- BERT sentiment classification
- Hyperparameter Tuning (GridSearchCV, RandomizedSearchCV)
- Cross-Validation & Learning Curves
- Feature Importance Analysis
- Voting & Stacking Ensembles

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook nyr_ml_analysis.ipynb
```
