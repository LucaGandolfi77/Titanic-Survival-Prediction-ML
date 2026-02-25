#!/usr/bin/env python3
"""Generate nyr_ml_analysis.ipynb – New Year's Resolutions ML + NLP + BERT."""
import json, pathlib

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip().split("\n")}

def code(src):
    lines = src.strip().split("\n")
    return {"cell_type": "code", "metadata": {}, "source": lines,
            "execution_count": None, "outputs": []}

cells = []

# ── 0. Title ──────────────────────────────────────────────────────────────
cells.append(md("""# 🎆 New Year's Resolutions – ML & NLP Analysis
**Dataset**: 4,723 tweets about New Year's Resolutions with categories, regions, gender, and text.

| # | Task | Type | Target / Method |
|---|------|------|-----------------|
| 1 | Tweet Category Classification | Multi-class (10 classes) | TF-IDF + ML models |
| 2 | Gender Prediction | Binary Classification | TF-IDF + metadata features |
| 3 | Sentiment Analysis (TextBlob) | NLP | Polarity & subjectivity scores |
| 4 | Sentiment Analysis (BERT) | Deep Learning | HuggingFace distilbert-sst2 |
| 5 | Tweet Clustering | Unsupervised | K-Means on TF-IDF vectors |
"""))

# ── 1. Imports ────────────────────────────────────────────────────────────
cells.append(md("## 1 · Imports"))
cells.append(code("""import warnings, os, pathlib, re, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     RandomizedSearchCV, cross_val_score,
                                     learning_curve, StratifiedKFold)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, silhouette_score)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, VotingClassifier, StackingClassifier)
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack, csr_matrix
from xgboost import XGBClassifier

from textblob import TextBlob

import jinja2, base64
from io import BytesIO

SEED = 42
PLOT_DIR = pathlib.Path("outputs/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", palette="viridis")
print("✅ Imports OK")
"""))

# ── 2. Load data ──────────────────────────────────────────────────────────
cells.append(md("## 2 · Load & Prepare Data"))
cells.append(code("""df = pd.read_csv("New_years_resolutions.csv", encoding="latin-1")

# Fix BOM column name
df.columns = [c.replace("\\ufeff", "") for c in df.columns]
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Parse date
df["tweet_created"] = pd.to_datetime(df["tweet_created"], errors="coerce")
df["tweet_hour"] = df["tweet_created"].dt.hour
df["tweet_day"] = df["tweet_created"].dt.day

# Text cleaning function
def clean_text(text):
    text = re.sub(r"http\\S+|www\\S+|https\\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\\w+", "", text)
    text = re.sub(r"#(\\w+)", r"\\1", text)  # keep hashtag text, remove #
    text = re.sub(r"[^a-zA-Z\\s]", "", text)
    return text.lower().strip()

df["clean_text"] = df["tweet_text"].apply(clean_text)
df["text_length"] = df["clean_text"].str.len()
df["word_count"] = df["clean_text"].str.split().str.len()

# Encode categorical
le_region = LabelEncoder()
df["region_enc"] = le_region.fit_transform(df["tweet_region"])

le_gender = LabelEncoder()
df["gender_enc"] = le_gender.fit_transform(df["user_gender"])

le_cat = LabelEncoder()
df["category_enc"] = le_cat.fit_transform(df["tweet_category"])

print(f"\\nCategories ({len(le_cat.classes_)}):")
for c in le_cat.classes_:
    print(f"  {c}: {(df['tweet_category'] == c).sum()}")

print(f"\\nGender: {df['user_gender'].value_counts().to_dict()}")
print(f"Regions: {df['tweet_region'].value_counts().to_dict()}")
df.head(3)
"""))

# ── 3. EDA ────────────────────────────────────────────────────────────────
cells.append(md("## 3 · Exploratory Data Analysis"))
cells.append(code("""fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1 – Category distribution
df["tweet_category"].value_counts().plot.bar(ax=axes[0, 0],
    color=sns.color_palette("viridis", df["tweet_category"].nunique()))
axes[0, 0].set_title("Tweet Category Distribution")
axes[0, 0].set_ylabel("Count"); axes[0, 0].tick_params(axis="x", rotation=45)

# 2 – Gender distribution
df["user_gender"].value_counts().plot.pie(ax=axes[0, 1], autopct="%1.1f%%",
    colors=["#06b6d4", "#f97316"], startangle=90)
axes[0, 1].set_title("Gender Distribution"); axes[0, 1].set_ylabel("")

# 3 – Region distribution
df["tweet_region"].value_counts().plot.bar(ax=axes[0, 2],
    color=sns.color_palette("mako", 4))
axes[0, 2].set_title("Region Distribution")
axes[0, 2].set_ylabel("Count"); axes[0, 2].tick_params(axis="x", rotation=0)

# 4 – Word count distribution
df["word_count"].hist(bins=30, ax=axes[1, 0], color="steelblue", edgecolor="white")
axes[1, 0].set_title("Word Count Distribution")
axes[1, 0].set_xlabel("Words"); axes[1, 0].set_ylabel("Count")

# 5 – Category by gender
cat_gender = pd.crosstab(df["tweet_category"], df["user_gender"])
cat_gender.plot.barh(ax=axes[1, 1], stacked=True, color=["#06b6d4", "#f97316"])
axes[1, 1].set_title("Categories by Gender")
axes[1, 1].set_xlabel("Count"); axes[1, 1].legend(title="Gender")

# 6 – Category by region
cat_region = pd.crosstab(df["tweet_category"], df["tweet_region"])
cat_region.plot.barh(ax=axes[1, 2], stacked=True, colormap="viridis")
axes[1, 2].set_title("Categories by Region")
axes[1, 2].set_xlabel("Count"); axes[1, 2].legend(title="Region")

plt.tight_layout()
plt.savefig(PLOT_DIR / "eda_overview.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ EDA plots saved")
"""))

# ── 4. TextBlob Sentiment ────────────────────────────────────────────────
cells.append(md("## 4 · Sentiment Analysis – TextBlob"))
cells.append(code("""print("Running TextBlob sentiment analysis on all tweets...")
df["tb_polarity"] = df["tweet_text"].apply(lambda t: TextBlob(t).sentiment.polarity)
df["tb_subjectivity"] = df["tweet_text"].apply(lambda t: TextBlob(t).sentiment.subjectivity)

# Classify sentiment
df["tb_sentiment"] = pd.cut(df["tb_polarity"], bins=[-1.01, -0.05, 0.05, 1.01],
                            labels=["Negative", "Neutral", "Positive"])

print(f"Sentiment distribution:\\n{df['tb_sentiment'].value_counts()}")
print(f"\\nPolarity stats:\\n{df['tb_polarity'].describe()}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1 – Sentiment distribution
df["tb_sentiment"].value_counts().plot.bar(ax=axes[0],
    color=["#ef4444", "#94a3b8", "#22c55e"])
axes[0].set_title("TextBlob Sentiment Distribution")
axes[0].set_ylabel("Count"); axes[0].tick_params(axis="x", rotation=0)

# 2 – Polarity histogram
df["tb_polarity"].hist(bins=50, ax=axes[1], color="teal", edgecolor="white")
axes[1].set_title("Polarity Distribution")
axes[1].set_xlabel("Polarity"); axes[1].axvline(0, color="red", linestyle="--")

# 3 – Sentiment by category
sent_cat = pd.crosstab(df["tweet_category"], df["tb_sentiment"])
sent_cat_pct = sent_cat.div(sent_cat.sum(axis=1), axis=0)
sent_cat_pct.plot.barh(ax=axes[2], stacked=True,
    color=["#ef4444", "#94a3b8", "#22c55e"])
axes[2].set_title("Sentiment by Category (TextBlob)")
axes[2].set_xlabel("Proportion"); axes[2].legend(title="Sentiment", bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.savefig(PLOT_DIR / "textblob_sentiment.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ TextBlob sentiment plots saved")
"""))

# ── 5. BERT Sentiment ────────────────────────────────────────────────────
cells.append(md("## 5 · Sentiment Analysis – BERT (DistilBERT)"))
cells.append(code("""from transformers import pipeline as hf_pipeline

print("Loading DistilBERT sentiment pipeline (distilbert-base-uncased-finetuned-sst-2-english)...")
bert_sentiment = hf_pipeline("sentiment-analysis",
                             model="distilbert-base-uncased-finetuned-sst-2-english",
                             device=-1,  # CPU
                             truncation=True, max_length=512)

# Run BERT on all tweets (batch for speed)
print(f"Running BERT sentiment on {len(df)} tweets...")
t0 = time.time()
BATCH = 64
bert_labels, bert_scores = [], []
texts = df["tweet_text"].tolist()
for i in range(0, len(texts), BATCH):
    batch = texts[i:i+BATCH]
    # Truncate long texts
    batch = [t[:500] for t in batch]
    results = bert_sentiment(batch)
    for r in results:
        bert_labels.append(r["label"])
        bert_scores.append(r["score"])
    if (i // BATCH) % 10 == 0:
        print(f"  Processed {min(i+BATCH, len(texts))}/{len(texts)}...")

df["bert_label"] = bert_labels
df["bert_score"] = bert_scores
elapsed = time.time() - t0
print(f"\\n✅ BERT sentiment done in {elapsed:.1f}s")
print(f"BERT sentiment distribution:\\n{df['bert_label'].value_counts()}")

# Compare TextBlob vs BERT
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1 – BERT sentiment distribution
df["bert_label"].value_counts().plot.bar(ax=axes[0],
    color=["#22c55e", "#ef4444"])
axes[0].set_title("BERT Sentiment Distribution")
axes[0].set_ylabel("Count"); axes[0].tick_params(axis="x", rotation=0)

# 2 – BERT confidence scores
df["bert_score"].hist(bins=50, ax=axes[1], color="purple", edgecolor="white")
axes[1].set_title("BERT Confidence Score Distribution")
axes[1].set_xlabel("Confidence")

# 3 – BERT sentiment by category
bert_cat = pd.crosstab(df["tweet_category"], df["bert_label"])
bert_cat_pct = bert_cat.div(bert_cat.sum(axis=1), axis=0)
bert_cat_pct.plot.barh(ax=axes[2], stacked=True,
    color=["#ef4444", "#22c55e"])
axes[2].set_title("BERT Sentiment by Category")
axes[2].set_xlabel("Proportion"); axes[2].legend(title="Label", bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.savefig(PLOT_DIR / "bert_sentiment.png", dpi=150, bbox_inches="tight")
plt.show()

# Sentiment comparison summary
print("\\n=== TextBlob vs BERT Comparison ===")
tb_pos_pct = (df["tb_sentiment"] == "Positive").mean() * 100
tb_neg_pct = (df["tb_sentiment"] == "Negative").mean() * 100
tb_neu_pct = (df["tb_sentiment"] == "Neutral").mean() * 100
bert_pos_pct = (df["bert_label"] == "POSITIVE").mean() * 100
bert_neg_pct = (df["bert_label"] == "NEGATIVE").mean() * 100
print(f"TextBlob: Positive={tb_pos_pct:.1f}%, Neutral={tb_neu_pct:.1f}%, Negative={tb_neg_pct:.1f}%")
print(f"BERT:     Positive={bert_pos_pct:.1f}%, Negative={bert_neg_pct:.1f}%")
print("✅ BERT sentiment plots saved")
"""))

# ── 6. Prepare classification features ────────────────────────────────────
cells.append(md("## 6 · Prepare Features (TF-IDF + Metadata)"))
cells.append(code("""# TF-IDF on cleaned text
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")
X_tfidf = tfidf.fit_transform(df["clean_text"])
print(f"TF-IDF shape: {X_tfidf.shape}")

# Metadata features
meta_cols = ["text_length", "word_count", "region_enc", "gender_enc",
             "tweet_hour", "tweet_day", "tb_polarity", "tb_subjectivity", "bert_score"]
X_meta = df[meta_cols].fillna(0).values
scaler_meta = StandardScaler()
X_meta_s = scaler_meta.fit_transform(X_meta)

# Combine TF-IDF + metadata
X_combined = hstack([X_tfidf, csr_matrix(X_meta_s)])
print(f"Combined feature matrix: {X_combined.shape}")

# Targets
y_cat = df["category_enc"].values
y_gender = df["gender_enc"].values

print(f"Category classes: {le_cat.classes_}")
print(f"Gender classes: {le_gender.classes_}")
"""))

# ── 7. Category Classification ──────────────────────────────────────────
cells.append(md("## 7 · Tweet Category Classification (10 Models)"))
cells.append(code("""X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_combined, y_cat, test_size=0.2, random_state=SEED, stratify=y_cat
)

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Multinomial NB": MultinomialNB(alpha=0.1),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=SEED),
    "AdaBoost": AdaBoostClassifier(n_estimators=150, random_state=SEED),
    "XGBoost": XGBClassifier(n_estimators=200, eval_metric="mlogloss", random_state=SEED, n_jobs=-1),
    "Linear SVM": LinearSVC(max_iter=2000, random_state=SEED),
    "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=SEED),
}

# MultinomialNB needs non-negative input – use TF-IDF only
X_train_tfidf_only = X_train_c[:, :3000]
X_test_tfidf_only = X_test_c[:, :3000]

clf_results = {}
for name, model in classifiers.items():
    print(f"Training {name}...", end=" ")
    if name == "Multinomial NB":
        model.fit(X_train_tfidf_only, y_train_c)
        y_pred = model.predict(X_test_tfidf_only)
    else:
        model.fit(X_train_c, y_train_c)
        y_pred = model.predict(X_test_c)
    acc = accuracy_score(y_test_c, y_pred)
    f1 = f1_score(y_test_c, y_pred, average="weighted")
    clf_results[name] = {"accuracy": acc, "f1": f1, "model": model, "y_pred": y_pred}
    print(f"Acc={acc:.4f}  F1={f1:.4f}")

best_clf_name = max(clf_results, key=lambda k: clf_results[k]["f1"])
print(f"\\n🏆 Best classifier: {best_clf_name} (F1={clf_results[best_clf_name]['f1']:.4f})")

# Bar chart
fig, ax = plt.subplots(figsize=(12, 6))
names = list(clf_results.keys())
accs = [clf_results[n]["accuracy"] for n in names]
f1s = [clf_results[n]["f1"] for n in names]
x = np.arange(len(names))
ax.bar(x - 0.2, accs, 0.4, label="Accuracy", color="steelblue")
ax.bar(x + 0.2, f1s, 0.4, label="F1 (weighted)", color="coral")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right")
ax.set_ylim(0, 1); ax.set_title("Category Classification – Model Comparison")
ax.legend(); plt.tight_layout()
plt.savefig(PLOT_DIR / "category_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Category model comparison saved")
"""))

# ── 8. Gender Classification ──────────────────────────────────────────────
cells.append(md("## 8 · Gender Prediction from Tweets"))
cells.append(code("""X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
    X_combined, y_gender, test_size=0.2, random_state=SEED, stratify=y_gender
)

gender_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=SEED),
    "XGBoost": XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=SEED, n_jobs=-1),
    "Linear SVM": LinearSVC(max_iter=2000, random_state=SEED),
    "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=SEED),
}

gender_results = {}
for name, model in gender_models.items():
    print(f"Training {name}...", end=" ")
    model.fit(X_train_g, y_train_g)
    y_pred = model.predict(X_test_g)
    acc = accuracy_score(y_test_g, y_pred)
    f1 = f1_score(y_test_g, y_pred, average="weighted")
    gender_results[name] = {"accuracy": acc, "f1": f1, "model": model, "y_pred": y_pred}
    print(f"Acc={acc:.4f}  F1={f1:.4f}")

best_gender_name = max(gender_results, key=lambda k: gender_results[k]["f1"])
print(f"\\n🏆 Best gender predictor: {best_gender_name} (F1={gender_results[best_gender_name]['f1']:.4f})")

fig, ax = plt.subplots(figsize=(10, 5))
names_g = list(gender_results.keys())
f1s_g = [gender_results[n]["f1"] for n in names_g]
ax.barh(names_g, f1s_g, color=sns.color_palette("rocket", len(names_g)))
ax.set_title("Gender Prediction – F1 Scores"); ax.set_xlabel("F1 (weighted)")
ax.set_xlim(0, 1)
for i, v in enumerate(f1s_g):
    ax.text(v + 0.01, i, f"{v:.4f}", va="center")
plt.tight_layout()
plt.savefig(PLOT_DIR / "gender_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Gender model comparison saved")
"""))

# ── 9. Clustering ─────────────────────────────────────────────────────────
cells.append(md("## 9 · Tweet Clustering (TF-IDF + K-Means)"))
cells.append(code("""# Reduce TF-IDF dimensionality for clustering
svd = TruncatedSVD(n_components=50, random_state=SEED)
X_svd = svd.fit_transform(X_tfidf)
print(f"SVD explained variance: {svd.explained_variance_ratio_.sum():.4f}")

K_range = range(2, 12)
inertias, sils = [], []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km.fit_predict(X_svd)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_svd, labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(list(K_range), inertias, "bo-"); ax1.set_title("Elbow Method")
ax1.set_xlabel("k"); ax1.set_ylabel("Inertia")
ax2.plot(list(K_range), sils, "rs-"); ax2.set_title("Silhouette Score")
ax2.set_xlabel("k"); ax2.set_ylabel("Score")
plt.tight_layout()
plt.savefig(PLOT_DIR / "elbow_silhouette.png", dpi=150, bbox_inches="tight")
plt.show()

best_k = list(K_range)[np.argmax(sils)]
print(f"Best k={best_k}, silhouette={max(sils):.4f}")

km_final = KMeans(n_clusters=best_k, random_state=SEED, n_init=10)
df["cluster"] = km_final.fit_predict(X_svd)

# Cluster vs actual category crosstab
clust_cat = pd.crosstab(df["cluster"], df["tweet_category"])
fig, ax = plt.subplots(figsize=(14, 6))
clust_cat.plot.bar(ax=ax, stacked=True, colormap="viridis")
ax.set_title(f"Cluster vs Tweet Category (k={best_k})")
ax.set_xlabel("Cluster"); ax.set_ylabel("Count")
ax.legend(title="Category", bbox_to_anchor=(1.05, 1), fontsize=8)
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig(PLOT_DIR / "clustering_results.png", dpi=150, bbox_inches="tight")
plt.show()

# Top words per cluster
print("\\nTop words per cluster:")
feature_names = np.array(tfidf.get_feature_names_out())
order_centroids = km_final.cluster_centers_.argsort()[:, ::-1]
# We need to project centroids back; use SVD components
centroids_tfidf = km_final.cluster_centers_ @ svd.components_
order_centroids_full = centroids_tfidf.argsort()[:, ::-1]
for i in range(best_k):
    top_words = [feature_names[j] for j in order_centroids_full[i, :10]]
    print(f"  Cluster {i}: {', '.join(top_words)}")

print("✅ Clustering plots saved")
"""))

# ── 10. Hyperparameter Tuning ─────────────────────────────────────────────
cells.append(md("## 10 · Hyperparameter Tuning"))
cells.append(code("""# GridSearchCV – Logistic Regression
print("GridSearchCV on Logistic Regression (category)...")
lr_grid = {
    "C": [0.1, 1.0, 10.0],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"],
}
gs_lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=SEED),
                     lr_grid, cv=3, scoring="f1_weighted", n_jobs=-1)
gs_lr.fit(X_train_c, y_train_c)
print(f"  Best params: {gs_lr.best_params_}")
print(f"  Best CV F1:  {gs_lr.best_score_:.4f}")

# RandomizedSearchCV – Random Forest
print("\\nRandomizedSearchCV on Random Forest (category)...")
rf_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2"],
}
rs_rf = RandomizedSearchCV(RandomForestClassifier(random_state=SEED, n_jobs=-1),
                           rf_dist, n_iter=15, cv=3, scoring="f1_weighted",
                           random_state=SEED, n_jobs=-1)
rs_rf.fit(X_train_c, y_train_c)
print(f"  Best params: {rs_rf.best_params_}")
print(f"  Best CV F1:  {rs_rf.best_score_:.4f}")

# Evaluate tuned models
for label, model in [("Tuned LR (Grid)", gs_lr.best_estimator_),
                     ("Tuned RF (Random)", rs_rf.best_estimator_)]:
    y_pred = model.predict(X_test_c)
    acc = accuracy_score(y_test_c, y_pred)
    f1 = f1_score(y_test_c, y_pred, average="weighted")
    clf_results[label] = {"accuracy": acc, "f1": f1, "model": model, "y_pred": y_pred}
    print(f"  {label}: Acc={acc:.4f}  F1={f1:.4f}")

print("\\n✅ Hyperparameter tuning complete")
"""))

# ── 11. CV + Confusion + Learning Curves ──────────────────────────────────
cells.append(md("## 11 · Cross-Validation, Confusion Matrices & Learning Curves"))
cells.append(code("""# ── 5-fold CV ──
cv_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=SEED),
    "XGBoost": XGBClassifier(n_estimators=200, eval_metric="mlogloss", random_state=SEED, n_jobs=-1),
}
cv_scores = {}
for name, model in cv_models.items():
    scores = cross_val_score(model, X_combined, y_cat, cv=5, scoring="f1_weighted", n_jobs=-1)
    cv_scores[name] = scores
    print(f"{name}: mean F1={scores.mean():.4f} ± {scores.std():.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.boxplot(cv_scores.values(), labels=cv_scores.keys())
ax.set_title("5-Fold Cross-Validation F1 Scores (Category)")
ax.set_ylabel("F1 (weighted)"); ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.savefig(PLOT_DIR / "cv_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Confusion matrices (top 4) ──
top4 = sorted(clf_results, key=lambda k: clf_results[k]["f1"], reverse=True)[:4]
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
for ax, name in zip(axes, top4):
    cm = confusion_matrix(y_test_c, clf_results[name]["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=le_cat.classes_, yticklabels=le_cat.classes_)
    ax.set_title(f"{name}\\nF1={clf_results[name]['f1']:.4f}", fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.tick_params(axis="both", labelsize=6, rotation=45)
plt.suptitle("Confusion Matrices – Top 4 Models", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(PLOT_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Learning curves ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (name, model) in zip(axes, [
    ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=SEED)),
    ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)),
]):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_combined, y_cat, cv=3, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 8), scoring="f1_weighted"
    )
    ax.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Train")
    ax.plot(train_sizes, val_scores.mean(axis=1), "s-", label="Validation")
    ax.set_title(f"Learning Curve – {name}")
    ax.set_xlabel("Training Size"); ax.set_ylabel("F1 (weighted)")
    ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / "learning_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ CV, confusion matrices & learning curves saved")
"""))

# ── 12. Ensembles ─────────────────────────────────────────────────────────
cells.append(md("## 12 · Voting & Stacking Ensembles"))
cells.append(code("""# Voting Classifier
print("Training Voting Classifier...")
voting = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)),
        ("gb", GradientBoostingClassifier(n_estimators=200, random_state=SEED)),
    ],
    voting="soft"
)
voting.fit(X_train_c, y_train_c)
y_pred_v = voting.predict(X_test_c)
acc_v = accuracy_score(y_test_c, y_pred_v)
f1_v = f1_score(y_test_c, y_pred_v, average="weighted")
clf_results["Voting Ensemble"] = {"accuracy": acc_v, "f1": f1_v, "model": voting, "y_pred": y_pred_v}
print(f"  Voting: Acc={acc_v:.4f}  F1={f1_v:.4f}")

# Stacking Classifier
print("Training Stacking Classifier...")
stacking = StackingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)),
        ("gb", GradientBoostingClassifier(n_estimators=200, random_state=SEED)),
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
    cv=3, n_jobs=-1
)
stacking.fit(X_train_c, y_train_c)
y_pred_s = stacking.predict(X_test_c)
acc_s = accuracy_score(y_test_c, y_pred_s)
f1_s = f1_score(y_test_c, y_pred_s, average="weighted")
clf_results["Stacking Ensemble"] = {"accuracy": acc_s, "f1": f1_s, "model": stacking, "y_pred": y_pred_s}
print(f"  Stacking: Acc={acc_s:.4f}  F1={f1_s:.4f}")

# Final rankings
print("\\n" + "="*60)
print("FINAL MODEL RANKING – Category Classification")
print("="*60)
ranking = sorted(clf_results.items(), key=lambda x: x[1]["f1"], reverse=True)
for i, (name, res) in enumerate(ranking, 1):
    print(f"  {i:>2}. {name:<25s} Acc={res['accuracy']:.4f}  F1={res['f1']:.4f}")
best_overall = ranking[0][0]
print(f"\\n🏆 Best overall: {best_overall} (F1={clf_results[best_overall]['f1']:.4f})")

print("\\n" + "="*60)
print("GENDER PREDICTION RANKING")
print("="*60)
g_ranking = sorted(gender_results.items(), key=lambda x: x[1]["f1"], reverse=True)
for i, (name, res) in enumerate(g_ranking, 1):
    print(f"  {i:>2}. {name:<25s} Acc={res['accuracy']:.4f}  F1={res['f1']:.4f}")
"""))

# ── 13. HTML Report ───────────────────────────────────────────────────────
cells.append(md("## 13 · Generate HTML Report"))
cells.append(code("""def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

images = {}
for p in sorted(PLOT_DIR.glob("*.png")):
    images[p.stem] = img_to_base64(p)

TEMPLATE = \"\"\"<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>🎆 New Year's Resolutions – ML & NLP Report</title>
<style>
:root{--bg:#0f172a;--card:#1e293b;--accent:#06b6d4;--text:#e2e8f0;--muted:#94a3b8}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;padding:2rem}
h1{text-align:center;font-size:2.2rem;margin-bottom:.4rem;color:var(--accent)}
.subtitle{text-align:center;color:var(--muted);margin-bottom:2rem}
.card{background:var(--card);border-radius:12px;padding:1.5rem;margin-bottom:1.5rem;box-shadow:0 4px 24px #0004}
.card h2{color:var(--accent);margin-bottom:1rem;font-size:1.3rem}
table{width:100%;border-collapse:collapse;margin:1rem 0}
th,td{padding:.55rem .8rem;text-align:left;border-bottom:1px solid #334155}
th{color:var(--accent);font-size:.85rem;text-transform:uppercase}
tr:hover{background:#ffffff08}
.best{background:#06b6d415;font-weight:700}
img{width:100%;border-radius:8px;margin:.8rem 0}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem}
@media(max-width:800px){.grid2{grid-template-columns:1fr}}
.tag{display:inline-block;padding:2px 10px;border-radius:6px;font-size:.82rem;background:#06b6d422;color:var(--accent);margin:2px}
.stat-row{display:flex;gap:1.5rem;flex-wrap:wrap;margin:.8rem 0}
.stat-box{background:#06b6d410;border:1px solid #06b6d433;border-radius:8px;padding:1rem 1.5rem;text-align:center;flex:1;min-width:150px}
.stat-box .val{font-size:1.6rem;font-weight:700;color:var(--accent)}
.stat-box .lbl{font-size:.8rem;color:var(--muted);margin-top:.3rem}
</style></head><body>
<h1>🎆 New Year's Resolutions – ML & NLP Report</h1>
<p class="subtitle">4,723 tweets · 10 categories · Sentiment Analysis (TextBlob + BERT)</p>

<div class="card"><h2>📊 Exploratory Data Analysis</h2>
<img src="data:image/png;base64,{{images.eda_overview}}" alt="EDA Overview">
</div>

<div class="card"><h2>💬 Sentiment Analysis – TextBlob</h2>
<div class="stat-row">
<div class="stat-box"><div class="val">{{"{:.1f}%".format(tb_pos_pct)}}</div><div class="lbl">Positive</div></div>
<div class="stat-box"><div class="val">{{"{:.1f}%".format(tb_neu_pct)}}</div><div class="lbl">Neutral</div></div>
<div class="stat-box"><div class="val">{{"{:.1f}%".format(tb_neg_pct)}}</div><div class="lbl">Negative</div></div>
</div>
<img src="data:image/png;base64,{{images.textblob_sentiment}}" alt="TextBlob Sentiment">
</div>

<div class="card"><h2>🤖 Sentiment Analysis – BERT (DistilBERT)</h2>
<div class="stat-row">
<div class="stat-box"><div class="val">{{"{:.1f}%".format(bert_pos_pct)}}</div><div class="lbl">Positive</div></div>
<div class="stat-box"><div class="val">{{"{:.1f}%".format(bert_neg_pct)}}</div><div class="lbl">Negative</div></div>
<div class="stat-box"><div class="val">{{"{:.3f}".format(bert_avg_conf)}}</div><div class="lbl">Avg Confidence</div></div>
</div>
<img src="data:image/png;base64,{{images.bert_sentiment}}" alt="BERT Sentiment">
</div>

<div class="card"><h2>🎯 Task 1 – Tweet Category Classification (10 categories)</h2>
<table><tr><th>#</th><th>Model</th><th>Accuracy</th><th>F1 (weighted)</th></tr>
{% for name, res in clf_ranking %}
<tr{% if loop.first %} class="best"{% endif %}>
<td>{{loop.index}}</td><td>{{name}}</td>
<td>{{"{:.4f}".format(res.accuracy)}}</td><td>{{"{:.4f}".format(res.f1)}}</td></tr>
{% endfor %}</table>
<img src="data:image/png;base64,{{images.category_model_comparison}}" alt="Category Model Comparison">
</div>

<div class="card"><h2>👤 Task 2 – Gender Prediction from Tweets</h2>
<table><tr><th>#</th><th>Model</th><th>Accuracy</th><th>F1 (weighted)</th></tr>
{% for name, res in gender_ranking %}
<tr{% if loop.first %} class="best"{% endif %}>
<td>{{loop.index}}</td><td>{{name}}</td>
<td>{{"{:.4f}".format(res.accuracy)}}</td><td>{{"{:.4f}".format(res.f1)}}</td></tr>
{% endfor %}</table>
<img src="data:image/png;base64,{{images.gender_model_comparison}}" alt="Gender Model Comparison">
</div>

<div class="card"><h2>🔬 Task 3 – Tweet Clustering</h2>
<p>Best k={{best_k}}, Silhouette={{"{:.4f}".format(best_sil)}}</p>
<div class="grid2">
<img src="data:image/png;base64,{{images.elbow_silhouette}}" alt="Elbow & Silhouette">
<img src="data:image/png;base64,{{images.clustering_results}}" alt="Clustering Results">
</div></div>

<div class="card"><h2>🔧 Hyperparameter Tuning & Cross-Validation</h2>
<img src="data:image/png;base64,{{images.cv_comparison}}" alt="CV Comparison">
<img src="data:image/png;base64,{{images.confusion_matrices}}" alt="Confusion Matrices">
<img src="data:image/png;base64,{{images.learning_curves}}" alt="Learning Curves">
</div>

</body></html>\"\"\"

from types import SimpleNamespace
clf_ranking = [(n, SimpleNamespace(**{k: v for k, v in r.items() if k != "model" and k != "y_pred"}))
               for n, r in sorted(clf_results.items(), key=lambda x: x[1]["f1"], reverse=True)]
gender_ranking = [(n, SimpleNamespace(**{k: v for k, v in r.items() if k != "model" and k != "y_pred"}))
                  for n, r in sorted(gender_results.items(), key=lambda x: x[1]["f1"], reverse=True)]

html = jinja2.Template(TEMPLATE).render(
    images=images,
    clf_ranking=clf_ranking,
    gender_ranking=gender_ranking,
    tb_pos_pct=tb_pos_pct, tb_neu_pct=tb_neu_pct, tb_neg_pct=tb_neg_pct,
    bert_pos_pct=bert_pos_pct, bert_neg_pct=bert_neg_pct,
    bert_avg_conf=df["bert_score"].mean(),
    best_k=best_k, best_sil=max(sils),
)

out_path = pathlib.Path("outputs/nyr_ml_report.html")
out_path.write_text(html)
print(f"✅ HTML Report generated: {out_path}")
print(f"   File size: {out_path.stat().st_size / 1024:.1f} KB")
print(f"   Embedded images: {len(images)}")
"""))

# ── Build notebook ────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14.3"}
    },
    "cells": cells
}

out = pathlib.Path("nyr_ml_analysis.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"✅ Notebook written: {out}  ({len(cells)} cells)")
