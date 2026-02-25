#!/usr/bin/env python3
"""Generate wine_ml_analysis.ipynb"""
import json, pathlib

def md(src):
    return {"cell_type":"markdown","metadata":{},"source":src.strip().split("\n")}

def code(src):
    return {"cell_type":"code","metadata":{},"source":src.strip().split("\n"),
            "execution_count":None,"outputs":[]}

cells = []

# ── 0. Title ──
cells.append(md("""# \\U0001f377 Wine Reviews (130k) \\u2013 ML Analysis
**Dataset**: 129,971 wine reviews \\u00b7 43 countries \\u00b7 707 grape varieties \\u00b7 Points 80\\u2013100

| # | Task | Type | Target |
|---|------|------|--------|
| 1 | Wine Quality Classification | Multi-class | Low / Medium / High / Premium |
| 2 | Wine Price Regression | Regression | Price (USD) |
| 3 | Variety Classification | Multi-class | Top 10 grape varieties |
| 4 | Wine Clustering | Unsupervised | K-Means on wine features |
"""))

# ── 1. Imports ──
cells.append(md("## 1 \\u00b7 Imports"))
cells.append(code("""import warnings, os, pathlib
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, mean_squared_error, r2_score,
                             mean_absolute_error, silhouette_score)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor,
                              AdaBoostClassifier, VotingClassifier, StackingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from xgboost import XGBClassifier

import jinja2, base64
from io import BytesIO

SEED = 42
PLOT_DIR = pathlib.Path("outputs/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", palette="viridis")
print("\\u2705 Imports OK")
"""))

# ── 2. Load & Feature Engineering ──
cells.append(md("## 2 \\u00b7 Load Data & Feature Engineering"))
cells.append(code("""df_raw = pd.read_csv("winemag-data-130k-v2.csv")
print(f"Raw shape: {df_raw.shape}")
print(f"Missing values:\\n{df_raw.isnull().sum()}")

df = df_raw.copy()

# ---- Fill missing values ----
df["country"]  = df["country"].fillna("Unknown")
df["province"] = df["province"].fillna("Unknown")
df["variety"]  = df["variety"].fillna("Unknown")
df["taster_name"] = df["taster_name"].fillna("Unknown")

# Drop rows without price (needed for regression)
df_full = df.copy()  # keep full set for variety classification
df = df.dropna(subset=["price"])
print(f"\\nAfter dropping missing price: {df.shape}")

# ---- Description length ----
df["desc_length"]    = df["description"].str.len()
df["desc_word_count"] = df["description"].str.split().str.len()

# ---- Extract vintage year from title ----
df["vintage"] = df["title"].str.extract(r"(\\d{4})").astype(float)
df["vintage"] = df["vintage"].where((df["vintage"] >= 1900) & (df["vintage"] <= 2025))

# ---- Encode categoricals ----
le_country  = LabelEncoder(); df["country_enc"]  = le_country.fit_transform(df["country"])
le_province = LabelEncoder(); df["province_enc"] = le_province.fit_transform(df["province"])
le_variety  = LabelEncoder(); df["variety_enc"]  = le_variety.fit_transform(df["variety"])
le_taster   = LabelEncoder(); df["taster_enc"]   = le_taster.fit_transform(df["taster_name"])

# ---- Frequency features ----
df["country_count"]  = df["country"].map(df["country"].value_counts())
df["variety_count"]  = df["variety"].map(df["variety"].value_counts())
df["winery_count"]   = df["winery"].map(df["winery"].value_counts())
df["province_count"] = df["province"].map(df["province"].value_counts())

# ---- Average price/points by group ----
df["country_avg_price"]  = df["country"].map(df.groupby("country")["price"].mean())
df["variety_avg_price"]  = df["variety"].map(df.groupby("variety")["price"].mean())
df["country_avg_points"] = df["country"].map(df.groupby("country")["points"].mean())
df["variety_avg_points"] = df["variety"].map(df.groupby("variety")["points"].mean())

# ---- Quality bins ----
df["quality"] = pd.cut(df["points"],
    bins=[79, 85, 89, 93, 100],
    labels=["Low", "Medium", "High", "Premium"])
print(f"\\nQuality distribution:\\n{df['quality'].value_counts().to_string()}")

# ---- Log price ----
df["log_price"] = np.log1p(df["price"])

print(f"\\nFinal shape: {df.shape}")
print(f"Points: {df['points'].min()}-{df['points'].max()}, Price: ${df['price'].min()}-${df['price'].max()}")
df.head(3)
"""))

# ── 3. EDA ──
cells.append(md("## 3 \\u00b7 Exploratory Data Analysis"))
cells.append(code("""fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1 - Points distribution
df["points"].hist(bins=21, ax=axes[0, 0], color="purple", edgecolor="white", alpha=0.8)
axes[0, 0].set_title("Points Distribution (80-100)")
axes[0, 0].set_xlabel("Points"); axes[0, 0].set_ylabel("Count")

# 2 - Price distribution (log scale)
df["price"][df["price"] < 300].hist(bins=60, ax=axes[0, 1], color="gold", edgecolor="white")
axes[0, 1].set_title("Price Distribution (< $300)")
axes[0, 1].set_xlabel("Price (USD)"); axes[0, 1].set_ylabel("Count")

# 3 - Top 15 countries
df["country"].value_counts().head(15).plot.barh(ax=axes[0, 2],
    color=sns.color_palette("rocket", 15))
axes[0, 2].set_title("Top 15 Countries"); axes[0, 2].set_xlabel("Reviews")
axes[0, 2].invert_yaxis()

# 4 - Quality class distribution
df["quality"].value_counts().sort_index().plot.bar(ax=axes[1, 0],
    color=["#e74c3c", "#f39c12", "#2ecc71", "#9b59b6"])
axes[1, 0].set_title("Quality Classes"); axes[1, 0].set_ylabel("Count")
axes[1, 0].tick_params(axis="x", rotation=0)

# 5 - Points vs Price scatter
axes[1, 1].scatter(df["points"], df["price"].clip(upper=500),
    alpha=0.02, s=3, c="teal")
axes[1, 1].set_title("Points vs Price"); axes[1, 1].set_xlabel("Points")
axes[1, 1].set_ylabel("Price (USD, clipped at $500)")

# 6 - Top 10 varieties
df["variety"].value_counts().head(10).plot.bar(ax=axes[1, 2],
    color=sns.color_palette("mako", 10))
axes[1, 2].set_title("Top 10 Grape Varieties"); axes[1, 2].set_ylabel("Count")
axes[1, 2].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(PLOT_DIR / "eda_overview.png", dpi=150, bbox_inches="tight")
plt.show()

# Points by country (top 15)
fig, ax = plt.subplots(figsize=(14, 7))
top15 = df["country"].value_counts().head(15).index
box_data = [df[df["country"] == c]["points"].values for c in top15]
bp = ax.boxplot(box_data, labels=top15, patch_artist=True)
colors = sns.color_palette("viridis", 15)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
ax.set_title("Points by Country (Top 15)"); ax.set_ylabel("Points")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(PLOT_DIR / "points_by_country.png", dpi=150, bbox_inches="tight")
plt.show()

# Correlation heatmap
num_cols = ["points", "price", "log_price", "desc_length", "desc_word_count",
            "country_enc", "province_enc", "variety_enc", "taster_enc",
            "country_count", "variety_count", "winery_count",
            "country_avg_price", "variety_avg_price",
            "country_avg_points", "variety_avg_points"]
fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm",
            ax=ax, linewidths=0.5, vmin=-1, vmax=1, annot_kws={"size": 7})
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(PLOT_DIR / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("\\u2705 All EDA plots saved")
"""))

# ── 4. Prepare classification features ──
cells.append(md("## 4 \\u00b7 Prepare Features for Quality Classification"))
cells.append(code("""feature_cols = [
    "price", "log_price", "desc_length", "desc_word_count",
    "country_enc", "province_enc", "variety_enc", "taster_enc",
    "country_count", "variety_count", "winery_count", "province_count",
    "country_avg_price", "variety_avg_price",
    "country_avg_points", "variety_avg_points"
]

df_clf = df.dropna(subset=feature_cols + ["quality"]).copy()
le_quality = LabelEncoder()
y_quality = le_quality.fit_transform(df_clf["quality"])
X_quality = df_clf[feature_cols].values
print(f"Quality classification: {X_quality.shape}")
print(f"Classes: {le_quality.classes_}")
print(f"Distribution: {dict(zip(*np.unique(y_quality, return_counts=True)))}")

X_train_q, X_test_q, y_train_q, y_test_q = train_test_split(
    X_quality, y_quality, test_size=0.2, random_state=SEED, stratify=y_quality
)
scaler_q = StandardScaler()
X_train_qs = scaler_q.fit_transform(X_train_q)
X_test_qs  = scaler_q.transform(X_test_q)
print(f"Train: {X_train_qs.shape}, Test: {X_test_qs.shape}")
"""))

# ── 5. Quality Classification (10 models) ──
cells.append(md("## 5 \\u00b7 Wine Quality Classification (10 Models)"))
cells.append(code("""classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(max_depth=15, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=SEED),
    "AdaBoost": AdaBoostClassifier(n_estimators=150, random_state=SEED),
    "XGBoost": XGBClassifier(n_estimators=200, eval_metric="mlogloss",
                              random_state=SEED, n_jobs=-1),
    "SVM": SVC(kernel="rbf", probability=True, random_state=SEED),
    "Naive Bayes": GaussianNB(),
    "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=SEED),
}

clf_results = {}
for name, model in classifiers.items():
    print(f"Training {name}...", end=" ")
    model.fit(X_train_qs, y_train_q)
    y_pred = model.predict(X_test_qs)
    acc = accuracy_score(y_test_q, y_pred)
    f1 = f1_score(y_test_q, y_pred, average="weighted")
    clf_results[name] = {"accuracy": acc, "f1": f1, "model": model, "y_pred": y_pred}
    print(f"Acc={acc:.4f}  F1={f1:.4f}")

best_clf_name = max(clf_results, key=lambda k: clf_results[k]["f1"])
print(f"\\n\\U0001f3c6 Best classifier: {best_clf_name} (F1={clf_results[best_clf_name]['f1']:.4f})")

# Bar chart
fig, ax = plt.subplots(figsize=(12, 6))
names = list(clf_results.keys())
accs = [clf_results[n]["accuracy"] for n in names]
f1s  = [clf_results[n]["f1"] for n in names]
x = np.arange(len(names))
ax.bar(x - 0.2, accs, 0.4, label="Accuracy", color="steelblue")
ax.bar(x + 0.2, f1s,  0.4, label="F1 (weighted)", color="coral")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right")
ax.set_ylim(0, 1); ax.set_title("Wine Quality - Model Comparison")
ax.legend(); plt.tight_layout()
plt.savefig(PLOT_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("\\u2705 Model comparison saved")
"""))

# ── 6. Price Regression ──
cells.append(md("## 6 \\u00b7 Wine Price Regression"))
cells.append(code("""reg_features = [
    "points", "desc_length", "desc_word_count",
    "country_enc", "province_enc", "variety_enc", "taster_enc",
    "country_count", "variety_count", "winery_count", "province_count",
    "country_avg_points", "variety_avg_points"
]

df_reg = df.dropna(subset=reg_features).copy()
X_reg = df_reg[reg_features].values
y_reg = df_reg["log_price"].values
print(f"Regression dataset: {X_reg.shape}")

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=SEED
)
scaler_r = StandardScaler()
X_train_rs = scaler_r.fit_transform(X_train_r)
X_test_rs  = scaler_r.transform(X_test_r)

regressors = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.001, max_iter=2000),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000),
    "Decision Tree": DecisionTreeRegressor(max_depth=15, random_state=SEED),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=SEED),
}

reg_results = {}
for name, model in regressors.items():
    print(f"Training {name}...", end=" ")
    model.fit(X_train_rs, y_train_r)
    y_pred_log = model.predict(X_test_rs)
    r2   = r2_score(y_test_r, y_pred_log)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_log))
    mae  = mean_absolute_error(y_test_r, y_pred_log)
    # Also in original price space
    y_pred_orig = np.expm1(y_pred_log)
    y_test_orig = np.expm1(y_test_r)
    r2_orig = r2_score(y_test_orig, y_pred_orig)
    reg_results[name] = {"r2": r2, "rmse": rmse, "mae": mae, "r2_orig": r2_orig,
                         "model": model, "y_pred": y_pred_log}
    print(f"R2(log)={r2:.4f}  R2(orig)={r2_orig:.4f}  RMSE(log)={rmse:.4f}")

best_reg_name = max(reg_results, key=lambda k: reg_results[k]["r2"])
print(f"\\n\\U0001f3c6 Best regressor: {best_reg_name} (R2={reg_results[best_reg_name]['r2']:.4f})")

# Feature importance
best_reg = reg_results[best_reg_name]["model"]
if hasattr(best_reg, "feature_importances_"):
    imp = best_reg.feature_importances_
    idx = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(np.array(reg_features)[idx], imp[idx], color="darkcyan")
    ax.set_title(f"Price Regression - Feature Importance ({best_reg_name})")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "price_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()

# Actual vs Predicted
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, (name, res) in zip(axes.flat, reg_results.items()):
    y_t = np.expm1(y_test_r)
    y_p = np.expm1(res["y_pred"])
    ax.scatter(y_t, y_p, alpha=0.03, s=2, c="teal")
    lims = [0, min(500, max(y_t.max(), y_p.max()) * 1.05)]
    ax.plot(lims, lims, "r--", linewidth=1.5)
    ax.set_title(f"{name}\\nR2={res['r2_orig']:.4f}")
    ax.set_xlabel("Actual ($)"); ax.set_ylabel("Predicted ($)")
    ax.set_xlim(lims); ax.set_ylim(lims)
plt.suptitle("Price Regression - Actual vs Predicted", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(PLOT_DIR / "price_regression_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\\u2705 Price regression plots saved")
"""))

# ── 7. Variety Classification ──
cells.append(md("## 7 \\u00b7 Variety Classification (Top 10 Grapes)"))
cells.append(code("""# Use top 10 varieties
top10_var = df["variety"].value_counts().head(10).index.tolist()
df_var = df[df["variety"].isin(top10_var)].copy()
print(f"Variety classification dataset: {df_var.shape}")
print(f"Varieties: {top10_var}")

var_features = [
    "points", "price", "log_price", "desc_length", "desc_word_count",
    "country_enc", "province_enc", "taster_enc",
    "country_count", "winery_count", "province_count",
    "country_avg_price", "country_avg_points"
]

le_var10 = LabelEncoder()
y_var = le_var10.fit_transform(df_var["variety"])
X_var = df_var[var_features].values
print(f"Shape: {X_var.shape}, Classes: {len(le_var10.classes_)}")

X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(
    X_var, y_var, test_size=0.2, random_state=SEED, stratify=y_var
)
scaler_v = StandardScaler()
X_train_vs = scaler_v.fit_transform(X_train_v)
X_test_vs  = scaler_v.transform(X_test_v)

var_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=SEED),
    "XGBoost": XGBClassifier(n_estimators=200, eval_metric="mlogloss",
                              random_state=SEED, n_jobs=-1),
    "SVM": SVC(kernel="rbf", probability=True, random_state=SEED),
    "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=SEED),
}

var_results = {}
for name, model in var_models.items():
    print(f"Training {name}...", end=" ")
    model.fit(X_train_vs, y_train_v)
    y_pred = model.predict(X_test_vs)
    acc = accuracy_score(y_test_v, y_pred)
    f1  = f1_score(y_test_v, y_pred, average="weighted")
    var_results[name] = {"accuracy": acc, "f1": f1, "model": model, "y_pred": y_pred}
    print(f"Acc={acc:.4f}  F1={f1:.4f}")

best_var_name = max(var_results, key=lambda k: var_results[k]["f1"])
print(f"\\n\\U0001f3c6 Best variety classifier: {best_var_name} (F1={var_results[best_var_name]['f1']:.4f})")

fig, ax = plt.subplots(figsize=(10, 5))
names_v = list(var_results.keys())
f1s_v   = [var_results[n]["f1"] for n in names_v]
ax.barh(names_v, f1s_v, color=sns.color_palette("rocket", len(names_v)))
ax.set_title("Variety Classification - F1 Scores"); ax.set_xlabel("F1 (weighted)")
ax.set_xlim(0, 1)
for i, v in enumerate(f1s_v):
    ax.text(v + 0.01, i, f"{v:.4f}", va="center")
plt.tight_layout()
plt.savefig(PLOT_DIR / "variety_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("\\u2705 Variety classification plot saved")
"""))

# ── 8. Clustering ──
cells.append(md("## 8 \\u00b7 Wine Clustering"))
cells.append(code("""clust_features = [
    "points", "log_price", "desc_length", "desc_word_count",
    "country_enc", "variety_enc", "province_enc",
    "country_count", "variety_count", "winery_count",
    "country_avg_price", "variety_avg_price",
    "country_avg_points", "variety_avg_points"
]

df_clust = df.dropna(subset=clust_features).copy()
X_clust = df_clust[clust_features].values
scaler_cl = StandardScaler()
X_clust_s = scaler_cl.fit_transform(X_clust)

K_range = range(2, 11)
inertias, sils = [], []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km.fit_predict(X_clust_s)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_clust_s, labels, sample_size=5000, random_state=SEED))

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
df_clust["cluster"] = km_final.fit_predict(X_clust_s)

cluster_profile = df_clust.groupby("cluster")[clust_features].mean()
print("\\nCluster profiles:")
print(cluster_profile.round(2).to_string())

fig, ax = plt.subplots(figsize=(14, 6))
cp_norm = (cluster_profile - cluster_profile.min()) / (cluster_profile.max() - cluster_profile.min() + 1e-9)
cp_norm.T.plot(kind="bar", ax=ax, colormap="viridis")
ax.set_title(f"Wine Cluster Profiles (k={best_k})")
ax.set_ylabel("Normalized value"); ax.set_xlabel("Feature")
ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1))
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(PLOT_DIR / "clustering_results.png", dpi=150, bbox_inches="tight")
plt.show()

# Cluster stats
for c in sorted(df_clust["cluster"].unique()):
    sub = df_clust[df_clust["cluster"] == c]
    print(f"Cluster {c}: n={len(sub)}, avg_points={sub['points'].mean():.1f}, "
          f"avg_price=${np.expm1(sub['log_price']).mean():.1f}, "
          f"top_country={sub['country'].mode().iloc[0]}")
print("\\u2705 Clustering plots saved")
"""))

# ── 9. Hyperparameter Tuning ──
cells.append(md("## 9 \\u00b7 Hyperparameter Tuning"))
cells.append(code("""# GridSearchCV - Random Forest
print("GridSearchCV on Random Forest...")
rf_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
}
gs_rf = GridSearchCV(RandomForestClassifier(random_state=SEED, n_jobs=-1),
                     rf_grid, cv=3, scoring="f1_weighted", n_jobs=-1)
gs_rf.fit(X_train_qs, y_train_q)
print(f"  Best params: {gs_rf.best_params_}")
print(f"  Best CV F1:  {gs_rf.best_score_:.4f}")

# RandomizedSearchCV - Gradient Boosting
print("\\nRandomizedSearchCV on Gradient Boosting...")
gb_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 1.0],
}
rs_gb = RandomizedSearchCV(GradientBoostingClassifier(random_state=SEED),
                           gb_dist, n_iter=20, cv=3, scoring="f1_weighted",
                           random_state=SEED, n_jobs=-1)
rs_gb.fit(X_train_qs, y_train_q)
print(f"  Best params: {rs_gb.best_params_}")
print(f"  Best CV F1:  {rs_gb.best_score_:.4f}")

# Evaluate tuned models
for label, model in [("Tuned RF (Grid)", gs_rf.best_estimator_),
                     ("Tuned GB (Random)", rs_gb.best_estimator_)]:
    y_pred = model.predict(X_test_qs)
    acc = accuracy_score(y_test_q, y_pred)
    f1  = f1_score(y_test_q, y_pred, average="weighted")
    clf_results[label] = {"accuracy": acc, "f1": f1, "model": model, "y_pred": y_pred}
    print(f"  {label}: Acc={acc:.4f}  F1={f1:.4f}")

print("\\n\\u2705 Hyperparameter tuning complete")
"""))

# ── 10. CV + Feature imp + confusion + learning ──
cells.append(md("## 10 \\u00b7 Cross-Validation, Feature Importance, Confusion Matrices & Learning Curves"))
cells.append(code("""# 5-fold CV
cv_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=SEED),
    "XGBoost": XGBClassifier(n_estimators=200, eval_metric="mlogloss",
                              random_state=SEED, n_jobs=-1),
}
scaler_cv = StandardScaler()
X_cvs = scaler_cv.fit_transform(X_quality)
cv_scores = {}
for name, model in cv_models.items():
    scores = cross_val_score(model, X_cvs, y_quality, cv=5, scoring="f1_weighted", n_jobs=-1)
    cv_scores[name] = scores
    print(f"{name}: mean F1={scores.mean():.4f} +/- {scores.std():.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.boxplot(cv_scores.values(), labels=cv_scores.keys())
ax.set_title("5-Fold Cross-Validation F1 Scores (Quality)")
ax.set_ylabel("F1 (weighted)"); ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.savefig(PLOT_DIR / "cv_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# Feature importance
best_clf_model = clf_results[best_clf_name]["model"]
if hasattr(best_clf_model, "feature_importances_"):
    imp = best_clf_model.feature_importances_
    idx = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(np.array(feature_cols)[idx], imp[idx], color="coral")
    ax.set_title(f"Feature Importance - {best_clf_name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()

# Confusion matrices (top 4)
top4 = sorted(clf_results, key=lambda k: clf_results[k]["f1"], reverse=True)[:4]
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
for ax, name in zip(axes, top4):
    cm = confusion_matrix(y_test_q, clf_results[name]["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=le_quality.classes_, yticklabels=le_quality.classes_)
    ax.set_title(f"{name}\\nF1={clf_results[name]['f1']:.4f}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.suptitle("Confusion Matrices - Top 4 Models", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(PLOT_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()

# Learning curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (name, model) in zip(axes, [
    ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)),
    ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, random_state=SEED)),
]):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_cvs, y_quality, cv=3, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 8), scoring="f1_weighted"
    )
    ax.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Train")
    ax.plot(train_sizes, val_scores.mean(axis=1), "s-", label="Validation")
    ax.set_title(f"Learning Curve - {name}")
    ax.set_xlabel("Training Size"); ax.set_ylabel("F1 (weighted)")
    ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / "learning_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("\\u2705 CV, feature importance, confusion matrices & learning curves saved")
"""))

# ── 11. Ensembles ──
cells.append(md("## 11 \\u00b7 Voting & Stacking Ensembles"))
cells.append(code("""# Voting Classifier
print("Training Voting Classifier...")
voting = VotingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)),
        ("gb", GradientBoostingClassifier(n_estimators=200, random_state=SEED)),
        ("xgb", XGBClassifier(n_estimators=200, eval_metric="mlogloss",
                               random_state=SEED, n_jobs=-1)),
    ],
    voting="soft"
)
voting.fit(X_train_qs, y_train_q)
y_pred_v = voting.predict(X_test_qs)
acc_v = accuracy_score(y_test_q, y_pred_v)
f1_v  = f1_score(y_test_q, y_pred_v, average="weighted")
clf_results["Voting Ensemble"] = {"accuracy": acc_v, "f1": f1_v, "model": voting, "y_pred": y_pred_v}
print(f"  Voting: Acc={acc_v:.4f}  F1={f1_v:.4f}")

# Stacking Classifier
print("Training Stacking Classifier...")
stacking = StackingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)),
        ("gb", GradientBoostingClassifier(n_estimators=200, random_state=SEED)),
        ("xgb", XGBClassifier(n_estimators=200, eval_metric="mlogloss",
                               random_state=SEED, n_jobs=-1)),
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
    cv=3, n_jobs=-1
)
stacking.fit(X_train_qs, y_train_q)
y_pred_s = stacking.predict(X_test_qs)
acc_s = accuracy_score(y_test_q, y_pred_s)
f1_s  = f1_score(y_test_q, y_pred_s, average="weighted")
clf_results["Stacking Ensemble"] = {"accuracy": acc_s, "f1": f1_s, "model": stacking, "y_pred": y_pred_s}
print(f"  Stacking: Acc={acc_s:.4f}  F1={f1_s:.4f}")

# Final ranking
print("\\n" + "="*60)
print("FINAL MODEL RANKING - Wine Quality Classification")
print("="*60)
ranking = sorted(clf_results.items(), key=lambda x: x[1]["f1"], reverse=True)
for i, (name, res) in enumerate(ranking, 1):
    print(f"  {i:>2}. {name:<25s} Acc={res['accuracy']:.4f}  F1={res['f1']:.4f}")
best_overall = ranking[0][0]
print(f"\\n\\U0001f3c6 Best overall: {best_overall} (F1={clf_results[best_overall]['f1']:.4f})")

print("\\n" + "="*60)
print("VARIETY CLASSIFICATION RANKING")
print("="*60)
v_ranking = sorted(var_results.items(), key=lambda x: x[1]["f1"], reverse=True)
for i, (name, res) in enumerate(v_ranking, 1):
    print(f"  {i:>2}. {name:<25s} Acc={res['accuracy']:.4f}  F1={res['f1']:.4f}")
"""))

# ── 12. HTML Report ──
cells.append(md("## 12 \\u00b7 Generate HTML Report"))

# Build the HTML report cell carefully to avoid triple-quote issues
html_template = (
    '<!DOCTYPE html>\\n'
    '<html lang="en"><head><meta charset="UTF-8">\\n'
    '<meta name="viewport" content="width=device-width,initial-scale=1">\\n'
    '<title>Wine Reviews - ML Report</title>\\n'
    '<style>\\n'
    ':root{--bg:#0f172a;--card:#1e293b;--accent:#a855f7;--text:#e2e8f0;--muted:#94a3b8}\\n'
    '*{margin:0;padding:0;box-sizing:border-box}\\n'
    'body{background:var(--bg);color:var(--text);font-family:"Segoe UI",system-ui,sans-serif;padding:2rem}\\n'
    'h1{text-align:center;font-size:2.2rem;margin-bottom:.4rem;color:var(--accent)}\\n'
    '.subtitle{text-align:center;color:var(--muted);margin-bottom:2rem}\\n'
    '.card{background:var(--card);border-radius:12px;padding:1.5rem;margin-bottom:1.5rem;box-shadow:0 4px 24px #0004}\\n'
    '.card h2{color:var(--accent);margin-bottom:1rem;font-size:1.3rem}\\n'
    'table{width:100%;border-collapse:collapse;margin:1rem 0}\\n'
    'th,td{padding:.55rem .8rem;text-align:left;border-bottom:1px solid #334155}\\n'
    'th{color:var(--accent);font-size:.85rem;text-transform:uppercase}\\n'
    'tr:hover{background:#ffffff08}\\n'
    '.best{background:#a855f715;font-weight:700}\\n'
    'img{width:100%;border-radius:8px;margin:.8rem 0}\\n'
    '.grid2{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem}\\n'
    '@media(max-width:800px){.grid2{grid-template-columns:1fr}}\\n'
    '</style></head><body>\\n'
    '<h1>Wine Reviews - ML Report</h1>\\n'
    '<p class="subtitle">129,971 reviews - 43 countries - 707 varieties</p>\\n'
    '\\n'
    '<div class="card"><h2>Exploratory Data Analysis</h2>\\n'
    '<img src="data:image/png;base64,{{images.eda_overview}}" alt="EDA Overview">\\n'
    '<div class="grid2">\\n'
    '<img src="data:image/png;base64,{{images.points_by_country}}" alt="Points by Country">\\n'
    '<img src="data:image/png;base64,{{images.correlation_heatmap}}" alt="Correlation Heatmap">\\n'
    '</div></div>\\n'
    '\\n'
    '<div class="card"><h2>Task 1 - Wine Quality Classification</h2>\\n'
    '<table><tr><th>#</th><th>Model</th><th>Accuracy</th><th>F1 (weighted)</th></tr>\\n'
    '{% for name, res in clf_ranking %}\\n'
    '<tr{% if loop.first %} class="best"{% endif %}>\\n'
    '<td>{{loop.index}}</td><td>{{name}}</td>\\n'
    '<td>{{"{:.4f}".format(res.accuracy)}}</td><td>{{"{:.4f}".format(res.f1)}}</td></tr>\\n'
    '{% endfor %}</table>\\n'
    '<img src="data:image/png;base64,{{images.model_comparison}}" alt="Model Comparison">\\n'
    '</div>\\n'
    '\\n'
    '<div class="card"><h2>Task 2 - Wine Price Regression</h2>\\n'
    '<table><tr><th>#</th><th>Model</th><th>R2 (log)</th><th>R2 (orig)</th><th>RMSE</th></tr>\\n'
    '{% for name, res in reg_ranking %}\\n'
    '<tr{% if loop.first %} class="best"{% endif %}>\\n'
    '<td>{{loop.index}}</td><td>{{name}}</td>\\n'
    '<td>{{"{:.4f}".format(res.r2)}}</td><td>{{"{:.4f}".format(res.r2_orig)}}</td>\\n'
    '<td>{{"{:.4f}".format(res.rmse)}}</td></tr>\\n'
    '{% endfor %}</table>\\n'
    '<div class="grid2">\\n'
    '{% if images.price_feature_importance %}\\n'
    '<img src="data:image/png;base64,{{images.price_feature_importance}}" alt="Price Feature Importance">\\n'
    '{% endif %}\\n'
    '<img src="data:image/png;base64,{{images.price_regression_results}}" alt="Price Regression">\\n'
    '</div></div>\\n'
    '\\n'
    '<div class="card"><h2>Task 3 - Variety Classification (Top 10 Grapes)</h2>\\n'
    '<table><tr><th>#</th><th>Model</th><th>Accuracy</th><th>F1 (weighted)</th></tr>\\n'
    '{% for name, res in var_ranking %}\\n'
    '<tr{% if loop.first %} class="best"{% endif %}>\\n'
    '<td>{{loop.index}}</td><td>{{name}}</td>\\n'
    '<td>{{"{:.4f}".format(res.accuracy)}}</td><td>{{"{:.4f}".format(res.f1)}}</td></tr>\\n'
    '{% endfor %}</table>\\n'
    '<img src="data:image/png;base64,{{images.variety_comparison}}" alt="Variety Comparison">\\n'
    '</div>\\n'
    '\\n'
    '<div class="card"><h2>Task 4 - Wine Clustering</h2>\\n'
    '<p>Best k={{best_k}}, Silhouette={{"{:.4f}".format(best_sil)}}</p>\\n'
    '<div class="grid2">\\n'
    '<img src="data:image/png;base64,{{images.elbow_silhouette}}" alt="Elbow and Silhouette">\\n'
    '<img src="data:image/png;base64,{{images.clustering_results}}" alt="Clustering Results">\\n'
    '</div></div>\\n'
    '\\n'
    '<div class="card"><h2>Hyperparameter Tuning and Cross-Validation</h2>\\n'
    '<div class="grid2">\\n'
    '<img src="data:image/png;base64,{{images.cv_comparison}}" alt="CV Comparison">\\n'
    '{% if images.feature_importance %}\\n'
    '<img src="data:image/png;base64,{{images.feature_importance}}" alt="Feature Importance">\\n'
    '{% endif %}\\n'
    '</div>\\n'
    '<img src="data:image/png;base64,{{images.confusion_matrices}}" alt="Confusion Matrices">\\n'
    '<img src="data:image/png;base64,{{images.learning_curves}}" alt="Learning Curves">\\n'
    '</div>\\n'
    '\\n'
    '</body></html>'
)

cells.append(code(f"""def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

images = {{}}
for p in sorted(PLOT_DIR.glob("*.png")):
    images[p.stem] = img_to_base64(p)

TEMPLATE = "{html_template}"

from types import SimpleNamespace
clf_ranking = [(n, SimpleNamespace(**{{k: v for k, v in r.items() if k not in ("model", "y_pred")}}))
               for n, r in sorted(clf_results.items(), key=lambda x: x[1]["f1"], reverse=True)]
reg_ranking = [(n, SimpleNamespace(**{{k: v for k, v in r.items() if k not in ("model", "y_pred")}}))
               for n, r in sorted(reg_results.items(), key=lambda x: x[1]["r2"], reverse=True)]
var_ranking = [(n, SimpleNamespace(**{{k: v for k, v in r.items() if k not in ("model", "y_pred")}}))
               for n, r in sorted(var_results.items(), key=lambda x: x[1]["f1"], reverse=True)]

html = jinja2.Template(TEMPLATE).render(
    images=images,
    clf_ranking=clf_ranking,
    reg_ranking=reg_ranking,
    var_ranking=var_ranking,
    best_k=best_k, best_sil=max(sils),
)

out_path = pathlib.Path("outputs/wine_ml_report.html")
out_path.write_text(html)
print(f"\\u2705 HTML Report generated: {{out_path}}")
print(f"   File size: {{out_path.stat().st_size / 1024:.1f}} KB")
print(f"   Embedded images: {{len(images)}}")
"""))

# ── Build notebook ──
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14.3"}
    },
    "cells": cells
}

out = pathlib.Path("wine_ml_analysis.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Notebook written: {out}  ({len(cells)} cells)")
