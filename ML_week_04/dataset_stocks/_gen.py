#!/usr/bin/env python3
"""Generate stocks_ml_analysis.ipynb – S&P 500 Stock Prices ML Analysis."""
import json, pathlib

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip().split("\n")}

def code(src):
    lines = src.strip().split("\n")
    return {"cell_type": "code", "metadata": {}, "source": lines,
            "execution_count": None, "outputs": []}

cells = []

# ── 0. Title ──────────────────────────────────────────────────────────────
cells.append(md("""# 📈 S&P 500 Stock Prices (2014–2017) – ML Analysis
**Dataset**: 497,472 daily records · 505 stocks · 2014-01-02 to 2017-12-29

| # | Task | Type | Target |
|---|------|------|--------|
| 1 | Price Direction Classification | Binary Classification | Next-day up / down |
| 2 | Daily Return Regression | Regression | Next-day return % |
| 3 | Volatility Classification | Multi-class Classification | Low / Medium / High volatility |
| 4 | Stock Clustering | Unsupervised | K-Means on stock-level features |
"""))

# ── 1. Imports ────────────────────────────────────────────────────────────
cells.append(md("## 1 · Imports"))
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

# ── 2. Load & Feature Engineering ────────────────────────────────────────
cells.append(md("## 2 · Load Data & Feature Engineering"))
cells.append(code("""df_raw = pd.read_csv("S&P 500 Stock Prices 2014-2017.csv")
print(f"Raw shape: {df_raw.shape}")
print(f"Missing values:\\n{df_raw.isnull().sum()}")
print(f"Unique symbols: {df_raw['symbol'].nunique()}")

df = df_raw.copy()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

# Fill small number of missing open/high/low with close
df["open"]  = df["open"].fillna(df["close"])
df["high"]  = df["high"].fillna(df["close"])
df["low"]   = df["low"].fillna(df["close"])

# Technical indicators per stock
df["daily_return"]    = df.groupby("symbol")["close"].pct_change()
df["daily_range"]     = (df["high"] - df["low"]) / df["close"]
df["open_close_pct"]  = (df["close"] - df["open"]) / df["open"]
df["gap"]             = df.groupby("symbol").apply(
    lambda g: g["open"] / g["close"].shift(1) - 1, include_groups=False
).reset_index(level=0, drop=True)

# Moving averages & momentum
for w in [5, 10, 20]:
    df[f"sma_{w}"]  = df.groupby("symbol")["close"].transform(lambda x: x.rolling(w).mean())
    df[f"vol_{w}"]  = df.groupby("symbol")["daily_return"].transform(lambda x: x.rolling(w).std())
df["sma_ratio_5_20"] = df["sma_5"] / df["sma_20"]

# Volume features
df["vol_sma_5"]  = df.groupby("symbol")["volume"].transform(lambda x: x.rolling(5).mean())
df["vol_ratio"]  = df["volume"] / df["vol_sma_5"]

# RSI (14-day)
delta = df.groupby("symbol")["close"].diff()
gain  = delta.clip(lower=0)
loss  = (-delta.clip(upper=0))
avg_gain = df.groupby("symbol")["close"].transform(
    lambda x: gain.loc[x.index].rolling(14).mean()
)
avg_loss = df.groupby("symbol")["close"].transform(
    lambda x: loss.loc[x.index].rolling(14).mean()
)
rs = avg_gain / (avg_loss + 1e-9)
df["rsi_14"] = 100 - (100 / (1 + rs))

# Calendar features
df["dow"]    = df["date"].dt.dayofweek
df["month"]  = df["date"].dt.month
df["year"]   = df["date"].dt.year

# Target: next-day direction (1=up, 0=down/flat)
df["next_return"] = df.groupby("symbol")["daily_return"].shift(-1)
df["direction"]   = (df["next_return"] > 0).astype(int)

# Drop warm-up rows (first 20 per symbol + last row per symbol)
df = df.groupby("symbol").apply(lambda g: g.iloc[20:-1], include_groups=False).reset_index(drop=True)
df = df.dropna()

print(f"\\nEngineered shape: {df.shape}")
print(f"Direction balance: {df['direction'].value_counts().to_dict()}")
df.head(3)
"""))

# ── 3. EDA ────────────────────────────────────────────────────────────────
cells.append(md("## 3 · Exploratory Data Analysis"))
cells.append(code("""fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1 - Distribution of daily returns
df["daily_return"].clip(-0.1, 0.1).hist(bins=80, ax=axes[0, 0], color="steelblue",
                                         edgecolor="white", alpha=0.8)
axes[0, 0].axvline(0, color="red", linestyle="--", linewidth=1.5)
axes[0, 0].set_title("Daily Return Distribution")
axes[0, 0].set_xlabel("Return"); axes[0, 0].set_ylabel("Count")

# 2 - Average close price per year
yearly = df.groupby("year")["close"].mean()
axes[0, 1].bar(yearly.index.astype(str), yearly.values, color="teal")
axes[0, 1].set_title("Average Close Price by Year")
axes[0, 1].set_ylabel("Price (USD)")

# 3 - Top 15 stocks by average volume
top_vol = df.groupby("symbol")["volume"].mean().sort_values(ascending=False).head(15)
top_vol.plot.barh(ax=axes[0, 2], color=sns.color_palette("rocket", 15))
axes[0, 2].set_title("Top 15 Stocks by Avg Volume"); axes[0, 2].set_xlabel("Avg Volume")
axes[0, 2].invert_yaxis()

# 4 - RSI distribution
df["rsi_14"].clip(0, 100).hist(bins=50, ax=axes[1, 0], color="orange", edgecolor="white")
axes[1, 0].set_title("RSI-14 Distribution"); axes[1, 0].set_xlabel("RSI")

# 5 - Intraday range distribution
df["daily_range"].clip(0, 0.1).hist(bins=50, ax=axes[1, 1], color="green", edgecolor="white")
axes[1, 1].set_title("Daily Range (High-Low)/Close"); axes[1, 1].set_xlabel("Range %")

# 6 - Direction balance
df["direction"].value_counts().plot.bar(ax=axes[1, 2], color=["#e74c3c", "#2ecc71"])
axes[1, 2].set_title("Next-Day Direction")
axes[1, 2].set_xticklabels(["Down (0)", "Up (1)"], rotation=0)
axes[1, 2].set_ylabel("Count")

plt.tight_layout()
plt.savefig(PLOT_DIR / "eda_overview.png", dpi=150, bbox_inches="tight")
plt.show()

# Correlation heatmap
feat_cols_plot = ["daily_return", "daily_range", "open_close_pct", "gap",
                  "sma_ratio_5_20", "vol_ratio", "rsi_14", "vol_5", "vol_10", "vol_20",
                  "dow", "month", "next_return", "direction"]
fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(df[feat_cols_plot].corr(), annot=True, fmt=".2f", cmap="coolwarm",
            ax=ax, linewidths=0.5, vmin=-1, vmax=1)
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(PLOT_DIR / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# Top 10 gainers & losers
total_ret = df.groupby("symbol").apply(
    lambda g: (g["close"].iloc[-1] / g["close"].iloc[0] - 1) * 100, include_groups=False
).sort_values()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
total_ret.tail(15).plot.barh(ax=ax1, color="green"); ax1.set_title("Top 15 Gainers (2014-2017)")
ax1.set_xlabel("Total Return %")
total_ret.head(15).plot.barh(ax=ax2, color="red"); ax2.set_title("Top 15 Losers (2014-2017)")
ax2.set_xlabel("Total Return %")
plt.tight_layout()
plt.savefig(PLOT_DIR / "gainers_losers.png", dpi=150, bbox_inches="tight")
plt.show()
print("\\u2705 All EDA plots saved")
"""))

# ── 4. Prepare classification features ────────────────────────────────────
cells.append(md("## 4 · Prepare Features for Direction Classification"))
cells.append(code("""feature_cols = [
    "daily_return", "daily_range", "open_close_pct", "gap",
    "sma_ratio_5_20", "vol_ratio", "rsi_14",
    "vol_5", "vol_10", "vol_20",
    "dow", "month"
]

X_dir = df[feature_cols].values
y_dir = df["direction"].values
print(f"Direction classification: {X_dir.shape}")
print(f"Up: {y_dir.sum()}, Down: {len(y_dir) - y_dir.sum()}")
print(f"Up rate: {y_dir.mean():.4f}")

# Time-based split: last 20% of dates for test
dates_sorted = df["date"].sort_values().unique()
cutoff = dates_sorted[int(len(dates_sorted) * 0.8)]
train_mask = df["date"] < cutoff
test_mask  = df["date"] >= cutoff
print(f"Train cutoff: {cutoff}")
print(f"Train: {train_mask.sum()}, Test: {test_mask.sum()}")

X_train_d = df.loc[train_mask, feature_cols].values
X_test_d  = df.loc[test_mask, feature_cols].values
y_train_d = df.loc[train_mask, "direction"].values
y_test_d  = df.loc[test_mask, "direction"].values

scaler_d = StandardScaler()
X_train_ds = scaler_d.fit_transform(X_train_d)
X_test_ds  = scaler_d.transform(X_test_d)
print(f"Train scaled: {X_train_ds.shape}, Test scaled: {X_test_ds.shape}")
"""))

# ── 5. Direction Classification (10 models) ──────────────────────────────
cells.append(md("## 5 · Price Direction Classification (10 Models)"))
cells.append(code("""classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=SEED),
    "AdaBoost": AdaBoostClassifier(n_estimators=150, random_state=SEED),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=5, eval_metric="logloss",
                              random_state=SEED, n_jobs=-1),
    "SVM": SVC(kernel="rbf", probability=True, random_state=SEED),
    "Naive Bayes": GaussianNB(),
    "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=SEED),
}

clf_results = {}
for name, model in classifiers.items():
    print(f"Training {name}...", end=" ")
    model.fit(X_train_ds, y_train_d)
    y_pred = model.predict(X_test_ds)
    acc = accuracy_score(y_test_d, y_pred)
    f1 = f1_score(y_test_d, y_pred, average="weighted")
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
ax.set_ylim(0, 1); ax.set_title("Price Direction - Model Comparison")
ax.legend(); plt.tight_layout()
plt.savefig(PLOT_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("\\u2705 Model comparison saved")
"""))

# ── 6. Daily Return Regression ──────────────────────────────────────────
cells.append(md("## 6 · Daily Return Regression"))
cells.append(code("""y_reg = df["next_return"].values

X_train_r = df.loc[train_mask, feature_cols].values
X_test_r  = df.loc[test_mask, feature_cols].values
y_train_r = df.loc[train_mask, "next_return"].values
y_test_r  = df.loc[test_mask, "next_return"].values

scaler_r = StandardScaler()
X_train_rs = scaler_r.fit_transform(X_train_r)
X_test_rs  = scaler_r.transform(X_test_r)

regressors = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.001, max_iter=2000),
    "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=2000),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=SEED),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=SEED),
}

reg_results = {}
for name, model in regressors.items():
    print(f"Training {name}...", end=" ")
    model.fit(X_train_rs, y_train_r)
    y_pred = model.predict(X_test_rs)
    r2   = r2_score(y_test_r, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred))
    mae  = mean_absolute_error(y_test_r, y_pred)
    reg_results[name] = {"r2": r2, "rmse": rmse, "mae": mae, "model": model, "y_pred": y_pred}
    print(f"R2={r2:.6f}  RMSE={rmse:.6f}  MAE={mae:.6f}")

best_reg_name = max(reg_results, key=lambda k: reg_results[k]["r2"])
print(f"\\n\\U0001f3c6 Best regressor: {best_reg_name} (R2={reg_results[best_reg_name]['r2']:.6f})")

# Actual vs Predicted
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, (name, res) in zip(axes.flat, reg_results.items()):
    ax.scatter(y_test_r, res["y_pred"], alpha=0.05, s=2, c="teal")
    lims = [-0.08, 0.08]
    ax.plot(lims, lims, "r--", linewidth=1.5)
    ax.set_title(f"{name}\\nR2={res['r2']:.6f}")
    ax.set_xlabel("Actual Return"); ax.set_ylabel("Predicted Return")
    ax.set_xlim(lims); ax.set_ylim(lims)
plt.suptitle("Daily Return Regression - Actual vs Predicted", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(PLOT_DIR / "return_regression_results.png", dpi=150, bbox_inches="tight")
plt.show()

# Feature importance
best_reg = reg_results[best_reg_name]["model"]
if hasattr(best_reg, "feature_importances_"):
    imp = best_reg.feature_importances_
    idx = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(np.array(feature_cols)[idx], imp[idx], color="darkcyan")
    ax.set_title(f"Return Regression - Feature Importance ({best_reg_name})")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "return_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
print("\\u2705 Return regression plots saved")
"""))

# ── 7. Volatility Classification ────────────────────────────────────────
cells.append(md("## 7 · Volatility Classification (Low / Medium / High)"))
cells.append(code("""# Build stock-level features for volatility classification
stock_feats = df.groupby("symbol").agg(
    mean_return=("daily_return", "mean"),
    std_return=("daily_return", "std"),
    mean_range=("daily_range", "mean"),
    mean_volume=("volume", "mean"),
    mean_close=("close", "mean"),
    mean_rsi=("rsi_14", "mean"),
    total_return=("daily_return", "sum"),
    mean_gap=("gap", "mean"),
    mean_vol_ratio=("vol_ratio", "mean"),
    count=("close", "count"),
).reset_index()

# Classify volatility into 3 classes by std_return terciles
stock_feats["vol_class"] = pd.qcut(stock_feats["std_return"], q=3,
                                    labels=["Low", "Medium", "High"])
print(f"Volatility class distribution:\\n{stock_feats['vol_class'].value_counts().to_string()}")

vol_features = ["mean_return", "mean_range", "mean_volume", "mean_close",
                "mean_rsi", "total_return", "mean_gap", "mean_vol_ratio"]

X_vol = stock_feats[vol_features].values
le_vol = LabelEncoder()
y_vol = le_vol.fit_transform(stock_feats["vol_class"])
print(f"\\nClasses: {le_vol.classes_}")
print(f"Shape: {X_vol.shape}")

X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(
    X_vol, y_vol, test_size=0.2, random_state=SEED, stratify=y_vol
)
scaler_v = StandardScaler()
X_train_vs = scaler_v.fit_transform(X_train_v)
X_test_vs  = scaler_v.transform(X_test_v)

vol_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=SEED),
    "XGBoost": XGBClassifier(n_estimators=200, eval_metric="mlogloss", random_state=SEED, n_jobs=-1),
    "SVM": SVC(kernel="rbf", probability=True, random_state=SEED),
    "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=SEED),
}

vol_results = {}
for name, model in vol_models.items():
    print(f"Training {name}...", end=" ")
    model.fit(X_train_vs, y_train_v)
    y_pred = model.predict(X_test_vs)
    acc = accuracy_score(y_test_v, y_pred)
    f1  = f1_score(y_test_v, y_pred, average="weighted")
    vol_results[name] = {"accuracy": acc, "f1": f1, "model": model, "y_pred": y_pred}
    print(f"Acc={acc:.4f}  F1={f1:.4f}")

best_vol_name = max(vol_results, key=lambda k: vol_results[k]["f1"])
print(f"\\n\\U0001f3c6 Best volatility classifier: {best_vol_name} (F1={vol_results[best_vol_name]['f1']:.4f})")

fig, ax = plt.subplots(figsize=(10, 5))
names_v = list(vol_results.keys())
f1s_v   = [vol_results[n]["f1"] for n in names_v]
ax.barh(names_v, f1s_v, color=sns.color_palette("rocket", len(names_v)))
ax.set_title("Volatility Classification - F1 Scores"); ax.set_xlabel("F1 (weighted)")
ax.set_xlim(0, 1)
for i, v in enumerate(f1s_v):
    ax.text(v + 0.01, i, f"{v:.4f}", va="center")
plt.tight_layout()
plt.savefig(PLOT_DIR / "volatility_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("\\u2705 Volatility classification plot saved")
"""))

# ── 8. Stock Clustering ──────────────────────────────────────────────────
cells.append(md("## 8 · Stock Clustering"))
cells.append(code("""clust_features = ["mean_return", "std_return", "mean_range", "mean_volume",
                  "mean_close", "mean_rsi", "total_return", "mean_gap", "mean_vol_ratio"]
X_clust = stock_feats[clust_features].values
scaler_cl = StandardScaler()
X_clust_s = scaler_cl.fit_transform(X_clust)

K_range = range(2, 11)
inertias, sils = [], []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km.fit_predict(X_clust_s)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_clust_s, labels))

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
stock_feats["cluster"] = km_final.fit_predict(X_clust_s)

# Cluster profiles
cluster_profile = stock_feats.groupby("cluster")[clust_features].mean()
print("\\nCluster profiles:")
print(cluster_profile.round(4).to_string())

# Cluster sizes
print(f"\\nCluster sizes: {stock_feats['cluster'].value_counts().sort_index().to_dict()}")

fig, ax = plt.subplots(figsize=(12, 6))
cp_norm = (cluster_profile - cluster_profile.min()) / (cluster_profile.max() - cluster_profile.min() + 1e-9)
cp_norm.T.plot(kind="bar", ax=ax, colormap="viridis")
ax.set_title(f"Stock Cluster Profiles (k={best_k})")
ax.set_ylabel("Normalized value"); ax.set_xlabel("Feature")
ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1)); ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(PLOT_DIR / "clustering_results.png", dpi=150, bbox_inches="tight")
plt.show()

# Show sample stocks per cluster
for c in sorted(stock_feats["cluster"].unique()):
    symbols = stock_feats[stock_feats["cluster"] == c]["symbol"].tolist()
    print(f"Cluster {c} ({len(symbols)} stocks): {symbols[:8]}{'...' if len(symbols) > 8 else ''}")
print("\\u2705 Clustering plots saved")
"""))

# ── 9. Hyperparameter Tuning ─────────────────────────────────────────────
cells.append(md("## 9 · Hyperparameter Tuning"))
cells.append(code("""# Use a subsample for tuning (direction classification is large)
np.random.seed(SEED)
sub_idx = np.random.choice(len(X_train_ds), size=min(50000, len(X_train_ds)), replace=False)
X_sub = X_train_ds[sub_idx]
y_sub = y_train_d[sub_idx]

# GridSearchCV - Random Forest
print("GridSearchCV on Random Forest (subsample)...")
rf_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, 15],
    "min_samples_split": [5, 10],
}
gs_rf = GridSearchCV(RandomForestClassifier(random_state=SEED, n_jobs=-1),
                     rf_grid, cv=3, scoring="f1_weighted", n_jobs=-1)
gs_rf.fit(X_sub, y_sub)
print(f"  Best params: {gs_rf.best_params_}")
print(f"  Best CV F1:  {gs_rf.best_score_:.4f}")

# RandomizedSearchCV - Gradient Boosting
print("\\nRandomizedSearchCV on Gradient Boosting (subsample)...")
gb_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
}
rs_gb = RandomizedSearchCV(GradientBoostingClassifier(random_state=SEED),
                           gb_dist, n_iter=15, cv=3, scoring="f1_weighted",
                           random_state=SEED, n_jobs=-1)
rs_gb.fit(X_sub, y_sub)
print(f"  Best params: {rs_gb.best_params_}")
print(f"  Best CV F1:  {rs_gb.best_score_:.4f}")

# Evaluate tuned models on full test set
for label, model in [("Tuned RF (Grid)", gs_rf.best_estimator_),
                     ("Tuned GB (Random)", rs_gb.best_estimator_)]:
    model.fit(X_train_ds, y_train_d)
    y_pred = model.predict(X_test_ds)
    acc = accuracy_score(y_test_d, y_pred)
    f1  = f1_score(y_test_d, y_pred, average="weighted")
    clf_results[label] = {"accuracy": acc, "f1": f1, "model": model, "y_pred": y_pred}
    print(f"  {label}: Acc={acc:.4f}  F1={f1:.4f}")

print("\\n\\u2705 Hyperparameter tuning complete")
"""))

# ── 10. CV + Confusion + Learning curves ────────────────────────────────
cells.append(md("## 10 · Cross-Validation, Feature Importance, Confusion Matrices & Learning Curves"))
cells.append(code("""# 5-fold CV (on subsample for speed)
cv_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=SEED),
    "XGBoost": XGBClassifier(n_estimators=100, max_depth=5, eval_metric="logloss",
                              random_state=SEED, n_jobs=-1),
}

# Subsample for CV
np.random.seed(SEED)
cv_idx = np.random.choice(len(X_dir), size=min(60000, len(X_dir)), replace=False)
X_cv_sub = scaler_d.transform(X_dir[cv_idx])
y_cv_sub = y_dir[cv_idx]

cv_scores = {}
for name, model in cv_models.items():
    scores = cross_val_score(model, X_cv_sub, y_cv_sub, cv=5, scoring="f1_weighted", n_jobs=-1)
    cv_scores[name] = scores
    print(f"{name}: mean F1={scores.mean():.4f} +/- {scores.std():.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.boxplot(cv_scores.values(), labels=cv_scores.keys())
ax.set_title("5-Fold Cross-Validation F1 Scores (Direction)")
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
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, name in zip(axes, top4):
    cm = confusion_matrix(y_test_d, clf_results[name]["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Down", "Up"], yticklabels=["Down", "Up"])
    ax.set_title(f"{name}\\nF1={clf_results[name]['f1']:.4f}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.suptitle("Confusion Matrices - Top 4 Models", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(PLOT_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()

# Learning curves (on subsample)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (name, model) in zip(axes, [
    ("Random Forest", RandomForestClassifier(n_estimators=50, max_depth=10, random_state=SEED, n_jobs=-1)),
    ("Gradient Boosting", GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=SEED)),
]):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_cv_sub, y_cv_sub, cv=3, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 6), scoring="f1_weighted"
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

# ── 11. Ensembles ─────────────────────────────────────────────────────────
cells.append(md("## 11 · Voting & Stacking Ensembles"))
cells.append(code("""# Voting Classifier
print("Training Voting Classifier...")
voting = VotingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=15, random_state=SEED, n_jobs=-1)),
        ("gb", GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=SEED)),
        ("xgb", XGBClassifier(n_estimators=200, max_depth=5, eval_metric="logloss",
                               random_state=SEED, n_jobs=-1)),
    ],
    voting="soft"
)
voting.fit(X_train_ds, y_train_d)
y_pred_v = voting.predict(X_test_ds)
acc_v = accuracy_score(y_test_d, y_pred_v)
f1_v  = f1_score(y_test_d, y_pred_v, average="weighted")
clf_results["Voting Ensemble"] = {"accuracy": acc_v, "f1": f1_v, "model": voting, "y_pred": y_pred_v}
print(f"  Voting: Acc={acc_v:.4f}  F1={f1_v:.4f}")

# Stacking Classifier
print("Training Stacking Classifier...")
stacking = StackingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1)),
        ("gb", GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=SEED)),
        ("xgb", XGBClassifier(n_estimators=100, max_depth=5, eval_metric="logloss",
                               random_state=SEED, n_jobs=-1)),
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
    cv=3, n_jobs=-1
)
stacking.fit(X_train_ds, y_train_d)
y_pred_s = stacking.predict(X_test_ds)
acc_s = accuracy_score(y_test_d, y_pred_s)
f1_s  = f1_score(y_test_d, y_pred_s, average="weighted")
clf_results["Stacking Ensemble"] = {"accuracy": acc_s, "f1": f1_s, "model": stacking, "y_pred": y_pred_s}
print(f"  Stacking: Acc={acc_s:.4f}  F1={f1_s:.4f}")

# Final ranking
print("\\n" + "="*60)
print("FINAL MODEL RANKING - Price Direction Classification")
print("="*60)
ranking = sorted(clf_results.items(), key=lambda x: x[1]["f1"], reverse=True)
for i, (name, res) in enumerate(ranking, 1):
    print(f"  {i:>2}. {name:<25s} Acc={res['accuracy']:.4f}  F1={res['f1']:.4f}")
best_overall = ranking[0][0]
print(f"\\n\\U0001f3c6 Best overall: {best_overall} (F1={clf_results[best_overall]['f1']:.4f})")

print("\\n" + "="*60)
print("VOLATILITY CLASSIFICATION RANKING")
print("="*60)
v_ranking = sorted(vol_results.items(), key=lambda x: x[1]["f1"], reverse=True)
for i, (name, res) in enumerate(v_ranking, 1):
    print(f"  {i:>2}. {name:<25s} Acc={res['accuracy']:.4f}  F1={res['f1']:.4f}")
"""))

# ── 12. HTML Report ───────────────────────────────────────────────────────
cells.append(md("## 12 · Generate HTML Report"))

html_cell = '''def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

images = {}
for p in sorted(PLOT_DIR.glob("*.png")):
    images[p.stem] = img_to_base64(p)

TEMPLATE = ""\"<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>S&P 500 Stocks - ML Report</title>
<style>
:root{--bg:#0f172a;--card:#1e293b;--accent:#10b981;--text:#e2e8f0;--muted:#94a3b8}
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
.best{background:#10b98115;font-weight:700}
img{width:100%;border-radius:8px;margin:.8rem 0}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem}
@media(max-width:800px){.grid2{grid-template-columns:1fr}}
</style></head><body>
<h1>S&P 500 Stock Prices - ML Report</h1>
<p class="subtitle">497,472 records - 505 stocks - 2014 to 2017</p>

<div class="card"><h2>Exploratory Data Analysis</h2>
<img src="data:image/png;base64,{{images.eda_overview}}" alt="EDA Overview">
<div class="grid2">
<img src="data:image/png;base64,{{images.correlation_heatmap}}" alt="Correlation Heatmap">
<img src="data:image/png;base64,{{images.gainers_losers}}" alt="Gainers and Losers">
</div></div>

<div class="card"><h2>Task 1 - Price Direction Classification</h2>
<table><tr><th>#</th><th>Model</th><th>Accuracy</th><th>F1 (weighted)</th></tr>
{% for name, res in clf_ranking %}
<tr{% if loop.first %} class="best"{% endif %}>
<td>{{loop.index}}</td><td>{{name}}</td>
<td>{{"{:.4f}".format(res.accuracy)}}</td><td>{{"{:.4f}".format(res.f1)}}</td></tr>
{% endfor %}</table>
<img src="data:image/png;base64,{{images.model_comparison}}" alt="Model Comparison">
</div>

<div class="card"><h2>Task 2 - Daily Return Regression</h2>
<table><tr><th>#</th><th>Model</th><th>R2</th><th>RMSE</th><th>MAE</th></tr>
{% for name, res in reg_ranking %}
<tr{% if loop.first %} class="best"{% endif %}>
<td>{{loop.index}}</td><td>{{name}}</td>
<td>{{"{:.6f}".format(res.r2)}}</td><td>{{"{:.6f}".format(res.rmse)}}</td>
<td>{{"{:.6f}".format(res.mae)}}</td></tr>
{% endfor %}</table>
<div class="grid2">
{% if images.return_feature_importance %}
<img src="data:image/png;base64,{{images.return_feature_importance}}" alt="Return Feature Importance">
{% endif %}
<img src="data:image/png;base64,{{images.return_regression_results}}" alt="Return Regression">
</div></div>

<div class="card"><h2>Task 3 - Volatility Classification (Low / Medium / High)</h2>
<table><tr><th>#</th><th>Model</th><th>Accuracy</th><th>F1 (weighted)</th></tr>
{% for name, res in vol_ranking %}
<tr{% if loop.first %} class="best"{% endif %}>
<td>{{loop.index}}</td><td>{{name}}</td>
<td>{{"{:.4f}".format(res.accuracy)}}</td><td>{{"{:.4f}".format(res.f1)}}</td></tr>
{% endfor %}</table>
<img src="data:image/png;base64,{{images.volatility_comparison}}" alt="Volatility Comparison">
</div>

<div class="card"><h2>Task 4 - Stock Clustering</h2>
<p>Best k={{best_k}}, Silhouette={{"{:.4f}".format(best_sil)}}</p>
<div class="grid2">
<img src="data:image/png;base64,{{images.elbow_silhouette}}" alt="Elbow and Silhouette">
<img src="data:image/png;base64,{{images.clustering_results}}" alt="Clustering Results">
</div></div>

<div class="card"><h2>Hyperparameter Tuning & Cross-Validation</h2>
<div class="grid2">
<img src="data:image/png;base64,{{images.cv_comparison}}" alt="CV Comparison">
{% if images.feature_importance %}
<img src="data:image/png;base64,{{images.feature_importance}}" alt="Feature Importance">
{% endif %}
</div>
<img src="data:image/png;base64,{{images.confusion_matrices}}" alt="Confusion Matrices">
<img src="data:image/png;base64,{{images.learning_curves}}" alt="Learning Curves">
</div>

</body></html>""\"

from types import SimpleNamespace
clf_ranking = [(n, SimpleNamespace(**{k: v for k, v in r.items() if k not in ("model", "y_pred")}))
               for n, r in sorted(clf_results.items(), key=lambda x: x[1]["f1"], reverse=True)]
reg_ranking = [(n, SimpleNamespace(**{k: v for k, v in r.items() if k not in ("model", "y_pred")}))
               for n, r in sorted(reg_results.items(), key=lambda x: x[1]["r2"], reverse=True)]
vol_ranking = [(n, SimpleNamespace(**{k: v for k, v in r.items() if k not in ("model", "y_pred")}))
               for n, r in sorted(vol_results.items(), key=lambda x: x[1]["f1"], reverse=True)]

html = jinja2.Template(TEMPLATE).render(
    images=images,
    clf_ranking=clf_ranking,
    reg_ranking=reg_ranking,
    vol_ranking=vol_ranking,
    best_k=best_k, best_sil=max(sils),
)

out_path = pathlib.Path("outputs/stocks_ml_report.html")
out_path.write_text(html)
print(f"Report generated: {out_path}")
print(f"   File size: {out_path.stat().st_size / 1024:.1f} KB")
print(f"   Embedded images: {len(images)}")'''

# Fix the escaped triple quotes
html_cell = html_cell.replace('""\\\"', '\"\"\"')
cells.append(code(html_cell))

# ── Build notebook ────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14.3"}
    },
    "cells": cells
}

out = pathlib.Path("stocks_ml_analysis.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Notebook written: {out}  ({len(cells)} cells)")
