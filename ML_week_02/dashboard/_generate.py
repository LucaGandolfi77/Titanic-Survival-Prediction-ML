#!/usr/bin/env python3
"""
_generate.py – Generate sample datasets and pre-trained models for the XAI dashboard.

Run: python _generate.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

np.random.seed(42)


# ══════════════════════════════════════════════════════════════
# 1. Credit Risk dataset
# ══════════════════════════════════════════════════════════════
def generate_credit_risk():
    print("Generating credit_risk_sample.csv …")
    n = 2000
    X, y = make_classification(
        n_samples=n, n_features=12, n_informative=8, n_redundant=2,
        n_classes=2, weights=[0.7, 0.3], random_state=42,
    )
    df = pd.DataFrame(X, columns=[
        "income", "debt_ratio", "credit_score", "employment_years",
        "loan_amount", "interest_rate", "num_accounts", "delinquencies",
        "credit_utilization", "monthly_payment", "loan_term", "savings",
    ])
    # Add protected attributes
    df["age"] = np.random.choice(["<25", "25-40", "40-60", ">60"], n, p=[0.15, 0.35, 0.35, 0.15])
    df["gender"] = np.random.choice(["M", "F"], n, p=[0.55, 0.45])
    df["default"] = y
    # Scale some features to realistic ranges
    df["income"] = (df["income"] * 15000 + 50000).round(0)
    df["credit_score"] = (df["credit_score"] * 100 + 650).clip(300, 850).round(0)
    df["employment_years"] = (df["employment_years"] * 5 + 8).clip(0, 40).round(1)
    df["loan_amount"] = (df["loan_amount"] * 20000 + 30000).clip(1000, 200000).round(0)
    df["interest_rate"] = (df["interest_rate"] * 5 + 10).clip(2, 30).round(2)
    df.to_csv(DATA_DIR / "credit_risk_sample.csv", index=False)
    return df


# ══════════════════════════════════════════════════════════════
# 2. Medical Diagnosis dataset
# ══════════════════════════════════════════════════════════════
def generate_medical():
    print("Generating medical_sample.csv …")
    n = 1500
    X, y = make_classification(
        n_samples=n, n_features=10, n_informative=7, n_redundant=1,
        n_classes=2, weights=[0.6, 0.4], random_state=123,
    )
    df = pd.DataFrame(X, columns=[
        "blood_pressure", "cholesterol", "glucose", "bmi",
        "heart_rate", "exercise_hours", "sleep_hours",
        "stress_level", "medication_adherence", "family_history_score",
    ])
    df["age_group"] = np.random.choice(["18-30", "30-50", "50-70", "70+"], n, p=[0.2, 0.3, 0.3, 0.2])
    df["sex"] = np.random.choice(["M", "F"], n, p=[0.5, 0.5])
    df["diagnosis"] = y
    # Scale
    df["blood_pressure"] = (df["blood_pressure"] * 20 + 120).clip(80, 200).round(0)
    df["cholesterol"] = (df["cholesterol"] * 40 + 200).clip(100, 350).round(0)
    df["glucose"] = (df["glucose"] * 30 + 100).clip(60, 300).round(0)
    df["bmi"] = (df["bmi"] * 8 + 25).clip(15, 50).round(1)
    df.to_csv(DATA_DIR / "medical_sample.csv", index=False)
    return df


# ══════════════════════════════════════════════════════════════
# 3. Train & save models
# ══════════════════════════════════════════════════════════════
def train_and_save(df: pd.DataFrame, name: str, target: str, protected: list):
    print(f"Training {name} …")
    feature_cols = [c for c in df.columns if c != target and c not in protected]
    X = df[feature_cols].copy()
    y = df[target].copy()

    # Encode categoricals
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = X[col].astype("category").cat.codes

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    perf = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    model_path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, model_path)
    print(f"  Saved → {model_path}")

    metadata = {
        "model_type": type(model).__name__,
        "features": list(X.columns),
        "target": target,
        "protected_attributes": protected,
        "n_features": len(X.columns),
        "performance": perf,
    }
    meta_path = MODELS_DIR / f"{name}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata → {meta_path}")
    print(f"  Accuracy: {perf['accuracy']}, F1: {perf['f1']}, AUC: {perf['roc_auc']}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    credit_df = generate_credit_risk()
    medical_df = generate_medical()

    train_and_save(credit_df, "credit_risk_model", "default", ["age", "gender"])
    train_and_save(medical_df, "medical_diagnosis_model", "diagnosis", ["age_group", "sex"])

    print("\n✅ All datasets and models generated!")
