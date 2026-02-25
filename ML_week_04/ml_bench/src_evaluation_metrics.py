# Metrics and evaluation utilities

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, mean_squared_error
)
from typing import Dict, Any

def compute_classification_metrics(y_true, y_pred, y_pred_proba=None) -> Dict[str, float]:
    """Compute classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    
    if y_pred_proba is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
    
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    
    return metrics

def compute_regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute regression metrics."""
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": np.mean(np.abs(y_true - y_pred)),
    }
