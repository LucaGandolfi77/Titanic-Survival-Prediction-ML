# Core ML and classical models

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

# ─────────────────────────────────────────────────────────
#  Classical ML Models
# ─────────────────────────────────────────────────────────

class MLModel:
    """Wrapper for classical ML models with unified interface."""
    
    def __init__(self, name: str, model_type: str, params: Dict[str, Any]):
        self.name = name
        self.model_type = model_type
        self.params = params
        self.model = self._build_model()
    
    def _build_model(self):
        if self.model_type == "logistic_regression":
            return LogisticRegression(**self.params)
        elif self.model_type == "random_forest":
            return RandomForestClassifier(**self.params)
        elif self.model_type == "svm":
            return SVC(**self.params, probability=True)
        elif self.model_type == "knn":
            return KNeighborsClassifier(**self.params)
        elif self.model_type == "xgboost":
            return xgb.XGBClassifier(**self.params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # SVM decision_function -> probabilities (approximate)
            scores = self.model.decision_function(X)
            proba = 1 / (1 + np.exp(-scores))
            return np.column_stack([1 - proba, proba])

# ─────────────────────────────────────────────────────────
#  Neural Networks (PyTorch)
# ─────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Multi-layer Perceptron for classification."""
    
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int, 
                 dropout_rate: float = 0.2):
        super().__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class CNN(nn.Module):
    """Convolutional Neural Network for image data (MNIST, Fashion-MNIST)."""
    
    def __init__(self, num_classes: int = 10, img_size: int = 28):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * (img_size // 4) ** 2, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
