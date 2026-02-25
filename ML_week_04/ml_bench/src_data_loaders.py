# Dataset loaders for ML benchmarking

from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from typing import Tuple

def load_iris() -> Tuple[np.ndarray, np.ndarray]:
    """Load Iris dataset."""
    data = datasets.load_iris()
    return data.data, data.target

def load_wine() -> Tuple[np.ndarray, np.ndarray]:
    """Load Wine dataset."""
    data = datasets.load_wine()
    return data.data, data.target

def load_breast_cancer() -> Tuple[np.ndarray, np.ndarray]:
    """Load Breast Cancer Wisconsin dataset."""
    data = datasets.load_breast_cancer()
    return data.data, data.target

def load_digits() -> Tuple[np.ndarray, np.ndarray]:
    """Load digits (8x8 images) dataset."""
    data = datasets.load_digits()
    return data.data, data.target

def load_housing() -> Tuple[np.ndarray, np.ndarray]:
    """Load California Housing (regression) dataset."""
    data = datasets.fetch_california_housing()
    return data.data, data.target

def load_dataset(dataset_name: str, train_size: float = 0.8, 
                 random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, 
                                                   np.ndarray, np.ndarray, str]:
    """
    Load and split dataset.
    
    Parameters
    ----------
    dataset_name : str
        Name: iris, wine, breast_cancer, digits, housing
    train_size : float
        Fraction for training
    random_state : int
        Random seed
    
    Returns
    -------
    X_train, X_test, y_train, y_test, task_type
    """
    loaders = {
        "iris": load_iris,
        "wine": load_wine,
        "breast_cancer": load_breast_cancer,
        "digits": load_digits,
        "housing": load_housing,
    }
    
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    X, y = loaders[dataset_name]()
    task_type = "regression" if dataset_name == "housing" else "classification"
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, task_type
