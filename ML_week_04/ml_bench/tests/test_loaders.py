# Unit tests for data loaders

import pytest
from src.data import load_dataset

def test_load_iris():
    """Test loading Iris dataset."""
    X_train, X_test, y_train, y_test, task = load_dataset("iris")
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]
    assert task == "classification"

def test_load_breast_cancer():
    """Test loading Breast Cancer dataset."""
    X_train, X_test, y_train, y_test, task = load_dataset("breast_cancer")
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert task == "classification"

def test_load_housing():
    """Test loading Housing dataset (regression)."""
    X_train, X_test, y_train, y_test, task = load_dataset("housing")
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert task == "regression"
