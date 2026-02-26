"""Breast-Cancer ML — minimal setup.py for editable installs."""

from setuptools import setup, find_packages

setup(
    name="breast-cancer-ml",
    version="1.0.0",
    description="Production-grade ML pipeline for Wisconsin Breast Cancer classification",
    author="LucaGandolfi77",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0"],
    },
)
