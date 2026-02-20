"""Setup for AutoML-Lite."""
from setuptools import setup, find_packages

setup(
    name="automl-lite",
    version="1.0.0",
    author="Luca Gandolfi",
    description="Automated ML pipeline for tabular data",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "scikit-learn>=1.4.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.2.0",
        "catboost>=1.2.2",
        "optuna>=3.5.0",
        "numpy>=1.26.0",
        "pandas>=2.1.0",
        "scipy>=1.12.0",
        "mlflow>=2.10.0",
        "jinja2>=3.1.3",
        "plotly>=5.18.0",
        "pyyaml>=6.0.1",
        "joblib>=1.3.2",
        "loguru>=0.7.2",
        "rich>=13.7.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=4.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "automl-lite=automl.__main__:main",
        ],
    },
)
