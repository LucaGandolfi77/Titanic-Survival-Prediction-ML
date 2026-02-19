"""Neural Network From Scratch — pip-installable package."""

from setuptools import setup, find_packages
from pathlib import Path

ROOT = Path(__file__).parent

setup(
    name="nn_from_scratch",
    version="1.0.0",
    description="Educational neural network framework built with NumPy only",
    author="Luca Gandolfi",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "matplotlib>=3.8.0",
        "scikit-learn>=1.3.0",
    ],
    entry_points={
        "console_scripts": [
            "train-mnist=train_mnist:main",
        ],
    },
)
