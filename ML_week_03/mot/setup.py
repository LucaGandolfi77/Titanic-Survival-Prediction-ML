"""Minimal setup.py for editable installs (pip install -e .)."""
from setuptools import find_packages, setup

setup(
    name="mot-tracker",
    version="0.1.0",
    description="Lightweight Mobile Object Tracker — Siamese network + TFLite",
    author="Your Name",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.14",
        "numpy>=1.24",
        "scipy>=1.11",
        "opencv-python>=4.8",
        "pyyaml>=6.0",
        "matplotlib>=3.8",
        "scikit-learn>=1.3",
    ],
)
