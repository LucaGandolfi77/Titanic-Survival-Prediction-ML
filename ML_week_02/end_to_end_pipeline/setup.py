"""
setup.py — Package Installation
================================
Enables `pip install -e .` for development.
"""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent

setup(
    name="titanic-mlops",
    version="1.0.0",
    description="End-to-end MLOps pipeline for Titanic survival prediction",
    author="Luca Gandolfi",
    python_requires=">=3.10",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=ROOT.joinpath("requirements.txt").read_text().splitlines(),
    extras_require={
        "dev": [
            "pytest>=7.4",
            "pytest-cov>=4.1",
            "ruff>=0.1",
            "black>=23.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "titanic-train=train:main",
            "titanic-optimize=optimize:main",
            "titanic-serve=serve:main",
        ],
    },
)
