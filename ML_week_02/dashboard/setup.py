"""Explainable AI Dashboard."""
from setuptools import setup, find_packages
from pathlib import Path

HERE = Path(__file__).resolve().parent

setup(
    name="xai-dashboard",
    version="1.0.0",
    description="Explainable AI Dashboard – Model Transparency & Fairness Analysis",
    long_description=(HERE / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Luca Gandolfi",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in (HERE / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ],
)
