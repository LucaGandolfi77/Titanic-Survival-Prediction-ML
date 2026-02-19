"""ML Playground – Interactive ML Training Dashboard."""
from setuptools import setup, find_packages
from pathlib import Path

HERE = Path(__file__).resolve().parent

setup(
    name="ml-playground",
    version="1.0.0",
    description="Interactive ML Playground built with Streamlit",
    long_description=(HERE / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Luca Gandolfi",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=(HERE / "requirements.txt").read_text().splitlines(),
    entry_points={
        "console_scripts": [
            "ml-playground=app:main",
        ],
    },
)
