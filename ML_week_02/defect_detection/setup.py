"""Setup script for defect_detection package."""
from setuptools import setup, find_packages

setup(
    name="defect_detection",
    version="1.0.0",
    description="Real-Time Manufacturing Defect Detection with YOLOv8",
    author="Luca Gandolfi",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "ultralytics>=8.1.0",
        "torch>=2.1.0",
        "opencv-python-headless>=4.9.0",
        "fastapi>=0.108.0",
        "uvicorn[standard]>=0.25.0",
        "streamlit>=1.30.0",
        "albumentations>=1.3.1",
        "pyyaml>=6.0.1",
        "loguru>=0.7.2",
    ],
    extras_require={
        "dev": ["pytest>=8.0", "pytest-cov", "pytest-asyncio"],
    },
)
