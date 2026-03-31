"""Data loading and generation package."""

from data.synthetic import (
    make_blobs_dataset,
    make_moons_dataset,
    make_circles_dataset,
    make_anisotropic_dataset,
    make_varied_variance_dataset,
    make_unbalanced_dataset,
    ALL_SYNTHETIC,
)
from data.real_datasets import get_real_dataset, ALL_REAL
from data.preprocessor import scale_data, reduce_dimensions
from data.noise_injector import inject_noise, inject_outliers

__all__ = [
    "make_blobs_dataset", "make_moons_dataset", "make_circles_dataset",
    "make_anisotropic_dataset", "make_varied_variance_dataset",
    "make_unbalanced_dataset", "ALL_SYNTHETIC",
    "get_real_dataset", "ALL_REAL",
    "scale_data", "reduce_dimensions",
    "inject_noise", "inject_outliers",
]
