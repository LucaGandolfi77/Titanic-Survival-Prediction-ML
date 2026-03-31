"""Evaluation sub-package: statistical tests and report generation."""

from evaluation.statistical_tests import (
    friedman_posthoc,
    pairwise_wilcoxon,
    cohens_d,
)
from evaluation.report_generator import generate_markdown_report

__all__ = [
    "friedman_posthoc",
    "pairwise_wilcoxon",
    "cohens_d",
    "generate_markdown_report",
]
