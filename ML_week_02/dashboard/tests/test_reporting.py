"""
test_reporting.py – Unit tests for summary generation and HTML rendering.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.reporting.summary_generator import (
    generate_executive_summary,
    generate_technical_report,
    to_plain_text,
)
from src.reporting.pdf_exporter import render_executive_html, render_technical_html


@pytest.fixture
def model_info():
    return {
        "model_type": "RandomForestClassifier",
        "n_features": 10,
        "task": "classification",
        "params": {"n_estimators": 100},
        "classes": [0, 1],
    }


@pytest.fixture
def performance():
    return {"accuracy": 0.92, "precision": 0.89, "recall": 0.88, "f1": 0.885, "roc_auc": 0.95}


@pytest.fixture
def fairness_results():
    return pd.DataFrame([
        {"protected_attribute": "gender", "metric": "demographic_parity_difference",
         "value": 0.05, "threshold": 0.1, "status": "✅ PASS"},
        {"protected_attribute": "gender", "metric": "disparate_impact_ratio",
         "value": 0.92, "threshold": 0.8, "status": "✅ PASS"},
    ])


class TestSummaryGenerator:
    def test_executive_summary(self, model_info, performance, fairness_results):
        summary = generate_executive_summary(model_info, performance, fairness_results)
        assert summary["model_type"] == "RandomForestClassifier"
        assert summary["fairness"]["verdict"] == "PASS"
        assert summary["primary_metric_value"] == 0.92

    def test_executive_summary_no_fairness(self, model_info, performance):
        summary = generate_executive_summary(model_info, performance)
        assert summary["fairness"] is None

    def test_technical_report(self, model_info, performance, fairness_results):
        report = generate_technical_report(
            model_info, performance, fairness_results=fairness_results,
        )
        assert report["model"]["type"] == "RandomForestClassifier"
        assert "accuracy" in report["performance"]

    def test_plain_text(self, model_info, performance):
        report = generate_technical_report(model_info, performance)
        text = to_plain_text(report)
        assert "MODEL" in text
        assert "PERFORMANCE" in text
        assert "accuracy" in text


class TestHTMLRendering:
    def test_executive_html(self, model_info, performance, fairness_results):
        summary = generate_executive_summary(model_info, performance, fairness_results)
        html = render_executive_html(summary)
        assert "<html" in html
        assert "Executive Summary" in html

    def test_technical_html(self, model_info, performance):
        report = generate_technical_report(model_info, performance)
        html = render_technical_html(report)
        assert "<html" in html
        assert "Technical Report" in html
