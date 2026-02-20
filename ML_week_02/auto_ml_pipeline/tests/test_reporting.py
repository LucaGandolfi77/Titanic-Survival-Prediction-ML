"""Tests for reporting modules."""

import pytest
from pathlib import Path

from automl.reporting.html_report import HTMLReportGenerator
from automl.reporting.api_generator import APIGenerator


class TestHTMLReport:

    def test_generate_report(self, tmp_path):
        class _Cfg:
            output_path = str(tmp_path / "report.html")
            title = "Test Report"

        gen = HTMLReportGenerator(_Cfg())
        results = {
            "task": "classification",
            "dataset_shape": "200 rows × 7 cols",
            "metrics": {"accuracy": 0.85, "f1": 0.82},
            "screening_results": [
                {"model": "rf", "mean_score": 0.83, "std_score": 0.02, "time_s": 1.0}
            ],
            "best_params": {"rf": {"n_estimators": 100}},
            "ensemble_method": "stacking",
        }
        path = gen.generate(results)
        assert path.exists()
        html = path.read_text()
        assert "Test Report" in html
        assert "0.8500" in html


class TestAPIGenerator:

    def test_generate_api(self, tmp_path):
        class _Cfg:
            output_path = str(tmp_path / "app.py")

        gen = APIGenerator(_Cfg())
        path = gen.generate(
            feature_names=["Age", "Fare", "Pclass"],
            model_path="model.pkl",
            task="classification",
        )
        assert path.exists()
        code = path.read_text()
        assert "FastAPI" in code
        assert "predict" in code
