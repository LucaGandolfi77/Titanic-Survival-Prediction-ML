"""
HTML report generation using Jinja2 + Plotly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)

try:
    from jinja2 import Environment, FileSystemLoader
except ImportError:
    Environment = None  # type: ignore

try:
    import plotly.io as pio
except ImportError:
    pio = None  # type: ignore

_TEMPLATE_DIR = Path(__file__).parent / "templates"


class HTMLReportGenerator:
    """Render a self-contained HTML report from pipeline results."""

    def __init__(self, config) -> None:
        self.output_path: str = getattr(config, "output_path", "outputs/report.html")
        self.title: str = getattr(config, "title", "AutoML-Lite Report")

    def generate(self, results: Dict[str, Any]) -> Path:
        """Write report.html and return its path."""
        if Environment is None:
            raise ImportError("pip install jinja2")

        env = Environment(
            loader=FileSystemLoader(str(_TEMPLATE_DIR)),
            autoescape=True,
        )
        template = env.get_template("report.html.j2")

        # Convert plotly figures to HTML div strings
        charts = {}
        for key in ("importance_chart", "confusion_matrix_chart"):
            fig = results.get(key)
            if fig is not None and pio is not None:
                try:
                    charts[key] = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
                except Exception:
                    charts[key] = ""
            else:
                charts[key] = ""

        context = {
            "title": self.title,
            "task": results.get("task", "unknown"),
            "dataset_shape": results.get("dataset_shape", ""),
            "profile_summary": results.get("profile_summary", {}),
            "screening_table": results.get("screening_results", []),
            "best_params": results.get("best_params", {}),
            "metrics": results.get("metrics", {}),
            "feature_importance_html": charts.get("importance_chart", ""),
            "confusion_matrix_html": charts.get("confusion_matrix_chart", ""),
            "ensemble_method": results.get("ensemble_method", ""),
        }

        html = template.render(**context)
        out = Path(self.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        logger.info(f"Report saved → {out}")
        return out
