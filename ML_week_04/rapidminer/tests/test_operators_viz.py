"""
test_operators_viz.py – Tests for visualization operators.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from engine.operator_base import get_operator_class


class TestDataDistribution:
    def test_returns_figure(self, iris_df):
        op = get_operator_class("Data Distribution")()
        op.set_param("column", "sepal_length")
        result = op.execute({"in": iris_df})
        assert "figure" in result
        assert isinstance(result["figure"], plt.Figure)
        plt.close(result["figure"])

    def test_auto_column(self, iris_df):
        op = get_operator_class("Data Distribution")()
        result = op.execute({"in": iris_df})
        assert isinstance(result["figure"], plt.Figure)
        plt.close(result["figure"])


class TestScatterPlot:
    def test_returns_figure(self, iris_df):
        op = get_operator_class("Scatter Plot")()
        op.set_param("x", "sepal_length")
        op.set_param("y", "sepal_width")
        result = op.execute({"in": iris_df})
        assert isinstance(result["figure"], plt.Figure)
        plt.close(result["figure"])

    def test_with_color(self, iris_df):
        op = get_operator_class("Scatter Plot")()
        op.set_param("x", "sepal_length")
        op.set_param("y", "petal_length")
        op.set_param("color_by", "species")
        result = op.execute({"in": iris_df})
        assert isinstance(result["figure"], plt.Figure)
        plt.close(result["figure"])


class TestBoxPlot:
    def test_returns_figure(self, iris_df):
        op = get_operator_class("Box Plot")()
        op.set_param("x", "species")
        op.set_param("y", "sepal_length")
        result = op.execute({"in": iris_df})
        assert isinstance(result["figure"], plt.Figure)
        plt.close(result["figure"])


class TestCorrelationHeatmap:
    def test_returns_figure(self, iris_df):
        op = get_operator_class("Correlation Heatmap")()
        result = op.execute({"in": iris_df})
        assert isinstance(result["figure"], plt.Figure)
        plt.close(result["figure"])


class TestParallelCoordinates:
    def test_returns_figure(self, iris_df):
        op = get_operator_class("Parallel Coordinates")()
        op.set_param("color_column", "species")
        result = op.execute({"in": iris_df})
        assert isinstance(result["figure"], plt.Figure)
        plt.close(result["figure"])
