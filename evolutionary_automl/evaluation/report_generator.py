"""
Report generator: produces Markdown and LaTeX-ready comparison tables.

Generates tables from experiment results that can be directly included
in a thesis document. Supports bolding of best results per row and
significance markers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


def generate_markdown_table(
    results: Dict[str, Dict[str, Dict[str, float]]],
    metric: str = "f1",
    output_path: Path | None = None,
) -> str:
    """Generate a Markdown comparison table.

    Args:
        results: Nested dict {dataset: {method: {metric: value, ...}}}.
        metric: Which metric to display.
        output_path: If provided, save the table to this file.

    Returns:
        Markdown table as a string.
    """
    datasets = list(results.keys())
    methods = list(next(iter(results.values())).keys())

    header = f"| Dataset | {' | '.join(methods)} |"
    separator = "|" + "|".join(["---"] * (len(methods) + 1)) + "|"

    rows = [header, separator]
    for ds in datasets:
        values = []
        scores = []
        for m in methods:
            val = results[ds][m].get(metric, 0.0)
            scores.append(val)
        best = max(scores)

        cells = []
        for m, s in zip(methods, scores):
            mean = results[ds][m].get(f"{metric}_mean", s)
            std = results[ds][m].get(f"{metric}_std", 0.0)
            cell = f"{mean:.4f} ± {std:.4f}"
            if abs(s - best) < 1e-6:
                cell = f"**{cell}**"
            cells.append(cell)

        rows.append(f"| {ds} | {' | '.join(cells)} |")

    table = "\n".join(rows)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(table)
    return table


def generate_latex_table(
    results: Dict[str, Dict[str, Dict[str, float]]],
    metric: str = "f1",
    caption: str = "Comparison of optimization methods",
    label: str = "tab:comparison",
    output_path: Path | None = None,
) -> str:
    """Generate a LaTeX-ready comparison table.

    Args:
        results: Nested dict {dataset: {method: {metric: value, ...}}}.
        metric: Which metric to display.
        caption: Table caption.
        label: LaTeX label.
        output_path: If provided, save to file.

    Returns:
        LaTeX table as a string.
    """
    datasets = list(results.keys())
    methods = list(next(iter(results.values())).keys())
    n_methods = len(methods)

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    col_spec = "l" + "c" * n_methods
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    header = "Dataset & " + " & ".join(methods) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for ds in datasets:
        scores = []
        for m in methods:
            scores.append(results[ds][m].get(metric, 0.0))
        best = max(scores)

        cells = []
        for m, s in zip(methods, scores):
            mean = results[ds][m].get(f"{metric}_mean", s)
            std = results[ds][m].get(f"{metric}_std", 0.0)
            cell = f"{mean:.4f} $\\pm$ {std:.4f}"
            if abs(s - best) < 1e-6:
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)

        lines.append(f"{ds} & {' & '.join(cells)} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    table = "\n".join(lines)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(table)
    return table


if __name__ == "__main__":
    sample = {
        "iris": {
            "GA": {"f1": 0.96, "f1_mean": 0.96, "f1_std": 0.02},
            "RandomSearch": {"f1": 0.93, "f1_mean": 0.93, "f1_std": 0.03},
            "GridSearch": {"f1": 0.94, "f1_mean": 0.94, "f1_std": 0.01},
        },
        "breast_cancer": {
            "GA": {"f1": 0.97, "f1_mean": 0.97, "f1_std": 0.01},
            "RandomSearch": {"f1": 0.95, "f1_mean": 0.95, "f1_std": 0.02},
            "GridSearch": {"f1": 0.96, "f1_mean": 0.96, "f1_std": 0.01},
        },
    }
    print(generate_markdown_table(sample))
    print()
    print(generate_latex_table(sample))
