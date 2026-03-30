"""
Report Generator
================
Generate Markdown and LaTeX tables from experiment results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


def generate_markdown_table(
    results: Dict[str, Dict[str, Any]],
    metric: str = "mean_accuracy",
    title: str = "Results Comparison",
) -> str:
    """Generate a Markdown table comparing methods.

    Args:
        results: {method_name: {metric_key: value, ...}}
        metric: which metric to highlight
    """
    methods = list(results.keys())
    if not methods:
        return ""

    all_keys = set()
    for v in results.values():
        all_keys.update(v.keys())
    # Filter to numeric keys
    numeric_keys = sorted(k for k in all_keys
                          if isinstance(next(iter(results.values())).get(k, ""), (int, float)))

    lines = [f"## {title}", ""]
    header = "| Method | " + " | ".join(numeric_keys) + " |"
    sep = "|" + "|".join(["---"] * (len(numeric_keys) + 1)) + "|"
    lines.extend([header, sep])

    # Find best per column
    best = {}
    for k in numeric_keys:
        vals = [(m, results[m].get(k, float("-inf"))) for m in methods]
        best_method = max(vals, key=lambda x: x[1])[0]
        best[k] = best_method

    for method in methods:
        row = f"| {method} |"
        for k in numeric_keys:
            val = results[method].get(k, "—")
            if isinstance(val, float):
                cell = f" {val:.4f} "
                if best.get(k) == method:
                    cell = f" **{val:.4f}** "
            else:
                cell = f" {val} "
            row += cell + "|"
        lines.append(row)

    return "\n".join(lines)


def generate_latex_table(
    results: Dict[str, Dict[str, Any]],
    metric_keys: List[str] | None = None,
    caption: str = "Comparison of methods",
    label: str = "tab:comparison",
) -> str:
    """Generate a LaTeX table with \\textbf{} on best result per column."""
    methods = list(results.keys())
    if not methods:
        return ""

    if metric_keys is None:
        all_keys = set()
        for v in results.values():
            all_keys.update(v.keys())
        metric_keys = sorted(k for k in all_keys
                             if isinstance(next(iter(results.values())).get(k, ""), (int, float)))

    n_cols = len(metric_keys) + 1
    col_fmt = "l" + "c" * len(metric_keys)

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_fmt}}}",
        "\\toprule",
        "Method & " + " & ".join(k.replace("_", "\\_") for k in metric_keys) + " \\\\",
        "\\midrule",
    ]

    best = {}
    for k in metric_keys:
        vals = [(m, results[m].get(k, float("-inf"))) for m in methods]
        best[k] = max(vals, key=lambda x: x[1])[0]

    for method in methods:
        cells = [method.replace("_", "\\_")]
        for k in metric_keys:
            val = results[method].get(k, "—")
            if isinstance(val, float):
                s = f"{val:.4f}"
                if best.get(k) == method:
                    s = f"\\textbf{{{s}}}"
                cells.append(s)
            else:
                cells.append(str(val))
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def save_tables(
    results: Dict[str, Dict[str, Any]],
    output_dir: Path,
    prefix: str = "comparison",
) -> None:
    """Save both Markdown and LaTeX tables to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    md = generate_markdown_table(results)
    (output_dir / f"{prefix}.md").write_text(md)

    tex = generate_latex_table(results)
    (output_dir / f"{prefix}.tex").write_text(tex)


if __name__ == "__main__":
    results = {
        "NAS-GA":       {"mean_accuracy": 0.9234, "mean_f1": 0.9200, "param_count": 45000},
        "NSGA-II":      {"mean_accuracy": 0.9180, "mean_f1": 0.9150, "param_count": 32000},
        "RandomSearch": {"mean_accuracy": 0.8900, "mean_f1": 0.8850, "param_count": 120000},
        "Manual":       {"mean_accuracy": 0.9100, "mean_f1": 0.9050, "param_count": 80000},
    }
    print(generate_markdown_table(results))
    print()
    print(generate_latex_table(results))
