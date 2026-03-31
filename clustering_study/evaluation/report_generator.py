"""Generate Markdown summary tables from experiment results."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from config import CFG


def _md_table(df: pd.DataFrame, float_fmt: str = ".3f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |"]
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:{float_fmt}}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _latex_table(df: pd.DataFrame, caption: str = "", label: str = "") -> str:
    cols = list(df.columns)
    header = " & ".join(cols) + " \\\\"
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}" if label else "",
        "\\begin{tabular}{" + "l" * len(cols) + "}",
        "\\toprule",
        header,
        "\\midrule",
    ]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def generate_markdown_report(
    csv_dir: Path | None = None,
    metrics: Sequence[str] = (
        "int_silhouette",
        "int_calinski_harabasz",
        "int_davies_bouldin",
        "ext_ari",
        "ext_nmi",
    ),
) -> str:
    csv_dir = csv_dir or CFG.OUTPUT_DIR
    sections: list[str] = ["# Clustering Study — Summary Report\n"]

    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        sections.append("*No CSV result files found.*\n")
        return "\n".join(sections)

    for csv_path in csv_files:
        exp_name = csv_path.stem
        df = pd.read_csv(csv_path)
        sections.append(f"## {exp_name}\n")
        sections.append(f"Rows: {len(df)}, Columns: {len(df.columns)}\n")

        available = [m for m in metrics if m in df.columns]
        if available and "method" in df.columns:
            agg = df.groupby("method")[available].agg(["mean", "std"])
            agg.columns = [f"{m} (μ±σ)" for m, _ in agg.columns[::2] for _ in ("",)]
            summary = df.groupby("method")[available].mean().round(4)
            sections.append(_md_table(summary.reset_index()))
            sections.append("")

    report = "\n".join(sections)
    out = csv_dir / "report.md"
    out.write_text(report, encoding="utf-8")
    return report
