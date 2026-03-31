"""
Report Generator
=================
Produce Markdown and LaTeX summary tables from experiment CSVs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

from config import CFG, ensure_dirs

logger = logging.getLogger(__name__)


def _latex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%")


def _make_summary_table(
    df: pd.DataFrame,
    group_cols: List[str],
    value_col: str = "test_accuracy",
) -> pd.DataFrame:
    """Group by *group_cols* and aggregate value_col → mean ± std."""
    agg = df.groupby(group_cols)[value_col].agg(["mean", "std", "count"]).reset_index()
    agg.columns = [*group_cols, f"{value_col}_mean", f"{value_col}_std", "n"]
    agg[f"{value_col}_mean"] = agg[f"{value_col}_mean"].round(4)
    agg[f"{value_col}_std"] = agg[f"{value_col}_std"].round(4)
    return agg


# ── Markdown ──────────────────────────────────────────────────────────


def generate_markdown_report(results_dir: Path | None = None) -> str:
    """Scan results CSVs and produce a combined Markdown report."""
    results_dir = results_dir or Path(CFG.RESULTS_DIR)
    ensure_dirs()

    sections: List[str] = ["# Decision-Tree Study — Summary Report\n"]

    csv_files = sorted(results_dir.glob("exp_*.csv"))
    if not csv_files:
        sections.append("*No experiment results found. Run experiments first.*\n")
        report = "\n".join(sections)
        out = results_dir / "report.md"
        out.write_text(report)
        return report

    for csv_path in csv_files:
        exp_name = csv_path.stem
        df = pd.read_csv(csv_path)
        sections.append(f"## {exp_name}\n")
        sections.append(f"- Rows: {len(df)}")
        sections.append(f"- Columns: {', '.join(df.columns)}\n")

        # Auto-summarise by strategy if present
        if "strategy" in df.columns and "test_accuracy" in df.columns:
            summary = _make_summary_table(df, ["strategy"])
            sections.append(summary.to_markdown(index=False))
            sections.append("")

        # Auto-summarise by dataset if present
        if "dataset" in df.columns and "test_accuracy" in df.columns:
            summary = _make_summary_table(df, ["dataset"])
            sections.append(summary.to_markdown(index=False))
            sections.append("")

    report = "\n".join(sections)
    out = results_dir / "report.md"
    out.write_text(report)
    logger.info("Markdown report saved → %s", out)
    return report


# ── LaTeX ─────────────────────────────────────────────────────────────


def generate_latex_tables(results_dir: Path | None = None) -> str:
    """Generate LaTeX longtable snippets from experiment CSVs."""
    results_dir = results_dir or Path(CFG.RESULTS_DIR)
    ensure_dirs()

    parts: List[str] = []

    csv_files = sorted(results_dir.glob("exp_*.csv"))
    for csv_path in csv_files:
        exp_name = csv_path.stem
        df = pd.read_csv(csv_path)

        if "strategy" in df.columns and "test_accuracy" in df.columns:
            summary = _make_summary_table(df, ["strategy"])
            parts.append(f"% Table: {exp_name} by strategy")
            parts.append(summary.to_latex(index=False, float_format="%.4f"))
            parts.append("")

    latex = "\n".join(parts)
    out = results_dir / "tables.tex"
    out.write_text(latex)
    logger.info("LaTeX tables saved → %s", out)
    return latex


# ── Convenience ───────────────────────────────────────────────────────


def generate_all_reports(results_dir: Path | None = None) -> None:
    """Generate both Markdown and LaTeX reports."""
    generate_markdown_report(results_dir)
    generate_latex_tables(results_dir)
