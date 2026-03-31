"""
Report Generator
=================
Markdown and LaTeX comparison tables from experiment DataFrames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from config import CFG
from evaluation.statistical_tests import pairwise_wilcoxon


def _bold_best(vals: pd.Series, higher_better: bool = True) -> List[str]:
    best = vals.max() if higher_better else vals.min()
    return [f"**{v:.4f}**" if np.isclose(v, best) else f"{v:.4f}" for v in vals]


def _latex_bold_best(vals: pd.Series, higher_better: bool = True) -> List[str]:
    best = vals.max() if higher_better else vals.min()
    return [f"\\textbf{{{v:.4f}}}" if np.isclose(v, best) else f"{v:.4f}" for v in vals]


def summary_table(
    df: pd.DataFrame,
    score_col: str = "test_accuracy",
    method_col: str = "method",
    higher_better: bool = True,
) -> pd.DataFrame:
    agg = df.groupby(method_col)[score_col].agg(["mean", "std", "count"]).reset_index()
    agg = agg.sort_values("mean", ascending=not higher_better).reset_index(drop=True)
    agg["rank"] = range(1, len(agg) + 1)
    return agg


def markdown_table(
    df: pd.DataFrame,
    score_col: str = "test_accuracy",
    method_col: str = "method",
    higher_better: bool = True,
    caption: str = "",
) -> str:
    tbl = summary_table(df, score_col, method_col, higher_better)
    lines = []
    if caption:
        lines.append(f"### {caption}\n")
    lines.append(f"| Rank | Method | {score_col} (mean ± std) |")
    lines.append("|------|--------|" + "-" * 25 + "|")
    for _, row in tbl.iterrows():
        mean_str = f"{row['mean']:.4f}"
        best = tbl["mean"].max() if higher_better else tbl["mean"].min()
        if np.isclose(row["mean"], best):
            mean_str = f"**{mean_str}**"
        lines.append(f"| {int(row['rank'])} | {row[method_col]} | {mean_str} ± {row['std']:.4f} |")
    return "\n".join(lines)


def latex_table(
    df: pd.DataFrame,
    score_col: str = "test_accuracy",
    method_col: str = "method",
    higher_better: bool = True,
    caption: str = "Method comparison",
    label: str = "tab:comparison",
) -> str:
    tbl = summary_table(df, score_col, method_col, higher_better)
    best = tbl["mean"].max() if higher_better else tbl["mean"].min()

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{clcc}",
        r"\toprule",
        f"Rank & Method & {score_col.replace('_', ' ')} & Std \\\\",
        r"\midrule",
    ]
    for _, row in tbl.iterrows():
        m = row["mean"]
        s = row["std"]
        mean_str = f"\\textbf{{{m:.4f}}}" if np.isclose(m, best) else f"{m:.4f}"
        lines.append(f"  {int(row['rank'])} & {row[method_col]} & {mean_str} & {s:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def full_report(
    results: Dict[str, pd.DataFrame],
    output_dir: Path | None = None,
) -> str:
    output_dir = output_dir or CFG.RESULTS_DIR / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    sections = []

    for exp_name, df in results.items():
        score_cols = [c for c in df.columns
                      if c.startswith("test_") and df[c].dtype in (np.float64, float)]
        if not score_cols:
            continue
        score_col = score_cols[0]
        md = markdown_table(df, score_col=score_col, caption=exp_name)
        ltx = latex_table(df, score_col=score_col, caption=exp_name,
                          label=f"tab:{exp_name}")
        sections.append(md)
        (output_dir / f"{exp_name}_table.tex").write_text(ltx, encoding="utf-8")

    report = "\n\n---\n\n".join(sections)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return report


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    rows = []
    for m in ["bagging", "random_forest", "adaboost"]:
        for s in range(15):
            rows.append({"method": m, "seed": s,
                         "test_accuracy": rng.normal(0.85 if m == "random_forest" else 0.82, 0.03)})
    df = pd.DataFrame(rows)
    print(markdown_table(df))
