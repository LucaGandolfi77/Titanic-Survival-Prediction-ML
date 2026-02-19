"""
summary_generator.py – Build executive and technical summary dicts from dashboard state.
"""
from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _safe_round(v: Any, decimals: int = 4) -> Any:
    if isinstance(v, (float, np.floating)):
        return round(float(v), decimals)
    return v


# ── Executive summary ─────────────────────────────────────────

def generate_executive_summary(
    model_info: Dict[str, Any],
    performance: Dict[str, float],
    fairness_results: Optional[pd.DataFrame] = None,
    top_features: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Produce a high-level summary dict ready for template rendering."""
    summary: Dict[str, Any] = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "model_type": model_info.get("model_type", "Unknown"),
        "n_features": model_info.get("n_features", "?"),
        "task": model_info.get("task", "classification"),
    }

    # Performance
    summary["performance"] = {k: _safe_round(v) for k, v in performance.items()}
    primary = performance.get("accuracy") or performance.get("r2") or next(iter(performance.values()), None)
    summary["primary_metric_name"] = "accuracy" if "accuracy" in performance else list(performance.keys())[0] if performance else "N/A"
    summary["primary_metric_value"] = _safe_round(primary)

    # Top features
    summary["top_features"] = (top_features or [])[:10]

    # Fairness
    if fairness_results is not None and not fairness_results.empty:
        n_fail = (fairness_results["status"] == "❌ FAIL").sum()
        n_warn = (fairness_results["status"] == "⚠️ WARNING").sum()
        n_pass = (fairness_results["status"] == "✅ PASS").sum()
        summary["fairness"] = {
            "pass": int(n_pass),
            "warn": int(n_warn),
            "fail": int(n_fail),
            "verdict": "FAIL" if n_fail > 0 else ("WARNING" if n_warn > 0 else "PASS"),
        }
    else:
        summary["fairness"] = None

    return summary


# ── Technical report data ─────────────────────────────────────

def generate_technical_report(
    model_info: Dict[str, Any],
    performance: Dict[str, float],
    shap_global: Optional[Dict[str, float]] = None,
    lime_local: Optional[List[Dict[str, Any]]] = None,
    fairness_results: Optional[pd.DataFrame] = None,
    recommendations: Optional[List[Dict[str, str]]] = None,
    dataset_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Produce a comprehensive report dict for the technical template."""
    report: Dict[str, Any] = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": {
            "type": model_info.get("model_type", "Unknown"),
            "params": model_info.get("params", {}),
            "n_features": model_info.get("n_features"),
            "classes": model_info.get("classes"),
            "task": model_info.get("task", "classification"),
        },
        "performance": {k: _safe_round(v) for k, v in performance.items()},
    }

    # Dataset
    report["dataset"] = dataset_stats or {}

    # SHAP global importance
    if shap_global:
        report["shap_global"] = {k: _safe_round(v) for k, v in
                                  sorted(shap_global.items(), key=lambda x: -abs(x[1]))[:20]}
    else:
        report["shap_global"] = None

    # LIME local explanations sample
    report["lime_examples"] = lime_local[:5] if lime_local else None

    # Fairness
    if fairness_results is not None and not fairness_results.empty:
        report["fairness"] = fairness_results.to_dict(orient="records")
    else:
        report["fairness"] = None

    # Recommendations
    report["recommendations"] = recommendations

    return report


# ── Plain-text summary (for download) ────────────────────────

def to_plain_text(report: Dict[str, Any]) -> str:
    """Render report dict as human-readable plain text."""
    lines = [
        "=" * 60,
        "  EXPLAINABLE AI – MODEL REPORT",
        f"  Generated: {report.get('generated_at', 'N/A')}",
        "=" * 60,
        "",
    ]

    # Model
    m = report.get("model", {})
    lines += [
        "MODEL",
        f"  Type:      {m.get('type')}",
        f"  Features:  {m.get('n_features')}",
        f"  Task:      {m.get('task')}",
        "",
    ]

    # Performance
    lines.append("PERFORMANCE")
    for k, v in report.get("performance", {}).items():
        lines.append(f"  {k:25s} {v}")
    lines.append("")

    # SHAP
    sg = report.get("shap_global")
    if sg:
        lines.append("GLOBAL FEATURE IMPORTANCE (SHAP)")
        for k, v in sg.items():
            lines.append(f"  {k:25s} {v:+.4f}")
        lines.append("")

    # Fairness
    fr = report.get("fairness")
    if fr:
        lines.append("FAIRNESS ANALYSIS")
        for row in fr:
            lines.append(f"  [{row.get('status', '')}]  {row.get('metric', '')} "
                         f"({row.get('protected_attribute', '')}): {row.get('value', '')}")
        lines.append("")

    # Recommendations
    recs = report.get("recommendations")
    if recs:
        lines.append("RECOMMENDATIONS")
        for i, rec in enumerate(recs, 1):
            lines.append(f"  {i}. [{rec.get('severity')}] {rec.get('category')}")
            lines.append(f"     {rec.get('finding')}")
            lines.append(f"     → {rec.get('recommendation')}")
            lines.append("")

    return "\n".join(lines)
