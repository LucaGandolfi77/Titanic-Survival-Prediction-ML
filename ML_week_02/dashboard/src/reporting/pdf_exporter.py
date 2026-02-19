"""
pdf_exporter.py – Render HTML reports and (optionally) export to PDF.

Uses Jinja2 for template rendering.  PDF conversion is optional and
relies on lightweight HTML-to-PDF (xhtml2pdf or weasyprint) only if
the dependency is available.  Otherwise, we simply offer the rendered
HTML for download or viewing in a browser.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATES_DIR = Path(__file__).parent / "templates"


def _get_jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )


# ── HTML rendering ────────────────────────────────────────────

def render_executive_html(summary: Dict[str, Any]) -> str:
    """Render the executive summary template."""
    env = _get_jinja_env()
    tmpl = env.get_template("executive_summary.html")
    return tmpl.render(summary=summary, now=datetime.datetime.now())


def render_technical_html(report: Dict[str, Any]) -> str:
    """Render the full technical report template."""
    env = _get_jinja_env()
    tmpl = env.get_template("technical_report.html")
    return tmpl.render(report=report, now=datetime.datetime.now())


# ── PDF export (best-effort) ─────────────────────────────────

def html_to_pdf(html: str, output_path: Path) -> bool:
    """Try converting HTML to PDF.  Returns True on success."""
    try:
        from xhtml2pdf import pisa  # type: ignore[import-untyped]

        with open(output_path, "wb") as f:
            status = pisa.CreatePDF(html, dest=f)
        return not status.err
    except ImportError:
        pass

    try:
        from weasyprint import HTML as WHTML  # type: ignore[import-untyped]

        WHTML(string=html).write_pdf(str(output_path))
        return True
    except ImportError:
        pass

    # Fallback: save as HTML
    output_path = output_path.with_suffix(".html")
    output_path.write_text(html, encoding="utf-8")
    return False


# ── Convenience wrappers ─────────────────────────────────────

def export_executive_report(
    summary: Dict[str, Any],
    output_dir: Path,
    filename: str = "executive_summary",
) -> Path:
    """Render + export executive summary.  Returns path to output file."""
    html = render_executive_html(summary)
    pdf_path = output_dir / f"{filename}.pdf"
    success = html_to_pdf(html, pdf_path)
    if success:
        return pdf_path
    return pdf_path.with_suffix(".html")


def export_technical_report(
    report: Dict[str, Any],
    output_dir: Path,
    filename: str = "technical_report",
) -> Path:
    """Render + export technical report.  Returns path to output file."""
    html = render_technical_html(report)
    pdf_path = output_dir / f"{filename}.pdf"
    success = html_to_pdf(html, pdf_path)
    if success:
        return pdf_path
    return pdf_path.with_suffix(".html")
