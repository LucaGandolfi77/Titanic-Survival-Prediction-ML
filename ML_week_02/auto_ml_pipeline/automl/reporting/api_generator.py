"""
FastAPI code generator – renders a ready-to-run inference API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..utils.logger import get_logger

logger = get_logger(__name__)

try:
    from jinja2 import Environment, FileSystemLoader
except ImportError:
    Environment = None  # type: ignore

_TEMPLATE_DIR = Path(__file__).parent / "templates"


class APIGenerator:
    """Generate a stand-alone FastAPI app for the trained model."""

    def __init__(self, config) -> None:
        self.output_path: str = getattr(config, "output_path", "outputs/app.py")

    def generate(
        self,
        feature_names: List[str],
        model_path: str,
        task: str,
    ) -> Path:
        if Environment is None:
            raise ImportError("pip install jinja2")

        env = Environment(
            loader=FileSystemLoader(str(_TEMPLATE_DIR)),
            autoescape=False,
        )
        template = env.get_template("api_template.py.j2")

        code = template.render(
            feature_names=feature_names,
            model_path=model_path,
            task=task,
        )

        out = Path(self.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(code, encoding="utf-8")
        logger.info(f"FastAPI app generated → {out}")
        return out
