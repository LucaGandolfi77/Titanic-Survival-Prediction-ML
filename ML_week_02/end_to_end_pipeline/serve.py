#!/usr/bin/env python
"""
serve.py — CLI Entry Point for the Prediction API
===================================================
Usage:
    python serve.py                        # Start on default host:port
    python serve.py --port 9000            # Override port
    python serve.py --reload               # Auto-reload for development
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn

from src.utils.config_loader import load_config


def main() -> None:
    cfg = load_config()
    serve_cfg = cfg["serving"]

    parser = argparse.ArgumentParser(description="Start the Titanic prediction API")
    parser.add_argument("--host", default=serve_cfg["host"], help="Bind host")
    parser.add_argument("--port", type=int, default=serve_cfg["port"], help="Bind port")
    parser.add_argument("--reload", action="store_true", default=serve_cfg.get("reload", False))
    args = parser.parse_args()

    uvicorn.run(
        "src.serving.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=serve_cfg.get("log_level", "info"),
    )


if __name__ == "__main__":
    main()
