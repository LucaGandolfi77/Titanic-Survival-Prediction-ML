"""
conftest.py – Shared fixtures for the XAI dashboard test suite.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the project root is on PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
